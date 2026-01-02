import os
import time
import asyncio
import logging
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import mlflow.pytorch
import torch
from prometheus_client import start_http_server, Summary, Counter, Gauge
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from PIL import Image
from torchvision import transforms
import io


LOG = logging.getLogger("uvicorn.error")
LOG.setLevel(logging.INFO)


# Configuration from environment variables
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model_artifacts")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
MAX_WAIT_MS = int(os.environ.get("MAX_WAIT_MS", "50"))
NUM_WORKER_THREADS = int(os.environ.get("NUM_WORKER_THREADS", "2"))
PROM_PORT = int(os.environ.get("PROM_PORT", "8801"))


# Prometheus metrics
REQUEST_TIME = Summary("inference_latency_seconds", "Inference latency in seconds")
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")
QUEUE_SIZE = Gauge("inference_queue_size", "Current inference queue size")
MODEL_LOAD_TIME = Gauge("model_load_time_seconds", "Time taken to load model")


# Response models
class PredictResponse(BaseModel):
    """Prediction response"""
    class_id: int = Field(..., description="Predicted class ID")
    class_probabilities: List[float] = Field(..., description="Probability for each class")
    confidence: float = Field(..., description="Confidence score (max probability)")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_path: str


class QueueItem:
    """Item in the inference queue"""
    def __init__(self, input_tensor: torch.Tensor, fut: asyncio.Future):
        self.input_tensor = input_tensor
        self.fut = fut


# Global state
model = None
device = torch.device('cpu')
inference_queue: asyncio.Queue = None
batch_worker_task: asyncio.Task = None
executor = None

# Image preprocessing transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


def load_model():
    """Load model from baked-in artifacts"""
    global model, device
    
    load_start = time.time()
    
    try:
        LOG.info(f"Loading model from: {MODEL_PATH}")
        
        # Check if model path exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")
        
        # List contents for debugging
        LOG.info(f"Model directory contents: {os.listdir(MODEL_PATH)}")
        
        # Load model from local path
        model = mlflow.pytorch.load_model(MODEL_PATH)
        
        # Determine device (GPU if available, otherwise CPU)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            LOG.info("CUDA available - using GPU")
        else:
            device = torch.device('cpu')
            LOG.info("CUDA not available - using CPU")
        
        # Move model to device and set to eval mode
        model.to(device)
        model.eval()
        
        load_time = time.time() - load_start
        MODEL_LOAD_TIME.set(load_time)
        
        LOG.info(f"✅ Model loaded successfully in {load_time:.2f}s on device: {device}")
        
    except Exception as e:
        LOG.error(f"❌ Failed to load model: {str(e)}")
        LOG.error(f"Model path: {MODEL_PATH}")
        raise


async def batch_worker():
    """Worker that processes inference requests in batches"""
    LOG.info("Starting Batch Worker")
    
    while True:
        try:
            # Wait for first item
            first_item: QueueItem = await inference_queue.get()
            items = [first_item]

            # Try to collect more items for batching
            start = time.time()
            elapsed = 0
            
            while len(items) < BATCH_SIZE and elapsed * 1000 < MAX_WAIT_MS:
                try:
                    item = inference_queue.get_nowait()
                    items.append(item)
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.005)  # Small sleep to prevent tight loop
                elapsed = time.time() - start
            
            LOG.debug(f"Processing batch of {len(items)} items")
            
            # Combine tensors into batch
            inputs = torch.cat([it.input_tensor for it in items], dim=0).to(device)

            # Run inference in thread pool (to not block event loop)
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(executor, run_inference_on_device, inputs)
            outputs = await fut

            # Set results for all futures
            for out, it in zip(outputs, items):
                if not it.fut.done():
                    it.fut.set_result(out)
            
            # Update queue size metric
            QUEUE_SIZE.set(inference_queue.qsize())
        
        except asyncio.CancelledError:
            LOG.info("Batch Worker Cancelled")
            break
        except Exception as e:
            LOG.exception(f"Error in batch_worker: {e}")


def run_inference_on_device(batched_input: torch.Tensor):
    """
    Run inference on device (called from thread pool)
    
    Args:
        batched_input: Batched input tensor [batch_size, C, H, W]
    
    Returns:
        List of prediction results
    """
    with torch.no_grad():
        # Forward pass
        output = model(batched_input)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(output, dim=1)
        
        # Get predictions and confidences
        confidences, predictions = torch.max(probs, dim=1)
        
        # Prepare results
        results = []
        for i in range(len(batched_input)):
            results.append({
                'class_id': predictions[i].item(),
                'class_probabilities': probs[i].cpu().numpy().tolist(),
                'confidence': confidences[i].item()
            })
        
        return results


def process_image_bytes(content: bytes) -> torch.Tensor:
    """
    Process image bytes into tensor
    
    Args:
        content: Raw image bytes
    
    Returns:
        Preprocessed tensor ready for inference
    """
    img = Image.open(io.BytesIO(content)).convert("RGB")
    tensor = image_transform(img).unsqueeze(0)  # Add batch dimension
    return tensor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global inference_queue, batch_worker_task, executor

    # Start Prometheus metrics server
    start_http_server(PROM_PORT)
    LOG.info(f"📊 Prometheus metrics server started on port {PROM_PORT}")
    
    # Load model
    load_model()

    # Initialize inference queue and worker
    inference_queue = asyncio.Queue()
    executor = ThreadPoolExecutor(max_workers=NUM_WORKER_THREADS)
    batch_worker_task = asyncio.create_task(batch_worker())

    LOG.info("🚀 Startup Complete - Ready to serve requests")
    
    yield  # Application runs here
    
    # Shutdown
    LOG.info("🛑 Shutting Down")
    
    if batch_worker_task:
        batch_worker_task.cancel()
        try:
            await batch_worker_task
        except asyncio.CancelledError:
            pass
    
    if executor:
        executor.shutdown(wait=True)

    LOG.info("✅ Shutdown Complete")


# Create FastAPI app
app = FastAPI(
    title="ViT Inference API",
    description="Vision Transformer model inference service with batching support",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Perform image classification inference
    
    Args:
        file: Uploaded image file
    
    Returns:
        Prediction results with class probabilities
    """
    REQUEST_COUNT.inc()
    start_time = time.time()

    # Read and process image
    content = await file.read()
    try:
        tensor = process_image_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    
    # Create future for result
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    item = QueueItem(tensor, fut)

    # Add to inference queue
    await inference_queue.put(item)
    QUEUE_SIZE.set(inference_queue.qsize())

    # Wait for result with timeout
    try:
        result = await asyncio.wait_for(fut, timeout=10.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")
    
    # Record metrics
    latency = time.time() - start_time
    REQUEST_TIME.observe(latency)
    
    return PredictResponse(
        class_id=result['class_id'],
        class_probabilities=result['class_probabilities'],
        confidence=result['confidence'],
        inference_time_ms=latency * 1000
    )


@app.get("/health_check", response_model=HealthResponse)
async def healthz():
    """
    Health check endpoint
    
    Returns:
        Health status
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path=MODEL_PATH
    )


@app.get("/readyz")
async def readyz():
    """
    Readiness check endpoint
    
    Returns:
        Readiness status
    """
    ready = model is not None and inference_queue is not None
    
    if not ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "ready": True,
        "model_loaded": model is not None,
        "queue_initialized": inference_queue is not None,
        "batch_worker_running": batch_worker_task is not None and not batch_worker_task.done()
    }


@app.get("/metrics-info")
async def metrics_info():
    """
    Information about available metrics
    
    Returns:
        Metrics endpoint information
    """
    return {
        "prometheus_endpoint": f"http://localhost:{PROM_PORT}/metrics",
        "available_metrics": [
            "inference_latency_seconds",
            "inference_requests_total",
            "inference_queue_size",
            "model_load_time_seconds"
        ]
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "ViT Inference API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/healthz",
            "readiness": "/readyz",
            "metrics": f"http://localhost:{PROM_PORT}/metrics",
            "docs": "/docs"
        }
    }