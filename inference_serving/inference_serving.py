import os
import time
import io
import asyncio
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import mlflow
import mlflow.pytorch
import torch
import numpy as np
from prometheus_client import start_http_server, Summary, Counter, Gauge
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager


LOG = logging.getLogger("uvicorn.error")
LOG.setLevel(logging.INFO)


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI","http://localhost:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "Vit_Classifier_test_register")  # registered model name
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
MAX_WAIT_MS = int(os.environ.get("MAX_WAIT_MS", "50"))  # max time to wait for batching
NUM_WORKER_THREADS = int(os.environ.get("NUM_WORKER_THREADS", "2"))


# Prometheus metrics (exposed on /metrics by client lib)
REQUEST_TIME = Summary("inference_latency_seconds", "Inference latency in seconds")
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")
QUEUE_SIZE = Gauge("inference_queue_size", "Current inference queue size")

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# NEW: Option to use local model path (for baked-in models)
USE_LOCAL_MODEL = os.environ.get("USE_LOCAL_MODEL", "false").lower() == "true"
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/app/model_artifacts")


# request response models
# class PredictResponse(BaseModel):
#     probs: List[float]
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
    model_name: str
    model_alias: str


class QueueItem:
    def __init__(self, input_tensor, fut: asyncio.Future):
        self.input_tensor = input_tensor
        self.fut = fut

model = None
device = torch.device('cpu')
inference_queue: asyncio.Queue = None
batch_worker_task: asyncio.Task = None
executor = None

# def load_model():
#     global model,device
#     LOG.info(f"Loading model from Mlflow: models:/{MODEL_NAME}@{MODEL_ALIAS}")
#     # Loads the model from the directory downloaded during the Docker build
#     MODEL_LOCAL_PATH = "/app/model_artifacts/data/model" # Adjust path as needed
#     # The actual PyTorch model file is often nested, e.g., in a 'data/model' folder
#     # model = mlflow.pytorch.load_model(MODEL_LOCAL_PATH)

#     # ==========
#     model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
#     # ================


#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')

#     model.to(device)
#     model.eval()
#     LOG.info(f"Model loaded and moved to device: {device}")

def load_model():
    """
    Load model either from MLflow registry (runtime) or local path (baked-in)
    """
    global model, device
    
    try:
        if USE_LOCAL_MODEL and os.path.exists(LOCAL_MODEL_PATH):
            LOG.info(f"Loading model from local path: {LOCAL_MODEL_PATH}")
            model = mlflow.pytorch.load_model(LOCAL_MODEL_PATH)
        else:
            LOG.info(f"Loading model from MLflow: models:/{MODEL_NAME}@{MODEL_ALIAS}")
            LOG.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
            
            # CRITICAL FIX: Set backend store explicitly for artifact resolution
            # This tells MLflow to use HTTP for artifact downloads
            os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
            
            # Download to temporary location first
            model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
            
            # This will force MLflow to download via HTTP if tracking server supports it
            model = mlflow.pytorch.load_model(
                model_uri,
                dst_path="/tmp/mlflow_model"  # Explicit temp path
            )
            LOG.info("Model successfully downloaded and loaded")
        
        # Move to appropriate device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        model.to(device)
        model.eval()
        LOG.info(f"Model loaded and moved to device: {device}")
        
    except Exception as e:
        LOG.error(f"Failed to load model: {str(e)}")
        LOG.error(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
        LOG.error(f"Model Name: {MODEL_NAME}, Alias: {MODEL_ALIAS}")
        raise

async def batch_worker():
    LOG.info("Starting Batch Worker")
    while True:
        try:
            first_item : QueueItem = await inference_queue.get()
            items = [first_item]

            start = time.time()
            elapsed = 0
            while len(items) < BATCH_SIZE and elapsed * 1000 < MAX_WAIT_MS :
                try:
                    item = inference_queue.get_nowait()
                    items.append(item)
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.005)
                elapsed = time.time() - start
            
            inputs = torch.cat([it.input_tensor for it in items], dim= 0).to(device)

            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(executor, run_inference_on_device, inputs)
            outputs = await fut

            for out,it in zip(outputs, items):
                if not it.fut.done():
                    it.fut.set_result(out)
            
            QUEUE_SIZE.set(inference_queue.qsize())
        
        except asyncio.CancelledError:
            LOG.info("Batch Worker Cancelled")
            break
        except Exception as e:
            LOG.exception("Error in batch_worker: %s",e)

def run_inference_on_device(batched_input: torch.Tensor):
    # with torch.no_grad():
    #     output = model(batched_input)
    #     probs = torch.softmax(output, dim=1).cpu().numpy()
    # return probs
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



    
@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_queue, batch_worker_task,executor

    prom_port = int(os.environ.get("PROM_PORT","8801"))
    start_http_server(prom_port)
    LOG.info(f"Prometheous metrics server started at port {prom_port}")
    load_model()

    inference_queue = asyncio.Queue()
    executor = ThreadPoolExecutor(max_workers=NUM_WORKER_THREADS)
    batch_worker_task = asyncio.create_task(batch_worker())


    LOG.info("Startup Complete")
    yield

    LOG.info("Shutting Down")
    if batch_worker_task:
        batch_worker_task.cancel()
    if executor:
        executor.shutdown(wait=True)

    LOG.info("Shutdown Complete...")



app = FastAPI(title="ViT Inference API",lifespan=lifespan)

def process_image_bytes(content: bytes):
    from PIL import Image
    from torchvision import transforms
    import io
    img = Image.open(io.BytesIO(content)).convert("RGB")
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
    ])
    t = transform(img).unsqueeze(0)
    return t

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    REQUEST_COUNT.inc()
    start_time = time.time()

    content = await file.read()
    try:
        tensor = process_image_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Image : {e}")
    
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    item = QueueItem(tensor, fut)

    await inference_queue.put(item)
    QUEUE_SIZE.set(inference_queue.qsize())

    try:
        result = await asyncio.wait_for(fut, timeout=10.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference Timeout")
    
    REQUEST_TIME.observe(time.time() - start_time)
    latency = time.time() - start_time
    return PredictResponse(
            class_id=result['class_id'],
            class_probabilities=result['class_probabilities'],
            confidence=result['confidence'],
            inference_time_ms=latency * 1000
        )

@app.get("/healthz")
def healthz():
    return {"status":"ok"}

@app.get("/readyz")
def readyz():
    ok = model is not None and inference_queue is not None
    return {"ready": ok}