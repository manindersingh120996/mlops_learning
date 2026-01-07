# Dockerfile

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git libgl1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch (CPU version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir python-multipart

# Copy application code
COPY inference_serving/inference_serving_docker.py /app/
COPY inference_serving/src /app/src

# Copy the pre-downloaded model artifacts
# IMPORTANT: Copy the entire model directory, not just MLmodel
# COPY inference_serving/model_artifacts/ /app/model_artifacts/
COPY inference_serving/model_artifacts/MLmodel/ /app/model_artifacts/

# Configure environment
ENV MODEL_PATH=/app/model_artifacts/MLmodel/
ENV BATCH_SIZE=8
ENV MAX_WAIT_MS=50
ENV NUM_WORKER_THREADS=2
ENV PROM_PORT=8801

# Expose ports
# Note: Cloud Run will set PORT environment variable dynamically
EXPOSE 8000 8801

# Health check - CRITICAL: Use correct endpoint path
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health_check || exit 1

# CRITICAL: Read PORT from environment variable that Cloud Run sets
CMD uvicorn inference_serving_docker:app --host 0.0.0.0 --port ${PORT:-8000}
