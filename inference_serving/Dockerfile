FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install Python dependencies
COPY requirements.txt /app/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir python-multipart

# Copy application code
COPY inference_serving/inference_serving_docker.py /app/
COPY inference_serving/src /app/src

# CRITICAL: Copy the pre-downloaded model artifacts
# This directory should exist after running download_model.py
COPY inference_serving/model_artifacts/MLmodel/ /app/model_artifacts/

# Configure to use the local model
ENV USE_LOCAL_MODEL=true
ENV LOCAL_MODEL_PATH=/app/model_artifacts
ENV MODEL_NAME=Vit_Classifier_test_register
ENV MODEL_ALIAS=production

# Expose ports
EXPOSE 8000 8801

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz').read()"

CMD ["uvicorn", "inference_serving_docker:app", "--host", "0.0.0.0", "--port", "8000"]

############ DOCKER RUNNING COMMAND
# docker run -p 8000:8000 -p 8801:8801 ^
# -e BATCH_SIZE=16 ^
# -e NUM_WORKER_THREADS=4 ^
# -e MAX_WAIT_MS=100 ^
# vit-inference