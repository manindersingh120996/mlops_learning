"""
Script to download MLflow model to local directory for Docker build
Run this ONCE before building Docker image
"""
import os
import mlflow
import mlflow.pytorch

# Configuration
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "Vit_Classifier_test_register")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")
OUTPUT_DIR = "./model_artifacts"

def download_model():
    """Download model from MLflow registry to local directory"""
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Model: {MODEL_NAME}@{MODEL_ALIAS}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Download model
    print("\n🔄 Downloading model from MLflow...")
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    
    try:
        # This downloads the model to OUTPUT_DIR
        model = mlflow.pytorch.load_model(model_uri, dst_path=OUTPUT_DIR)
        print(f"✅ Model successfully downloaded to: {OUTPUT_DIR}")
        print(f"\n📁 Contents of {OUTPUT_DIR}:")
        for root, dirs, files in os.walk(OUTPUT_DIR):
            level = root.replace(OUTPUT_DIR, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f'{subindent}{file}')
        
        print("\n✅ Ready to build Docker image!")
        print(f"   Next step: docker build -t vit-inference -f inference_serving/Dockerfile .")
        
# When you train a new model, just:
# python download_model.py  # Downloads latest model
# docker build -t vit-inference:v2.0 .  # Rebuilds with new model


    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure MLflow server is running: mlflow server --host 0.0.0.0 --port 5000")
        print("2. Check model exists: mlflow models list")
        print(f"3. Verify model alias: Check if '{MODEL_ALIAS}' alias exists for '{MODEL_NAME}'")
        raise

if __name__ == "__main__":
    download_model()