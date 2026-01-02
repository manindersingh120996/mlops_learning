import mlflow
from mlflow.tracking import MlflowClient
import argparse
from pathlib import Path
import os

# client = MlflowClient()
def model_register(run_id,model_name):
    client = MlflowClient()
    # print(run_id)

    model_src = f"runs:/{run_id}/model"

    print(f"[INFO] Registering model from: {model_src}")
    try:
        client.get_registered_model(model_name)
        print(f"[INFO] Registered model '{model_name}' already exists.")
    except:
        print(f"[INFO] Registered model '{model_name}' NOT found. Creating it...")
        client.create_registered_model(model_name)

    version = client.create_model_version(
        name = model_name,
        source=model_src,
        run_id=run_id
    )
    latest_versions = client.get_latest_versions(model_name)
    latest_version = latest_versions[-1].version     # last = highest version


    client.set_registered_model_alias(name = model_name,alias="candidate",version=latest_version)
    
    print(f"[SUCCESS] Model registered: name={model_name}, version={version.version}")
    return version.version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",required=True)
    parser.add_argument("--model_name",default="Vit_Classifier_test_register")
    parser.add_argument("--run_id")
    print(os.environ.get("MLFLOW_TRACKING_URI"))
    MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://mlflow-server-ix2lz64yiq-uc.a.run.app/"
)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    arg = parser.parse_args()
    print(arg)
    print(Path(arg.model_path))
    if arg.run_id:
        run_id = arg.run_id
        print("run ID is provided...")

    try:
        run_id = ("./outputs"/ Path(arg.model_path) / "run_id.txt").read_text()
        print(("./outputs"/ Path(arg.model_path) / "run_id.txt").read_text())
    except Exception as e:
        print(f"Provided path do not have run_id.txt in it, Please add argurment '--run_id' to explicitly provide the run ID of {arg.model_path} run.")
    
        
    if arg.run_id and arg.run_id != run_id:
        raise ValueError("Provided path's run_id not matching with provided argument's run_ID")
    
    print(f"using '{arg.model_name}' to register the model...")

    model_register(run_id, arg.model_name)




