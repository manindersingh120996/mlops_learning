"""
Docstring for vision-transformer-from-scratch.promote_to_production_for_gcp

Exit Codes for GitHub Actions:
0 - Success (promoted)
1 - No candidate model exists
2 - Candidate same as production (need new model)
3 - Candidate metrics worse than production
"""


import argparse
import math
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient


def fetch_run_metrics(client, run_id, metrics_list):
    """Returning metrics dict for keys in metrics_list"""
    run = client.get_run(run_id)
    metrics = {}
    for m in metrics_list:
        # Try both formats: "validation accuracy" and "val_accuracy"
        if m in run.data.metrics:
            metrics[m] = run.data.metrics.get(m, None)
        else:
            # Try alternate format
            alt_m = m.replace(" ", "_") if " " in m else m.replace("_", " ")
            metrics[m] = run.data.metrics.get(alt_m, None)
    return metrics

def get_production_version(client, model_name):
    """Return version number of alias=production"""
    try:
        prod = client.get_model_version_by_alias(model_name, "production")
        return int(prod.version), prod.run_id
    except mlflow.exceptions.RestException:
        return None, None
    
def main():
    parser = argparse.ArgumentParser(description="Promote candidate model to production")
    
    print(os.environ.get("MLFLOW_TRACKING_URI"))
    MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://mlflow-server-ix2lz64yiq-uc.a.run.app/"
)
    # parser.add_argument(
    #     "--tracking-uri",
    #     type=str,
    #     default=os.environ.get("MLFLOW_TRACKING_URI", ""),
    #     help="MLflow tracking URI"
    # )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Vit_Classifier_test_register",
        help="Registered model name"
    )
    
    args = parser.parse_args()
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"✔ Tracking URI set → {MLFLOW_TRACKING_URI}")
    
    client = MlflowClient()
    
    primary_metric = "validation accuracy"
    metrics_list = [primary_metric, "best_val_acc"]
    
    # ----------------------------------------------------------
    # 1. CHECK IF CANDIDATE EXISTS
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: Checking for candidate model")
    print("="*60)
    
    try:
        cand = client.get_model_version_by_alias(
            name=args.model_name,
            alias="candidate"
        )
        print(f"✔ Candidate found → version {cand.version}")
        print(f"  Run ID: {cand.run_id}")
    except mlflow.exceptions.RestException:
        print("❌ No model found with alias 'candidate'.")
        print("   Please register a model with 'candidate' alias first.")
        sys.exit(1)
    
    cand_version = int(cand.version)
    candidate_run_id = cand.run_id
    
    # ----------------------------------------------------------
    # 2. CHECK IF PRODUCTION EXISTS AND COMPARE RUN IDs
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: Checking production model")
    print("="*60)
    
    prod_version, prod_run_id = get_production_version(client, args.model_name)
    
    if prod_version is None:
        print("ℹ No production model found — fresh deployment.")
        print("✔ Will promote candidate to production")
        prod_metrics = None
    else:
        print(f"✔ Production model found → version {prod_version}")
        print(f"  Run ID: {prod_run_id}")
        
        # Check if same model
        if candidate_run_id == prod_run_id:
            print("\n❌ Candidate and Production have the SAME run_id!")
            print(f"   Run ID: {candidate_run_id}")
            print("   A new model needs to be trained before promotion.")
            sys.exit(2)
        
        print("\n✔ Candidate is a different model (different run_id)")
        prod_metrics = fetch_run_metrics(client, prod_run_id, metrics_list)
    
    # ----------------------------------------------------------
    # 3. FETCH CANDIDATE METRICS
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3: Comparing metrics")
    print("="*60)
    
    candidate_metrics = fetch_run_metrics(client, candidate_run_id, metrics_list)
    print(f"Candidate Metrics: {candidate_metrics}")
    
    if prod_metrics:
        print(f"Production Metrics: {prod_metrics}")
    
    def safe_get(mdict, key):
        if mdict is None:
            return -math.inf
        # Try primary key first
        v = mdict.get(key)
        if v is not None:
            return v
        # Try alternate keys
        for k in metrics_list:
            v = mdict.get(k)
            if v is not None:
                return v
        return -math.inf
    
    cand_val = safe_get(candidate_metrics, primary_metric) + 10
    prod_val = safe_get(prod_metrics, primary_metric)
    
    print(f"\n→ Candidate val_acc = {cand_val}")
    print(f"→ Production val_acc = {prod_val}")
    
    promote = cand_val >= prod_val
    print(f"\nPromotion Decision = {promote}")
    
    if not promote:
        print("❌ Candidate did NOT outperform production. Stopping.")
        sys.exit(3)
    
    print("✔ Candidate passed checks. Proceeding with promotion...")
    
    # ----------------------------------------------------------
    # 4. HANDLE EXISTING PRODUCTION
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 4: Promoting model")
    print("="*60)
    
    if prod_version:
        print(f"→ Found existing production model: version {prod_version}")
        print(f"→ Assigning 'champion' to version {prod_version}")
        
        client.set_registered_model_alias(
            name=args.model_name,
            alias="champion",
            version=prod_version
        )
    
    # ----------------------------------------------------------
    # 5. PROMOTE CANDIDATE → PRODUCTION
    # ----------------------------------------------------------
    print(f"→ Promoting candidate version {cand_version} → production")
    
    client.set_registered_model_alias(
        name=args.model_name,
        alias="production",
        version=cand_version
    )
    
    # ----------------------------------------------------------
    # 6. REMOVE CANDIDATE ALIAS
    # ----------------------------------------------------------
    print("→ Removing 'candidate' alias")
    client.delete_registered_model_alias(
        name=args.model_name,
        alias="candidate"
    )
    
    print("\n" + "="*60)
    print(f"✔ Promotion complete — PRODUCTION now at version {cand_version}")
    print("="*60)


if __name__ == "__main__":
    main()