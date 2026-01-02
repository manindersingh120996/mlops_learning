import argparse
import math
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import hydra
from omegaconf import DictConfig, OmegaConf


def fetch_run_metrics(client, run_id, metrics_list):
    """Returning metrics dict for keys in metrics_list"""
    run = client.get_run(run_id)
    metrics = {}
    for m in metrics_list:
        metrics[m] = run.data.metrics.get(m, None)
    return metrics


def get_production_version(client, model_name):
    """Return version number of alias=production"""
    try:
        prod = client.get_model_version_by_alias(model_name, "production")
        return int(prod.version), prod.run_id
    except mlflow.exceptions.RestException:
        return None, None


@hydra.main(config_path='configs', config_name='staging', version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    run_id = cfg.model_params.candidate_run_id
    if not run_id:
        raise ValueError("❌ Run ID not provided")

    mlflow.set_tracking_uri(cfg.mlflow_params.tracking_uri)
    print(f"✔ Tracking URI set → {cfg.mlflow_params.tracking_uri}")

    client = MlflowClient()

    primary_metric = "validation accuracy"
    metrics_list = [primary_metric]

    # ----------------------------------------------------------
    # 1. FETCH CANDIDATE METRICS
    # ----------------------------------------------------------
    candidate_metrics = fetch_run_metrics(client, run_id, metrics_list) 
    print("Candidate Metrics:", candidate_metrics)

    try:
        cand = client.get_model_version_by_alias(
            name=cfg.model_params.model_name,
            alias="candidate"
        )
    except mlflow.exceptions.RestException:
        raise ValueError("❌ No model found with alias 'candidate'.")

    cand_version = int(cand.version)
    print(f"✔ Candidate alias found → version {cand_version}")

    # ----------------------------------------------------------
    # 2. FETCH PRODUCTION METRICS
    # ----------------------------------------------------------
    prod_version, prod_run_id = get_production_version(
        client, cfg.model_params.model_name
    )

    if prod_version is None:
        print("ℹ No production model found — fresh deployment.")
        prod_metrics = None
    else:
        prod_metrics = fetch_run_metrics(client, prod_run_id, metrics_list)
        print(f"Production v{prod_version} metrics:", prod_metrics)

    def safe_get(mdict, key):
        v = mdict.get(key) if mdict else None
        return v if v is not None else -math.inf

    cand_val = safe_get(candidate_metrics, primary_metric) + 10.2
    prod_val = safe_get(prod_metrics, primary_metric)

    print(f"→ Candidate val_acc = {cand_val}")
    print(f"→ Production val_acc = {prod_val}")

    promote = cand_val >= prod_val
    print(f"Promotion Decision = {promote}")

    if not promote:
        print("❌ Candidate did NOT outperform production. Stopping.")
        return

    print("✔ Candidate passed checks. Proceeding with promotion...")

    # ----------------------------------------------------------
    # 3. HANDLE EXISTING PRODUCTION
    # ----------------------------------------------------------
    if prod_version:
        print(f"→ Found existing production model: version {prod_version}")
        print(f"→ Assigning 'champion' to version {prod_version}")

        client.set_registered_model_alias(
            name=cfg.model_params.model_name,
            alias="champion",
            version=prod_version
        )

    # ----------------------------------------------------------
    # 4. PROMOTE CANDIDATE → PRODUCTION
    # ----------------------------------------------------------
    print(f"→ Promoting candidate version {cand_version} → production")

    client.set_registered_model_alias(
        name=cfg.model_params.model_name,
        alias="production",
        version=cand_version
    )

    # ----------------------------------------------------------
    # 5. REMOVE CANDIDATE ALIAS (IMPORTANT)
    # ----------------------------------------------------------
    print("→ Removing 'candidate' alias")
    client.delete_registered_model_alias(
        name=cfg.model_params.model_name,
        alias="candidate"
    )

    print(f"✔ Promotion complete — PRODUCTION now at version {cand_version}")


if __name__ == "__main__":
    main()
