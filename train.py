import torch
import torch.nn as nn
import math
from torchinfo import summary
from torch.utils.data import DataLoader, Dataset
import hydra
from hydra.core.hydra_config import HydraConfig
import os
import json
import pandas as pd
import logging
from omegaconf import DictConfig, OmegaConf
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from pathlib import Path

from src import model, dataset
import mlflow

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("✅ Setting Device as CUDA...")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("🫡  Device is set to MPS...")
    device = torch.device('mps')
else:
    print("No accelerator available 🥺 ...using CPU for this task...")
    device = torch.device('cpu')

log = logging.getLogger(__name__)


def dataset_creation(cfg: DictConfig):
    train_path = cfg.train_dataset_path
    val_path = cfg.test_dataset_path
    val_transform = dataset.val_transform
    train_transform = dataset.train_transform
    
    train_dataset = dataset.TomAndJerryDataset(
        dataset_path=train_path,
        transform=train_transform
    )
    val_dataset = dataset.TomAndJerryDataset(
        dataset_path=val_path,
        transform=val_transform
    )
    
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_dataset, train_data_loader, val_dataset, val_data_loader


def saving_training_plots(history_df, lr, output_dir):
    """Save training plots"""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].plot(history_df["train loss"], label="Train Loss")
    ax[0].plot(history_df["val loss"], label="Validation Loss")
    ax[0].set_title("Loss Curves")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(history_df["train acc"], label="Train Accuracy")
    ax[1].plot(history_df["val acc"], label="Validation Accuracy")
    ax[1].set_title("Accuracy Curves")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    ax[2].plot(lr, label="Learning rate per step")
    ax[2].set_title("Learning Rate Curve")
    ax[2].set_xlabel("steps")
    ax[2].legend()

    plot_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    mlflow.log_artifact(plot_path)
    log.info(f"Saved training plot to {plot_path}")
    return plot_path


def accuracy_fn(y_pred, y_true):
    """Calculate accuracy"""
    preds = torch.argmax(y_pred, dim=1)
    correct = (preds == y_true).sum().item()
    acc = correct / y_true.size(0)
    return acc


@hydra.main(config_path='configs', config_name='config', version_base=None)
def main_loop(cfg: DictConfig):
    # Set MLflow tracking URI
    # mlflow.set_tracking_uri("http://localhost:5000")
    MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://mlflow-server-ix2lz64yiq-uc.a.run.app/"
)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment("Learning ML Flow with ViT")
    except Exception:
        experiment = mlflow.get_experiment_by_name("Learning ML Flow with ViT")
        experiment_id = experiment.experiment_id
        mlflow.set_experiment("Learning ML Flow with ViT")

    with mlflow.start_run(run_name=f"run-{HydraConfig.get().runtime.output_dir}") as run:
        output_dir = HydraConfig.get().runtime.output_dir
        run_id = run.info.run_id
        
        # Log tags
        mlflow.set_tag("hydra.output_dir", output_dir)
        mlflow.set_tag("script", "train.py")
        mlflow.set_tag("framework", "PyTorch")
        mlflow.set_tag("model", "VisionTransformer")

        log.info(f"All artifacts will be saved to: {output_dir}")
        log.info(f"MLflow Run ID: {run_id}")

        # Log configuration
        mlflow.log_text(OmegaConf.to_yaml(cfg), "config_used.yaml")

        # Flatten and log parameters
        def flatten(d, parent_key="", sep="."):
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten(v, new_key, sep=sep))
                else:
                    items[new_key] = v
            return items

        params = OmegaConf.to_container(cfg, resolve=True)
        flat_params = flatten(params)
        mlflow.log_params(flat_params)

        log.info("--- Configuration ---")
        log.info(f"\n{OmegaConf.to_yaml(cfg)}")
        log.info("---------------------")

        # Create datasets
        log.info("Dataset Creation Begin")
        train_dataset, train_data_loader, val_dataset, val_data_loader = dataset_creation(cfg=cfg.train)
        num_of_classes_in_dataset = len(train_dataset.classes)
        log.info("Dataset Created")

        # Verify configuration
        log.info(f"Dataset Classes: {train_dataset.class_to_index}")
        try:
            assert num_of_classes_in_dataset == cfg.model.number_of_classes, \
                f"Mismatch: config expects {cfg.model.number_of_classes} classes, but dataset has {num_of_classes_in_dataset}."
            log.info("✅ Configuration verification successful.")
        except AssertionError as e:
            log.error(f"CONFIGURATION ERROR: {e}")
            import sys
            sys.exit(1)

        # Save class mapping
        index_to_class = train_dataset.index_to_class
        mapping_save_path = os.path.join(output_dir, "class_mapping.json")
        with open(mapping_save_path, 'w+') as f:
            json.dump(index_to_class, f, indent=4)
        log.info(f"Mapping saved at: {mapping_save_path}")

        mlflow.log_param("num_classes", num_of_classes_in_dataset)
        mlflow.log_param("train_size", len(train_dataset))
        mlflow.log_param("val_data_size", len(val_dataset))
        mlflow.log_artifact(mapping_save_path)

        # Save run ID
        output_dir_path = Path(output_dir)
        (output_dir_path / "run_id.txt").write_text(run_id)
        mlflow.log_text(run_id, "run_id.txt")

        # Create model
        log.info("Model Creation Begin")
        vit_model = model.VisionTransformer(
            in_channels=cfg.model.in_channels,
            image_size=cfg.model.image_size,
            patch_size=cfg.model.patch_size,
            number_of_encoder=cfg.model.number_of_encoder,
            embeddings=cfg.model.embedding_dims,
            d_ff_scale=cfg.model.d_ff_scale_factor,
            heads=cfg.model.heads,
            input_dropout_rate=cfg.model.input_dropout_rate,
            attention_dropout_rate=cfg.model.attention_dropout_rate,
            feed_forward_dropout_rate=cfg.model.feed_forward_dropout_rate,
            number_of_classes=cfg.model.number_of_classes
        ).to(device)

        test_input = torch.randn((
            cfg.train.batch_size,
            cfg.model.in_channels,
            cfg.model.image_size,
            cfg.model.image_size
        )).to(device)

        log.info("Model Creation Completed.")
        log.info("--- Model Summary ---")
        log.info(summary(vit_model, input_data=test_input))
        log.info("--------------------")

        mlflow.log_param("model_params", sum(p.numel() for p in vit_model.parameters()))
        mlflow.log_param("embedding_dim", cfg.model.embedding_dims)

        # Setup training
        total_training_steps = cfg.train.epochs * len(train_data_loader)
        num_warmup_steps = int(0.23 * total_training_steps)
        
        mlflow.log_param("warmup_steps", num_warmup_steps)
        mlflow.log_param("total_training_steps", total_training_steps)

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, total_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        optimizer = optim.AdamW(
            vit_model.parameters(),
            lr=cfg.optimizer.lr,
            betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            weight_decay=cfg.optimizer.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

        # Training loop
        log.info("MODEL TRAINING BEGINS...")
        train_acc = []
        train_loss = []
        learning_rates = []
        val_acc = []
        val_loss = []
        n_train = len(train_data_loader)
        n_val = len(val_data_loader)
        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(cfg.train.epochs):
            # Training phase
            vit_model.train()
            loss_average = 0
            accuracy_average = 0

            for image, label in train_data_loader:
                optimizer.zero_grad()
                learning_rates.append(scheduler.get_last_lr()[0])
                
                image = image.to(device)
                label = label.to(device)
                
                pred_logits = vit_model(image)
                loss = criterion(pred_logits, label)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(vit_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                loss_average += loss.item()
                accuracy_average += accuracy_fn(pred_logits, label)

            epoch_avg_loss = loss_average / n_train
            epoch_avg_accuracy = accuracy_average / n_train
            train_loss.append(epoch_avg_loss)
            train_acc.append(epoch_avg_accuracy)

            # Validation phase
            vit_model.eval()
            val_avg_loss = 0
            val_avg_acc = 0

            with torch.inference_mode():
                for test_images, test_labels in val_data_loader:
                    test_images = test_images.to(device)
                    test_labels = test_labels.to(device)

                    pred_logits = vit_model(test_images)
                    loss = criterion(pred_logits, test_labels)

                    val_avg_loss += loss.item()
                    val_avg_acc += accuracy_fn(pred_logits, test_labels)

            val_epoch_avg_loss = val_avg_loss / n_val
            val_epoch_avg_accuracy = val_avg_acc / n_val
            val_loss.append(val_epoch_avg_loss)
            val_acc.append(val_epoch_avg_accuracy)

            # Log metrics
            mlflow.log_metric("train_loss", epoch_avg_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_avg_accuracy, step=epoch)
            mlflow.log_metric("val_loss", val_epoch_avg_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_epoch_avg_accuracy, step=epoch)
            mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch)

            log.info(
                f"Epoch {epoch+1} | "
                f"train_loss: {epoch_avg_loss:.4f} | train_accuracy: {epoch_avg_accuracy:.4f} | "
                f"val_loss: {val_epoch_avg_loss:.4f} | val_accuracy: {val_epoch_avg_accuracy:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

            # Model checkpointing - FIXED: Proper model logging
            if val_epoch_avg_accuracy > best_val_acc:
                best_val_acc = val_epoch_avg_accuracy
                best_epoch = epoch + 1
                
                # Save local checkpoint
                model_path = os.path.join(output_dir, "best_model.pt")
                torch.save(vit_model.state_dict(), model_path)
                
                # CRITICAL FIX: Log model correctly to MLflow
                # Move model to CPU for serialization
                vit_model_cpu = vit_model.cpu()
                
                # Create a sample input on CPU
                sample_input = torch.randn((
                    1,  # batch size 1 for inference
                    cfg.model.in_channels,
                    cfg.model.image_size,
                    cfg.model.image_size
                ))
                
                # Log model WITHOUT registered_model_name (register separately)
                mlflow.pytorch.log_model(
                    pytorch_model=vit_model_cpu,
                    artifact_path="model",  # This creates clean path
                    signature=mlflow.models.infer_signature(
                        sample_input.numpy(),
                        vit_model_cpu(sample_input).detach().numpy()
                    ),
                    input_example=sample_input.numpy(),
                    pip_requirements=[
                        f"torch=={torch.__version__}",
                        "torchvision",
                        "pillow",
                    ]
                )
                
                # Move model back to original device
                vit_model.to(device)
                
                mlflow.log_metric("best_val_acc", best_val_acc)
                mlflow.log_metric("best_epoch", best_epoch)
                
                log.info(f"✅ New best model logged to MLflow (val_acc: {best_val_acc:.4f})")

        learning_rates.append(scheduler.get_last_lr()[0])

        # Save training history
        log.info("Storing training artifacts details")
        data = list(zip(train_loss, train_acc, val_loss, val_acc))
        df = pd.DataFrame(data, columns=['train loss', 'train acc', 'val loss', 'val acc'])
        
        csv_path = os.path.join(output_dir, "training_history.csv")
        df.to_csv(csv_path, index_label="epoch")
        
        # Save plots
        log.info("Working on Training Plots...")
        plot_path = saving_training_plots(df, learning_rates, output_dir)
        
        # Save learning rates
        lr_df = pd.DataFrame({'learning_rates_per_step': learning_rates})
        lr_csv_path = os.path.join(output_dir, "learning_rates_per_step.csv")
        lr_df.to_csv(lr_csv_path, index_label="steps")

        mlflow.log_artifact(csv_path)
        mlflow.log_artifact(lr_csv_path)

        log.info(f"😎 Training Completed, details stored in {output_dir}")
        log.info(f"📊 Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        log.info(f"🔗 MLflow Run ID: {run_id}")


if __name__ == "__main__":
    main_loop()