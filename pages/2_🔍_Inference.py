import streamlit as st
from omegaconf import OmegaConf
import torch
import pandas as pd
from PIL import Image
import glob
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
from src.model import VisionTransformer
from src.dataset import val_transform
import numpy as np

# Configure page
st.set_page_config(layout="wide", page_title="ViT Inference Studio")

# Consistent styling with training app
st.markdown("""
    <style>
        .stButton>button { background-color: #4CAF50; color: white; }
        .stButton>button:hover { background-color: #45a049; }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .model-card {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            transition: transform 0.2s;
        }
        .model-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Device configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
    device_info = "🚀 Using CUDA GPU"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
    device_info = "🫡 Using Apple Silicon GPU"
else:
    device = torch.device('cpu')
    device_info = "💻 Using CPU"

# Helper Functions
@st.cache_data
def find_experiments(search_dirs=["outputs", "multirun"]):
    """Find all trained models in the specified directories."""
    all_valid_exp = []
    exp_dirs = []
    
    for base_dir in search_dirs:
        if os.path.exists(base_dir):
            if "multirun" in base_dir:
                glob_pattern = os.path.join(base_dir, "*", "*", "*", "")
                exp_dirs.extend(glob.glob(glob_pattern))
            if "outputs" in base_dir:
                glob_pattern = os.path.join(base_dir, "*", "*", "")
                exp_dirs.extend(glob.glob(glob_pattern))
    
    for exp_dir in exp_dirs:
        if (os.path.exists(os.path.join(exp_dir, "best_model.pt")) and
            os.path.exists(os.path.join(exp_dir, "training_history.csv")) and
            os.path.exists(os.path.join(exp_dir, ".hydra/config.yaml"))):
            
            history = pd.read_csv(os.path.join(exp_dir, "training_history.csv"))
            best_acc = history["val acc"].max()
            final_loss = history["val loss"].iloc[-1]
            total_epochs = len(history)
            
            # Get timestamp from path
            path_parts = Path(exp_dir).parts
            timestamp = f"{path_parts[-2]} {path_parts[-1]}" if len(path_parts) >= 2 else "Unknown"
            
            all_valid_exp.append({
                "path": exp_dir,
                "best_val_acc": best_acc,
                "final_val_loss": final_loss,
                "total_epochs": total_epochs,
                "timestamp": timestamp,
                "history": history
            })
    
    return sorted(all_valid_exp, key=lambda x: x["best_val_acc"], reverse=True)

@st.cache_resource
def load_model(exp_path):
    """Load model from experiment path."""
    cfg_path = os.path.join(exp_path, '.hydra', 'config.yaml')
    cfg = OmegaConf.load(cfg_path)
    
    # Use default values for missing config parameters
    model = VisionTransformer(
        in_channels=getattr(cfg.model, 'in_channels', 3),
        image_size=getattr(cfg.model, 'image_size', 224),
        patch_size=getattr(cfg.model, 'patch_size', 16),
        number_of_encoder=getattr(cfg.model, 'number_of_encoder', 6),
        embeddings=getattr(cfg.model, 'embedding_dims', 256),
        d_ff_scale=getattr(cfg.model, 'd_ff_scale_factor', 4),
        heads=getattr(cfg.model, 'heads', 8),
        input_dropout_rate=getattr(cfg.model, 'input_dropout_rate', 0.1),
        attention_dropout_rate=getattr(cfg.model, 'attention_dropout_rate', 0.1),
        feed_forward_dropout_rate=getattr(cfg.model, 'feed_forward_dropout_rate', 0.1),
        number_of_classes=getattr(cfg.model, 'number_of_classes', 4)
    ).to(device)
    
    model_path = os.path.join(exp_path, 'best_model.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load class mapping if available
    class_mapping_path = os.path.join(exp_path, 'class_mapping.json')
    class_mapping = None
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
    
    return model, cfg, class_mapping

def display_training_plots(experiment):
    """Display training plots from the selected experiment."""
    history = experiment["history"]
    
    # Create two columns for plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Loss plot
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(history.index + 1, history["train loss"], label="Train Loss", color='#1f77b4', linewidth=2)
        ax1.plot(history.index + 1, history["val loss"], label="Validation Loss", color='#ff7f0e', linewidth=2)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Training and Validation Loss", fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        
    with col2:
        # Accuracy plot
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(history.index + 1, history["train acc"], label="Train Accuracy", color='#2ca02c', linewidth=2)
        ax2.plot(history.index + 1, history["val acc"], label="Validation Accuracy", color='#d62728', linewidth=2)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_title("Training and Validation Accuracy", fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
    
    # Display metrics summary
    st.subheader("📊 Training Metrics Summary")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Best Val Accuracy", f"{experiment['best_val_acc']:.2%}")
    with metric_cols[1]:
        st.metric("Final Val Loss", f"{experiment['final_val_loss']:.4f}")
    with metric_cols[2]:
        st.metric("Total Epochs", experiment['total_epochs'])
    with metric_cols[3]:
        st.metric("Training Date", experiment['timestamp'])

def predict_image(image, model, transform, class_mapping=None):
    """Make prediction on a single image."""
    # Transform and predict
    transformed_image = transform(image).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        logits = model(transformed_image)
        probs = torch.softmax(logits, dim=1)
        
    # Get top 5 predictions
    top_probs, top_indices = torch.topk(probs[0], k=min(5, probs.shape[1]))
    
    results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        class_name = class_mapping[str(idx)] if class_mapping and str(idx) in class_mapping else f"Class {idx}"
        results.append({
            "class": class_name,
            "confidence": prob,
            "index": idx
        })
    
    return results

# Main App
def main():
    st.title("🔍 Vision Transformer Inference Studio")
    st.markdown(f"**Device:** {device_info}")
    
    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    
    # Find all experiments
    experiments = find_experiments()
    
    if not experiments:
        st.warning("No trained models found in 'outputs/' or 'multirun/' directories. Please train a model first.")
        return
    
    # Model selection interface
    st.sidebar.subheader(f"Found {len(experiments)} trained model(s)")
    
    # Create selection options
    model_options = []
    for i, exp in enumerate(experiments):
        option = f"{i+1}. Acc: {exp['best_val_acc']:.2%} | {exp['timestamp']}"
        model_options.append(option)
    
    selected_option = st.sidebar.selectbox(
        "Select a model:",
        model_options,
        help="Models are sorted by validation accuracy"
    )
    
    selected_idx = model_options.index(selected_option)
    selected_experiment = experiments[selected_idx]
    
    # Display selected model info
    st.sidebar.success(f"Selected: Model {selected_idx + 1}")
    st.sidebar.info(f"Path: `{selected_experiment['path']}`")
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📸 Image Inference", "📈 Training History", "ℹ️ Model Info"])
    
    with tab1:
        st.header("Upload Image for Classification")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to classify using the selected model"
        )
        
        # Test with sample images if available
        sample_images_dir = "random_test_images"
        if os.path.exists(sample_images_dir):
            sample_images = glob.glob(os.path.join(sample_images_dir, "*"))[:-1]
            if sample_images:
                st.subheader("Or try with sample images:")
                sample_cols = st.columns(len(sample_images))
                for idx, (col, img_path) in enumerate(zip(sample_cols, sample_images)):
                    with col:
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True)
                        if st.button(f"Use Sample {idx+1}", key=f"sample_{idx}"):
                            uploaded_file = img_path
                            st.session_state.use_sample = img_path
        
        # Process uploaded image
        if uploaded_file is not None or st.session_state.get('use_sample'):
            # Load the selected model
            with st.spinner("Loading model..."):
                model, config, class_mapping = load_model(selected_experiment['path'])
            
            # Handle both uploaded files and sample images
            if st.session_state.get('use_sample'):
                image = Image.open(st.session_state.use_sample).convert("RGB")
                st.session_state.use_sample = None
            else:
                image = Image.open(uploaded_file).convert("RGB")
            
            # Display image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Input Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Predictions")
                
                # Make prediction
                with st.spinner("Running inference..."):
                    results = predict_image(image, model, val_transform, class_mapping)
                
                # Display top prediction
                top_result = results[0]
                st.success(f"**Predicted Class:** {top_result['class']}")
                st.metric("Confidence", f"{top_result['confidence']:.2%}")
                
                # Display top 5 predictions as bar chart
                st.subheader("Top 5 Predictions")
                for i, result in enumerate(results, 1):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        # Convert numpy float to Python float for st.progress
                        st.progress(float(result['confidence']))
                    with col_b:
                        st.write(f"{result['class']} ({result['confidence']:.1%})")
    
    with tab2:
        st.header("📈 Training History")
        display_training_plots(selected_experiment)
        
        # Option to view raw data
        with st.expander("View Raw Training Data"):
            st.dataframe(selected_experiment['history'])
    
    with tab3:
        st.header("ℹ️ Model Information")
        
        # Load and display config
        cfg_path = os.path.join(selected_experiment['path'], '.hydra', 'config.yaml')
        cfg = OmegaConf.load(cfg_path)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Architecture")
            st.json({
                "Embedding Dimensions": cfg.model.embedding_dims,
                "Number of Encoders": cfg.model.number_of_encoder,
                "Attention Heads": cfg.model.heads,
                "Patch Size": cfg.model.patch_size,
                "Image Size": cfg.model.image_size,
                "Number of Classes": cfg.model.number_of_classes
            })
        
        with col2:
            st.subheader("Training Configuration")
            training_config = {}
            
            # Safely access configuration values that might be missing
            if hasattr(cfg, 'train'):
                training_config["Epochs"] = getattr(cfg.train, 'epochs', 'N/A')
                training_config["Batch Size"] = getattr(cfg.train, 'batch_size', 'N/A')
            
            if hasattr(cfg, 'optimizer'):
                training_config["Learning Rate"] = getattr(cfg.optimizer, 'lr', 'N/A')
                training_config["Weight Decay"] = getattr(cfg.optimizer, 'weight_decay', 'N/A')
            
            st.json(training_config)
        
        # Display full config
        with st.expander("View Full Configuration"):
            st.json(OmegaConf.to_container(cfg, resolve=True))

if __name__ == "__main__":
    main()
