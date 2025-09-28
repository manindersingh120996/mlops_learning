import streamlit as st
from omegaconf import OmegaConf
import subprocess
import pandas as pd
import re
import os
from functools import reduce
import psutil
import itertools
import time
import threading
import queue

st.set_page_config(layout="wide", page_title="ViT Training Studio")

# --- UI Styling & Components ---
st.markdown("""
    <style>
        .stButton>button { background-color: #4CAF50; color: white; }
        .stButton>button:hover { background-color: #45a049; }
        .stop-button .stButton>button { background-color: #f44336; color: white; }
        .stop-button .stButton>button:hover { background-color: #e53935; }
        .log-container { 
            background-color: #212529; 
            color: white; 
            border-radius: 5px; 
            padding: 15px; 
            height: 400px; 
            overflow-y: auto; 
            font-family: 'Courier New', Courier, monospace; 
            font-size: 11px;
            line-height: 1.2;
            white-space: pre-wrap;
            word-wrap: break-word;
            border: 1px solid #444;
        }
        .error-message { color: #f44336; }
        .success-message { color: #4CAF50; }
        .warning-message { color: #ff9800; }
        .parameter-group { 
            border: 1px solid #ddd; 
            border-radius: 5px; 
            padding: 15px; 
            margin: 10px 0; 
            background-color: #f8f9fa;
        }
        .validation-success { color: #4CAF50; font-weight: bold; }
        .validation-error { color: #f44336; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- Helper & Core Logic Functions ---

@st.cache_data
def load_config():
    """Loads the base configuration from the YAML file."""
    try:
        with open("configs/config.yaml", "r") as f:
            return OmegaConf.load(f)
    except FileNotFoundError:
        st.error("Configuration file not found. Please ensure 'configs/config.yaml' exists.")
        return None

def is_process_running(pid):
    """Check if a process with the given PID is running."""
    return psutil.pid_exists(pid) if pid else False

def validate_single_config(cfg):
    """Validates a single run configuration with detailed feedback."""
    errors = []
    warnings = []
    
    # Model Architecture Validation
    if cfg.model.embedding_dims % cfg.model.heads != 0:
        errors.append(f"❌ Embedding Dimensions ({cfg.model.embedding_dims}) must be divisible by Heads ({cfg.model.heads})")
    else:
        st.sidebar.markdown("✅ **Model Architecture**: Valid", unsafe_allow_html=True)
    
    # Optimizer Validation
    if not (0 < cfg.optimizer.lr < 1.0):
        errors.append(f"❌ Learning Rate ({cfg.optimizer.lr}) must be between 0 and 1")
    if cfg.optimizer.weight_decay < 0:
        errors.append(f"❌ Weight Decay ({cfg.optimizer.weight_decay}) must be non-negative")
    
    # Training Parameters Validation
    if cfg.train.batch_size <= 0:
        errors.append("❌ Batch Size must be greater than 0")
    elif cfg.train.batch_size > 512:
        warnings.append(f"⚠️ Large Batch Size ({cfg.train.batch_size}) may require significant memory")
    
    if cfg.train.epochs <= 0:
        errors.append("❌ Number of Epochs must be greater than 0")
    elif cfg.train.epochs > 1000:
        warnings.append(f"⚠️ High Epoch Count ({cfg.train.epochs}) will require long training time")
    
    # Architecture Warnings
    if cfg.model.d_ff_scale_factor * cfg.model.embedding_dims > 8192:
        warnings.append("⚠️ Very large Feed-Forward dimension may cause memory issues")
    
    return errors, warnings

def validate_multirun_params(params):
    """Enhanced validation for multi-run parameters with live feedback."""
    errors = []
    warnings = []
    total_runs = 0
    
    try:
        param_combinations = []
        
        # Parse and validate each parameter
        for key, value_str in params.items():
            if not value_str.strip():
                continue
                
            values = [v.strip() for v in value_str.split(',') if v.strip()]
            if not values:
                continue
                
            param_combinations.append((key, values))
            
            # Type-specific validation
            if key == "model.embedding_dims":
                dims = [int(v) for v in values]
                for d in dims:
                    if d <= 0:
                        errors.append(f"❌ Invalid Embedding Dimension: {d}")
                    elif d > 2048:
                        warnings.append(f"⚠️ Large Embedding Dimension: {d}")
            
            elif key == "model.heads":
                heads = [int(v) for v in values]
                for h in heads:
                    if h <= 0:
                        errors.append(f"❌ Invalid Head Count: {h}")
                    elif h > 32:
                        warnings.append(f"⚠️ Very High Head Count: {h}")
            
            elif key == "optimizer.lr":
                lrs = [float(v) for v in values]
                for lr in lrs:
                    if not (0 < lr < 1.0):
                        errors.append(f"❌ Invalid Learning Rate: {lr}")
        
        # Calculate total combinations
        if param_combinations:
            total_runs = 1
            for _, values in param_combinations:
                total_runs *= len(values)
        
        # Cross-parameter validation (embedding_dims % heads == 0)
        if "model.embedding_dims" in params and "model.heads" in params:
            dims = [int(x.strip()) for x in params["model.embedding_dims"].split(',') if x.strip()]
            heads = [int(x.strip()) for x in params["model.heads"].split(',') if x.strip()]
            
            invalid_combos = []
            for d, h in itertools.product(dims, heads):
                if d % h != 0:
                    invalid_combos.append(f"({d}, {h})")
            
            if invalid_combos:
                errors.append(f"❌ Invalid (Embedding_Dim, Head) combinations: {', '.join(invalid_combos[:5])}{'...' if len(invalid_combos) > 5 else ''}")
        
        # Resource warnings
        if total_runs > 50:
            warnings.append(f"⚠️ High number of runs ({total_runs}) will take significant time")
        elif total_runs > 100:
            errors.append(f"❌ Too many runs ({total_runs}). Consider reducing parameter combinations.")
            
    except ValueError as e:
        errors.append(f"❌ Parameter parsing error: {str(e)}")
    
    return errors, warnings, total_runs

def generate_single_run_command(cfg, default_cfg):
    """Generate single run command with Hydra overrides for changed parameters."""
    base_command = ["python", os.path.abspath("train.py")]
    
    overrides = []
    
    # Compare with default config and add overrides for changed parameters
    def compare_and_add_override(path, current_val, default_val):
        if current_val != default_val:
            overrides.append(f"{path}={current_val}")
    
    # Model parameters
    compare_and_add_override("model.embedding_dims", cfg.model.embedding_dims, default_cfg.model.embedding_dims)
    compare_and_add_override("model.heads", cfg.model.heads, default_cfg.model.heads)
    compare_and_add_override("model.number_of_encoder", cfg.model.number_of_encoder, default_cfg.model.number_of_encoder)
    compare_and_add_override("model.d_ff_scale_factor", cfg.model.d_ff_scale_factor, default_cfg.model.d_ff_scale_factor)
    compare_and_add_override("model.patch_size", cfg.model.patch_size, default_cfg.model.patch_size)
    compare_and_add_override("model.input_dropout_rate", cfg.model.input_dropout_rate, default_cfg.model.input_dropout_rate)
    compare_and_add_override("model.attention_dropout_rate", cfg.model.attention_dropout_rate, default_cfg.model.attention_dropout_rate)
    compare_and_add_override("model.feed_forward_dropout_rate", cfg.model.feed_forward_dropout_rate, default_cfg.model.feed_forward_dropout_rate)
    
    # Training parameters
    compare_and_add_override("train.epochs", cfg.train.epochs, default_cfg.train.epochs)
    compare_and_add_override("train.batch_size", cfg.train.batch_size, default_cfg.train.batch_size)
    
    # Optimizer parameters
    compare_and_add_override("optimizer.lr", cfg.optimizer.lr, default_cfg.optimizer.lr)
    compare_and_add_override("optimizer.weight_decay", cfg.optimizer.weight_decay, default_cfg.optimizer.weight_decay)
    compare_and_add_override("optimizer.beta1", cfg.optimizer.beta1, default_cfg.optimizer.beta1)
    compare_and_add_override("optimizer.beta2", cfg.optimizer.beta2, default_cfg.optimizer.beta2)
    
    # Add warmup_steps if available
    if hasattr(cfg.optimizer, 'warmup_steps') and hasattr(default_cfg.optimizer, 'warmup_steps'):
        compare_and_add_override("optimizer.warmup_steps", cfg.optimizer.warmup_steps, default_cfg.optimizer.warmup_steps)
    
    return base_command + overrides
def generate_hydra_multirun_command(params, mode="multirun"):
    """Generate Hydra multi-run command with proper overrides."""
    base_command = ["python", os.path.abspath("train.py")]
    
    if mode == "multirun":
        base_command.append("--multirun")
    
    # Add parameter overrides for non-empty parameters only
    overrides = []
    for key, value_str in params.items():
        if value_str.strip():
            # For multi-run, use comma-separated values
            if mode == "multirun":
                overrides.append(f"{key}={value_str}")
            else:
                # For single run, use first value
                first_value = value_str.split(',')[0].strip()
                overrides.append(f"{key}={first_value}")
    
    return base_command + overrides

# --- UI Rendering Functions ---

def display_validation_status(errors, warnings):
    """Display validation results with enhanced feedback."""
    if errors:
        st.sidebar.error("❌ Configuration Issues:")
        for error in errors:
            st.sidebar.markdown(f"<span class='validation-error'>{error}</span>", unsafe_allow_html=True)
    
    if warnings:
        st.sidebar.warning("⚠️ Warnings:")
        for warning in warnings:
            st.sidebar.markdown(f"<span class='warning-message'>{warning}</span>", unsafe_allow_html=True)
    
    if not errors and not warnings:
        st.sidebar.success("✅ Configuration Valid")

def display_enhanced_config_widgets(cfg):
    """Enhanced single-run configuration with better organization and validation."""
    with st.expander("🏗️ Model Architecture", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cfg.model.embedding_dims = st.number_input(
                "Embedding Dimensions (d_model)", 
                min_value=64, max_value=2048, step=64,
                value=cfg.model.embedding_dims,
                help="Must be divisible by number of heads. Common values: 256, 512, 768, 1024"
            )
            
            cfg.model.heads = st.slider(
                "Attention Heads", 1, 32, cfg.model.heads,
                help="Number of attention heads. Must divide embedding_dims evenly."
            )
            
        with col2:
            cfg.model.number_of_encoder = st.slider(
                "Encoder Layers", 1, 24, cfg.model.number_of_encoder,
                help="Number of transformer encoder blocks"
            )
            
            cfg.model.d_ff_scale_factor = st.slider(
                "Feed-Forward Scale", 1, 8, cfg.model.d_ff_scale_factor,
                help="Multiplier for feed-forward layer size (typically 2-4x)"
            )
            
        with col3:
            cfg.model.patch_size = st.selectbox(
                "Patch Size", [4, 8, 16, 32], 
                index=[4, 8, 16, 32].index(cfg.model.patch_size),
                help="Size of image patches for tokenization"
            )
            
            # Live validation display
            if cfg.model.embedding_dims % cfg.model.heads == 0:
                st.success(f"✅ Head dimension: {cfg.model.embedding_dims // cfg.model.heads}")
            else:
                st.error(f"❌ Incompatible dimensions")

    with st.expander("🏋️‍♀️ Training Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cfg.train.epochs = st.number_input(
                "Training Epochs", min_value=1, max_value=1000, 
                value=cfg.train.epochs,
                help="Number of complete passes through the training data"
            )
            
        with col2:
            cfg.train.batch_size = st.number_input(
                "Batch Size", min_value=1, max_value=512, 
                value=cfg.train.batch_size,
                help="Number of samples processed before updating weights"
            )
            
        with col3:
            # Add warmup steps if available in config
            if hasattr(cfg.optimizer, 'warmup_steps'):
                cfg.optimizer.warmup_steps = st.number_input(
                    "Warmup Steps", min_value=0, 
                    value=getattr(cfg.optimizer, 'warmup_steps', 1000),
                    help="Learning rate warmup steps"
                )

    with st.expander("⚙️ Optimization & Regularization", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Optimizer")
            cfg.optimizer.lr = st.number_input(
                "Learning Rate", 
                min_value=1e-6, max_value=1.0, 
                value=float(cfg.optimizer.lr), 
                format="%.6f", step=1e-5,
                help="Step size for gradient updates"
            )
            cfg.optimizer.weight_decay = st.number_input(
                "Weight Decay", 
                min_value=0.0, max_value=1.0,
                value=float(cfg.optimizer.weight_decay), 
                format="%.4f", step=0.0001,
                help="L2 regularization strength"
            )
            
        with col2:
            st.subheader("Adam Parameters")
            cfg.optimizer.beta1 = st.number_input(
                "Beta 1 (Momentum)", 
                min_value=0.800, max_value=0.999, 
                value=float(cfg.optimizer.beta1), 
                format="%.3f", step=0.001,
                help="Exponential decay rate for first moment estimates"
            )
            cfg.optimizer.beta2 = st.number_input(
                "Beta 2 (RMSprop)", 
                min_value=0.900, max_value=0.9999, 
                value=float(cfg.optimizer.beta2), 
                format="%.3f", step=0.001,
                help="Exponential decay rate for second moment estimates"
            )
            
        with col3:
            st.subheader("Dropout Rates")
            cfg.model.input_dropout_rate = st.slider(
                "Input Dropout", 0.0, 0.9, 
                cfg.model.input_dropout_rate, 0.05,
                help="Dropout rate for input embeddings"
            )
            cfg.model.attention_dropout_rate = st.slider(
                "Attention Dropout", 0.0, 0.9, 
                cfg.model.attention_dropout_rate, 0.05,
                help="Dropout rate in attention layers"
            )
            cfg.model.feed_forward_dropout_rate = st.slider(
                "Feed-Forward Dropout", 0.0, 0.9, 
                cfg.model.feed_forward_dropout_rate, 0.05,
                help="Dropout rate in feed-forward layers"
            )
    
    return cfg

def display_enhanced_multirun_widgets(cfg):
    """Enhanced multi-run configuration interface."""
    params = {}
    
    st.info("💡 **Multi-Run Configuration**: Enter comma-separated values for parameters you want to vary. Leave empty to use default values.")
    
    # Live preview container
    preview_container = st.container()
    
    with st.expander("🏗️ Model Architecture Parameters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            params["model.embedding_dims"] = st.text_input(
                "Embedding Dimensions", 
                value=str(cfg.model.embedding_dims),
                help="e.g., '256,512,768' for multiple values",
                placeholder="256,512,768"
            )
            
            params["model.heads"] = st.text_input(
                "Attention Heads", 
                value=str(cfg.model.heads),
                help="e.g., '8,12,16' - must divide embedding_dims",
                placeholder="8,12,16"
            )
            
            params["model.number_of_encoder"] = st.text_input(
                "Encoder Layers", 
                value=str(cfg.model.number_of_encoder),
                help="e.g., '6,12,18'",
                placeholder="6,12,18"
            )
            
        with col2:
            params["model.d_ff_scale_factor"] = st.text_input(
                "Feed-Forward Scale Factor", 
                value=str(cfg.model.d_ff_scale_factor),
                help="e.g., '2,4,6'",
                placeholder="2,4,6"
            )
            
            params["model.patch_size"] = st.text_input(
                "Patch Size", 
                value=str(cfg.model.patch_size),
                help="e.g., '8,16,32'",
                placeholder="8,16,32"
            )

    with st.expander("🏋️‍♀️ Training & Optimization Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            params["train.epochs"] = st.text_input(
                "Training Epochs", 
                value=str(cfg.train.epochs),
                help="e.g., '50,100,200'",
                placeholder="50,100,200"
            )
            
            params["train.batch_size"] = st.text_input(
                "Batch Size", 
                value=str(cfg.train.batch_size),
                help="e.g., '16,32,64'",
                placeholder="16,32,64"
            )
            
        with col2:
            params["optimizer.lr"] = st.text_input(
                "Learning Rate", 
                value=str(cfg.optimizer.lr),
                help="e.g., '0.001,0.0001,0.00001'",
                placeholder="0.001,0.0001"
            )
            
            params["optimizer.weight_decay"] = st.text_input(
                "Weight Decay", 
                value=str(cfg.optimizer.weight_decay),
                help="e.g., '0.0,0.01,0.1'",
                placeholder="0.0,0.01,0.1"
            )
            
        with col3:
            params["model.input_dropout_rate"] = st.text_input(
                "Input Dropout Rate", 
                value=str(cfg.model.input_dropout_rate),
                help="e.g., '0.0,0.1,0.2'",
                placeholder="0.0,0.1,0.2"
            )
            
            params["model.attention_dropout_rate"] = st.text_input(
                "Attention Dropout Rate", 
                value=str(cfg.model.attention_dropout_rate),
                help="e.g., '0.0,0.1,0.2'",
                placeholder="0.0,0.1,0.2"
            )

    # Live parameter combination preview
    with preview_container:
        errors, warnings, total_runs = validate_multirun_params(params)
        
        if total_runs > 0:
            st.success(f"✅ **Configuration Preview**: {total_runs} total training runs will be executed")
            
            # Show sample combinations
            if total_runs <= 10:
                st.info("**All Parameter Combinations:**")
                sample_combinations = []
                active_params = {k: v.split(',') for k, v in params.items() if v.strip()}
                if active_params:
                    keys = list(active_params.keys())
                    for combo in itertools.product(*active_params.values()):
                        combo_dict = {k: v.strip() for k, v in zip(keys, combo)}
                        sample_combinations.append(combo_dict)
                    
                    for i, combo in enumerate(sample_combinations, 1):
                        st.write(f"**Run {i}:** {combo}")
            else:
                st.info(f"**Sample Combinations** (showing first 5 of {total_runs}):")
                # Show first few combinations as preview
                active_params = {k: v.split(',') for k, v in params.items() if v.strip()}
                if active_params:
                    keys = list(active_params.keys())
                    for i, combo in enumerate(itertools.islice(itertools.product(*active_params.values()), 5), 1):
                        combo_dict = {k: v.strip() for k, v in zip(keys, combo)}
                        st.write(f"**Run {i}:** {combo_dict}")
    
    return params

def display_log_container():
    """Create a fixed-height scrollable log container."""
    return st.empty()

# --- Main Application ---
def main():
    # Initialize session state
    if 'cfg' not in st.session_state:
        cfg = load_config()
        if cfg is None:
            return
        st.session_state.cfg = cfg
    
    if 'log_content' not in st.session_state:
        st.session_state.log_content = ""
    
    if 'current_process' not in st.session_state:
        st.session_state.current_process = None
        
    if 'total_runs' not in st.session_state:
        st.session_state.total_runs = 1
        
    if 'completed_runs' not in st.session_state:
        st.session_state.completed_runs = 0

    # Sidebar Configuration
    st.sidebar.title("🔬 ViT Training Studio")
    st.sidebar.markdown("---")
    
    # Training Mode Selection
    run_mode = st.sidebar.radio(
        "**Training Mode**", 
        ("Single Run", "Multi-Run"), 
        horizontal=True,
        help="Single Run: Train one model configuration\nMulti-Run: Train multiple configurations using Hydra"
    )
    
    # Load default config
    default_cfg = load_config()
    if default_cfg is None:
        return

    # Main Interface
    st.title("🚀 Vision Transformer Training Studio")
    st.markdown("Configure and monitor your ViT training experiments with real-time feedback.")
    
    # Configuration Section
    st.header("⚙️ Training Configuration")
    config_source = st.radio(
        "Configuration Source", 
        ["Use Default Configuration", "Customize Parameters"], 
        horizontal=True,
        help="Default: Use config.yaml settings\nCustomize: Modify parameters through UI"
    )

    errors, warnings = [], []
    total_runs = 1
    
    if config_source == "Customize Parameters":
        if run_mode == "Single Run":
            st.session_state.cfg = display_enhanced_config_widgets(st.session_state.cfg)
            errors, warnings = validate_single_config(st.session_state.cfg)
        else:
            st.session_state.multi_run_params = display_enhanced_multirun_widgets(st.session_state.cfg)
            errors, warnings, total_runs = validate_multirun_params(st.session_state.multi_run_params)
            st.session_state.total_runs = total_runs
    else:
        # Using default configuration
        st.info("Using default configuration from `configs/config.yaml`")
        if run_mode == "Multi-Run":
            st.warning("Multi-run mode requires parameter customization. Please select 'Customize Parameters' to specify variations.")
            errors.append("Multi-run mode requires parameter customization")
    
    # Display validation status
    display_validation_status(errors, warnings)
    
    # Training Controls
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Training Controls")
    
    training_active = st.session_state.current_process is not None and is_process_running(st.session_state.current_process.pid if st.session_state.current_process else None)
    
    if training_active:
        # Stop button
        st.sidebar.markdown('<div class="stop-button">', unsafe_allow_html=True)
        if st.sidebar.button("🛑 Stop Training", use_container_width=True, type="primary"):
            if st.session_state.current_process:
                try:
                    st.session_state.current_process.terminate()
                    st.session_state.current_process.wait(timeout=5)
                except:
                    try:
                        st.session_state.current_process.kill()
                    except:
                        pass
                st.session_state.current_process = None
            st.toast("🛑 Training stopped by user", icon="⚠️")
            st.rerun()
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Training status
        if run_mode == "Multi-Run":
            progress = st.session_state.completed_runs / st.session_state.total_runs if st.session_state.total_runs > 0 else 0
            st.sidebar.progress(progress, text=f"Progress: {st.session_state.completed_runs}/{st.session_state.total_runs} runs")
    else:
        # Start button
        start_disabled = bool(errors) or (run_mode == "Multi-Run" and config_source == "Use Default Configuration")
        
        if st.sidebar.button("🚀 Start Training", use_container_width=True, disabled=start_disabled, type="primary"):
            # Reset training state
            st.session_state.log_content = ""
            st.session_state.completed_runs = 0
            
            # Generate command based on mode
            if run_mode == "Single Run":
                if config_source == "Customize Parameters":
                    # Generate single run command with Hydra overrides
                    cmd = generate_single_run_command(st.session_state.cfg, default_cfg)
                else:
                    # Use default configuration
                    cmd = ["python", os.path.abspath("train.py")]
            else:
                # Multi-run mode with Hydra - ensure we have parameter overrides
                if config_source == "Customize Parameters":
                    # Filter out empty parameters for cleaner command
                    filtered_params = {k: v for k, v in st.session_state.multi_run_params.items() if v.strip()}
                    cmd = generate_hydra_multirun_command(filtered_params, "multirun")
                else:
                    # This shouldn't happen due to validation, but fallback
                    cmd = ["python", os.path.abspath("train.py"), "--multirun"]
            
            # Start training process
            try:
                st.session_state.current_process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True, 
                    encoding='utf-8',
                    cwd=os.path.dirname(os.path.abspath("train.py"))
                )
                st.toast(f"🚀 {run_mode} training started!", icon="✅")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start training: {str(e)}")
        
        if start_disabled and errors:
            st.sidebar.error("Fix configuration errors to start training")

    # Training Progress Section
    if training_active:
        st.header("📊 Training Progress")
        
        # Multi-run progress bar
        if run_mode == "Multi-Run":
            overall_progress = st.session_state.completed_runs / st.session_state.total_runs
            st.progress(overall_progress, text=f"Overall Progress: {st.session_state.completed_runs}/{st.session_state.total_runs} runs completed")
        
        # Live metrics
        col1, col2, col3, col4 = st.columns(4)
        metrics_containers = {
            'train_loss': col1.empty(),
            'train_acc': col2.empty(),
            'val_loss': col3.empty(),
            'val_acc': col4.empty()
        }
        
        # Charts
        loss_chart = st.empty()
        acc_chart = st.empty()
        
        # Log container
        st.subheader("📋 Training Logs")
        log_container = st.empty()
        
        # Read training output
        history = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        
        try:
            while training_active and st.session_state.current_process:
                # Check if process is still running
                if st.session_state.current_process.poll() is not None:
                    # Process finished
                    st.session_state.current_process = None
                    if run_mode == "Multi-Run":
                        st.session_state.completed_runs = st.session_state.total_runs
                    st.toast("✅ Training completed!", icon="🎉")
                    st.rerun()
                    break
                
                # Read new output
                try:
                    line = st.session_state.current_process.stdout.readline()
                    if line:
                        st.session_state.log_content += line
                        
                        # Update log display (replace content, not append)
                        # Truncate extremely long lines for better display
                        display_log = st.session_state.log_content
                        if len(display_log) > 50000:  # Keep last 50k characters
                            display_log = "...\n" + display_log[-50000:]
                        
                        log_container.markdown(
                            f'<div class="log-container">{display_log}</div>', 
                            unsafe_allow_html=True
                        )
                        
                        # Parse metrics
                        epoch_match = re.search(r"Epoch (\d+)", line)
                        if epoch_match:
                            metrics_match = re.search(r"train_loss: ([\d.]+) \| train_accuracy: ([\d.]+) \| val_loss: ([\d.]+) \| val_accuracy: ([\d.]+)", line)
                            if metrics_match:
                                epoch = int(epoch_match.group(1))
                                train_loss, train_acc, val_loss, val_acc = map(float, metrics_match.groups())
                                
                                # Update metrics display
                                metrics_containers['train_loss'].metric("Train Loss", f"{train_loss:.4f}")
                                metrics_containers['train_acc'].metric("Train Accuracy", f"{train_acc:.4f}")
                                metrics_containers['val_loss'].metric("Val Loss", f"{val_loss:.4f}")
                                metrics_containers['val_acc'].metric("Val Accuracy", f"{val_acc:.4f}")
                                
                                # Update charts
                                new_row = pd.DataFrame([{
                                    'epoch': epoch,
                                    'train_loss': train_loss,
                                    'val_loss': val_loss,
                                    'train_acc': train_acc,
                                    'val_acc': val_acc
                                }])
                                history = pd.concat([history, new_row], ignore_index=True)
                                
                                if len(history) > 0:
                                    # Loss chart
                                    loss_data = history[['epoch', 'train_loss', 'val_loss']].set_index('epoch')
                                    loss_chart.line_chart(loss_data, use_container_width=True)
                                    
                                    # Accuracy chart
                                    acc_data = history[['epoch', 'train_acc', 'val_acc']].set_index('epoch')
                                    acc_chart.line_chart(acc_data, use_container_width=True)
                        
                        # Check for run completion in multi-run
                        if run_mode == "Multi-Run" and "Multirun execution finished" in line:
                            st.session_state.completed_runs = st.session_state.total_runs
                            
                    time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                    
                except Exception as e:
                    st.error(f"Error reading training output: {str(e)}")
                    break
                    
        except Exception as e:
            st.error(f"Training monitoring error: {str(e)}")
            if st.session_state.current_process:
                st.session_state.current_process = None

if __name__ == "__main__":
    main()