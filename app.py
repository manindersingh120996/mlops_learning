import streamlit as st

# Configure the main page
st.set_page_config(
    page_title="Vision Transformer Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main page content
st.title("🤖 Vision Transformer Studio")
st.markdown("### Complete Pipeline for Vision Transformer (ViT) Training and Inference")

st.markdown("""
Welcome to the **Vision Transformer Studio**! This application provides a comprehensive interface for training and using Vision Transformer models.

---
""")

# Feature highlights
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🚀 Training
    - Single & Multi-run experiments
    - Real-time progress monitoring
    - Parameter validation
    - Hydra configuration support
    - Process control (Start/Stop)
    """)

with col2:
    st.markdown("""
    ### 🔍 Inference
    - Automatic model discovery
    - Training history visualization
    - Image classification
    - Confidence scores
    - Model comparison
    """)

st.markdown("---")

# Instructions
st.markdown("""
### 📚 How to Use

👈 **Select a page from the sidebar** to begin:
- **🚀 Training** - Configure and train new models
- **🔍 Inference** - Use trained models for predictions

### 📊 Project Information

Based on the paper: *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*

**Output Directories:**
- `outputs/` - Single training runs
- `multirun/` - Multi-run experiments

### 🎯 Quick Start

1. Go to **Training** page to train your first model
2. Once trained, visit **Inference** page to test it on images
3. View training plots and metrics for any trained model

""")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and PyTorch | Vision Transformer from Scratch")
