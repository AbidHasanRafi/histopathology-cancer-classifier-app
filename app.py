import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# ======================
# APP CONFIGURATION
# ======================
st.set_page_config(
    page_title="Histopathology Classifier", 
    layout="wide", 
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# Dark mode compatible CSS
st.markdown("""
<style>
    :root {
        --primary: #4a8cff;
        --background: #0E1117;
        --secondary-background: #262730;
        --text: #FAFAFA;
        --card-background: #1E1E1E;
    }
    
    .main {
        background-color: var(--background);
        color: var(--text);
    }
    
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    .stSelectbox, .stFileUploader {
        background-color: var(--secondary-background);
        color: var(--text);
    }
    
    .st-expander {
        background-color: var(--secondary-background);
        border-radius: 8px;
    }
    
    .prediction-card {
        background-color: var(--card-background);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary);
    }
    
    .sample-image {
        border-radius: 8px;
        transition: transform 0.2s;
    }
    
    .sample-image:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .tab-content {
        padding: 1rem;
        background-color: var(--secondary-background);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Histopathology Classifier - Cancer Classification")

# ======================
# CONSTANTS
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_ORDER = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']
CLASS_COLORS = {
    'colon_aca': '#FF6B6B',
    'colon_n': '#4ECDC4',
    'lung_aca': '#FFA07A',
    'lung_n': '#7FB3D5',
    'lung_scc': '#D4A5A5'
}
HUMAN_NAMES = {
    'colon_aca': 'Colon Adenocarcinoma',
    'colon_n': 'Colon Normal Tissue',
    'lung_aca': 'Lung Adenocarcinoma',
    'lung_n': 'Lung Normal Tissue',
    'lung_scc': 'Lung Squamous Cell Carcinoma'
}

# Sample images (replace with your actual paths)
SAMPLE_IMAGES = {
    "Lung Adenocarcinoma": "samples/lung_aca.jpeg",
    "Lung Squamous": "samples/lung_scc.jpeg",
    "Colon Adenocarcinoma": "samples/colon_aca.jpeg",
    "Normal Lung": "samples/lung_n.jpeg",
    "Normal Colon": "samples/colon_n.jpeg"
}

# ======================
# MODEL LOADING
# ======================
@st.cache_resource
def load_model(model_path='best_model.pth'):
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            return None
            
        model = models.resnet101(pretrained=False)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(CLASS_ORDER)))
        
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model.to(device)
    except Exception as e:
        st.error(f"🚨 Error loading model: {str(e)}")
        return None

# ======================
# IMAGE PROCESSING
# ======================
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    try:
        img = Image.open(image).convert('RGB')
        if min(img.size) < 224:
            st.warning("⚠️ Image dimensions are small - minimum 224x224 recommended")
        return img, TRANSFORM(img)
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None, None

# ======================
# EXPLAINABILITY METHODS
# ======================
def grad_cam(model, img_tensor, target_class=None):
    try:
        feature_maps = []
        gradients = []
        
        def forward_hook(module, input, output):
            feature_maps.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())
        
        target_layer = model.layer4[-1].conv2
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        img_tensor = img_tensor.unsqueeze(0).to(device)
        img_tensor.requires_grad = True
        outputs = model(img_tensor)
        
        if target_class is None:
            target_class = torch.argmax(outputs).item()
        
        one_hot = torch.zeros_like(outputs)
        one_hot[0][target_class] = 1.0
        model.zero_grad()
        outputs.backward(gradient=one_hot)
        
        feature_map = feature_maps[0].squeeze(0)
        gradient = gradients[0].squeeze(0)
        weights = torch.mean(gradient, dim=(1, 2))
        
        cam = torch.zeros(feature_map.shape[1:], dtype=torch.float32).to(device)
        for j, w in enumerate(weights):
            cam += w * feature_map[j]
        
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        
        return cam, target_class
    except Exception as e:
        st.error(f"❌ Grad-CAM error: {str(e)}")
        return None, None
    finally:
        if 'forward_handle' in locals(): forward_handle.remove()
        if 'backward_handle' in locals(): backward_handle.remove()

def guided_grad_cam(model, img_tensor, target_class=None):
    """Simpler alternative implementation"""
    try:
        # Get standard Grad-CAM first
        cam, target_class = grad_cam(model, img_tensor, target_class)
        if cam is None:
            return None, None
            
        # Get guided backpropagation
        img_tensor = img_tensor.unsqueeze(0).to(device)
        img_tensor.requires_grad = True
        outputs = model(img_tensor)
        
        if target_class is None:
            target_class = torch.argmax(outputs).item()
        
        model.zero_grad()
        one_hot = torch.zeros_like(outputs)
        one_hot[0][target_class] = 1
        outputs.backward(gradient=one_hot)
        
        # Get guided gradients
        guided_grads = img_tensor.grad.squeeze(0)
        guided_grads = guided_grads * (guided_grads > 0).float()  # Guided ReLU
        guided_grads = torch.mean(guided_grads, dim=0).cpu().numpy()
        
        # Combine with Grad-CAM
        guided_cam = cam * guided_grads
        guided_cam = (guided_cam - guided_cam.min()) / (guided_cam.max() - guided_cam.min() + 1e-10)
        
        return guided_cam, target_class
    except Exception as e:
        st.error(f"❌ Guided Grad-CAM error: {str(e)}")
        return None, None

# ======================
# VISUALIZATIONS
# ======================
def plot_probability_distribution(probs):
    plt.style.use('dark_background')
    df = pd.DataFrame({
        'Class': [HUMAN_NAMES[c] for c in CLASS_ORDER],
        'Probability': probs[0].cpu().numpy(),
        'Color': [CLASS_COLORS[c] for c in CLASS_ORDER]
    }).sort_values('Probability', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df['Class'], df['Probability'], color=df['Color'])
    ax.set_xlim(0, 1)
    ax.set_title('Class Probability Distribution', fontsize=14, pad=20)
    ax.set_xlabel('Probability', fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_cam_visualization(img_pil, cam, title):
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original
    ax1.imshow(img_pil)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Heatmap
    ax2.imshow(cam, cmap='inferno')
    ax2.set_title(f'{title} Heatmap')
    ax2.axis('off')
    
    # Overlay
    img_np = np.array(img_pil.resize((224, 224))) / 255.0
    ax3.imshow(img_np)
    ax3.imshow(cam, cmap='inferno', alpha=0.5)
    ax3.set_title('Overlay')
    ax3.axis('off')
    
    plt.tight_layout()
    return fig

# ======================
# MAIN APP
# ======================
def main():
    # Load model
    model = load_model()
    if model is None:
        st.stop()

    # Sidebar
    with st.sidebar:
        # Improved Sample Gallery
        st.markdown("### Sample Gallery")
        
        # Dropdown to select sample
        selected_sample = st.selectbox(
            "Select sample image:", 
            list(SAMPLE_IMAGES.keys()),
            key="sample_selector"
        )
        
        # Display selected sample with enhanced styling
        if selected_sample:
            sample_path = SAMPLE_IMAGES[selected_sample]
            if os.path.exists(sample_path):
                with st.container():
                    st.markdown(f"**{selected_sample}**")
                    
                    # Display the image directly using st.image with proper styling
                    st.image(
                        sample_path,
                        caption=selected_sample,
                        use_container_width=True,
                        output_format="JPEG",
                        clamp=True
                    )
                    
                    # Styled use button
                    if st.button(
                        f"Analyze {selected_sample}",
                        key=f"analyze_{selected_sample}",
                        help=f"Click to analyze this {selected_sample} sample",
                        type="primary"
                    ):
                        st.session_state.sample_image = sample_path
                        st.rerun()  # Refresh to show the selected image
        
        st.markdown("---")
        st.markdown("### Settings")
        explain_method = st.selectbox(
            "Explanation Method",
            ["Grad-CAM", "Guided Grad-CAM", "Both"],
            index=0
        )
        debug_mode = st.checkbox("Show debug info", False)

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Image")
        if 'sample_image' in st.session_state and os.path.exists(st.session_state.sample_image):
            uploaded_file = st.session_state.sample_image
        else:
            uploaded_file = st.file_uploader(
                "Or upload your own image", 
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
        
        if uploaded_file:
            img_pil, img_tensor = preprocess_image(uploaded_file)
            if img_pil is not None and img_tensor is not None:
                st.image(img_pil, use_container_width=True, caption="Input Image")

    with col2:
        if uploaded_file and img_pil is not None and img_tensor is not None:
            st.subheader("Analysis Results")
            
            # Prediction
            with torch.no_grad():
                outputs = model(img_tensor.unsqueeze(0).to(device))
                probs = torch.nn.functional.softmax(outputs, dim=1)
                pred_prob, pred_class = torch.max(probs, 1)
                pred_prob = pred_prob.item()
                pred_class = pred_class.item()
            
            # Prediction card
            st.markdown(f"""
            <div class="prediction-card">
                <h3>Prediction Result</h3>
                <p style="font-size: 1.2rem;">
                    <b>Diagnosis:</b> <span style="color: {CLASS_COLORS[CLASS_ORDER[pred_class]]};">
                    {HUMAN_NAMES[CLASS_ORDER[pred_class]]}</span>
                </p>
                <p style="font-size: 1.2rem;">
                    <b>Confidence:</b> {pred_prob*100:.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.pyplot(plot_probability_distribution(probs))
    
    # Explainability section
    if uploaded_file and img_pil is not None:
        st.markdown("---")
        st.subheader("Model Explanation")
        
        tabs = st.tabs(["Grad-CAM", "Guided Grad-CAM"] if explain_method == "Both" else [explain_method])
        
        if explain_method in ["Grad-CAM", "Both"]:
            with tabs[0]:
                with st.spinner('Generating Grad-CAM visualization...'):
                    cam, _ = grad_cam(model, img_tensor, pred_class)
                    if cam is not None:
                        st.pyplot(create_cam_visualization(img_pil, cam, "Grad-CAM"))
                        with st.expander("Interpretation Guide"):
                            st.markdown("""
                            **Grad-CAM Interpretation:**
                            - Hotter colors (yellow/white) show areas most important for the prediction
                            - Compare with known pathological features:
                                - **Adenocarcinoma**: Look for glandular patterns
                                - **Squamous Cell Carcinoma**: Look for keratin pearls
                                - **Normal Tissue**: Regular cellular structure
                            """)
        
        if explain_method in ["Guided Grad-CAM", "Both"]:
            with tabs[-1] if explain_method == "Both" else tabs[0]:
                with st.spinner('Generating Guided Grad-CAM visualization...'):
                    guided_cam, _ = guided_grad_cam(model, img_tensor, pred_class)
                    if guided_cam is not None:
                        st.pyplot(create_cam_visualization(img_pil, guided_cam, "Guided Grad-CAM"))
                        with st.expander("Interpretation Guide"):
                            st.markdown("""
                            **Guided Grad-CAM Interpretation:**
                            - Shows both high-level and fine-grained important features
                            - More precise than standard Grad-CAM
                            - Bright areas indicate the most influential regions
                            """)
        
        # Debug info
        if debug_mode:
            with st.expander("Debug Information", expanded=False):
                st.write("Model outputs:", outputs.cpu().numpy())
                st.write("Probabilities:", probs.cpu().numpy())
                st.write("Image tensor shape:", img_tensor.shape)
                st.write("Image stats:", f"Mean: {img_tensor.mean().item():.4f}, Std: {img_tensor.std().item():.4f}")

if __name__ == "__main__":
    main()