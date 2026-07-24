import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, mobilenet_v2
from PIL import Image
import torch.nn.functional as F

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SkinSync | AI Diagnostics",
    page_icon="✨",
    layout="centered"
)

# --- STYLING ---
st.markdown("""
    <style>
    .main-header { font-size:2.3rem; font-weight:700; text-align:center; color: #333; }
    .sub-header { font-size:1.0rem; text-align:center; color: #666; margin-bottom:1.5rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">✨ SkinSync Diagnostic Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a facial image for instant Skin Type & Condition analysis</div>\nTHIS IS NOT A DIAGNOSIS TOOL. ALWAYS CONSULT A DERMATOLOGIST BEFORE STARTING ANY SKIN TREATMENTS.', unsafe_allow_html=True)

# --- DEVICE ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- CLASS MAPPINGS ---
SKIN_TYPE_CLASSES = ['Dry_Skin', 'Normal_Skin', 'Oily_Skin']
ACNE_CLASSES = ['Acne', 'Normal', 'Other']

# --- TRANSFORMS ---
skin_type_transform = transforms.Compose([
    transforms.Resize((228, 228)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

acne_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- MODEL LOADERS ---
@st.cache_resource
def load_skin_type_model():
    model = efficientnet_b0()
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 3)
    )
    model.load_state_dict(torch.load('best_skin_type_model.pth', map_location=device))
    model.eval().to(device)
    return model

@st.cache_resource
def load_acne_model():
    model = mobilenet_v2()
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 3)
    )
    model.load_state_dict(torch.load('best_acne_detection_model.pth', map_location=device))
    model.eval().to(device)
    return model

with st.spinner("Loading models into memory..."):
    try:
        skin_model = load_skin_type_model()
        acne_model = load_acne_model()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# --- UI CONTROLS ---
show_probs = st.sidebar.checkbox("Show Detailed Probabilities", value=True)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Run Diagnostic Analysis", use_container_width=True):
        with st.spinner("Analyzing image..."):
            
            # 1. SKIN TYPE
            t_skin = skin_type_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                out_skin = skin_model(t_skin)
                probs_skin = F.softmax(out_skin, dim=1)[0]
                conf_skin, pred_skin = torch.max(probs_skin, 0)
            skin_label = SKIN_TYPE_CLASSES[pred_skin.item()]
            skin_conf = conf_skin.item() * 100

            # 2. ACNE
            t_acne = acne_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                out_acne = acne_model(t_acne)
                probs_acne = F.softmax(out_acne, dim=1)[0]
                conf_acne, pred_acne = torch.max(probs_acne, 0)
            acne_label = ACNE_CLASSES[pred_acne.item()]
            acne_conf = conf_acne.item() * 100

        # --- RESULTS DISPLAY ---
        st.subheader("📊 Analysis Results")
        
        CONFIDENCE_THRESHOLD = 50.0
        r_col1, r_col2 = st.columns(2)

        # Skin Type Column
        with r_col1:
            if skin_conf < CONFIDENCE_THRESHOLD:
                st.warning(f"⚠️ **Low Confidence ({skin_conf:.1f}%)**\n\nLighting or texture is ambiguous. Top guess: **{skin_label.replace('_', ' ')}**")
            else:
                st.metric(
                    label="Skin Type",
                    value=skin_label.replace("_", " "),
                    delta=f"{skin_conf:.1f}% Confidence"
                )
            if show_probs:
                for i, name in enumerate(SKIN_TYPE_CLASSES):
                    st.progress(float(probs_skin[i]), text=f"{name.replace('_', ' ')}: {probs_skin[i]*100:.1f}%")

        # Acne Column
        with r_col2:
            st.metric(
                label="Acne Condition",
                value=acne_label.replace("_", " "),
                delta=f"{acne_conf:.1f}% Confidence"
            )
            if show_probs:
                for i, name in enumerate(ACNE_CLASSES):
                    st.progress(float(probs_acne[i]), text=f"{name.replace('_', ' ')}: {probs_acne[i]*100:.1f}%")

        st.success("Diagnostics Complete!")