import os
import cv2
import PIL.Image as Image
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0, mobilenet_v2

st.set_page_config(
    page_title="SkinSync | Multi-Model AI Diagnostics",
    page_icon="✨",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header { font-size:2.5rem; font-weight:700; text-align:center; color: #1E293B; }
    .sub-header { font-size:1.0rem; text-align:center; color: #64748B; margin-bottom:1.5rem; }
    .disclaimer { font-size:0.8rem; text-align:center; color: #94A3B8; margin-bottom:2rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">✨ SkinSync Diagnostic Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Attribute AI Analysis for Skin Type & Clinical Features</div>', unsafe_allow_html=True)
st.markdown('<div class="disclaimer">⚠️ FOR EDUCATIONAL/DEMONSTRATION PURPOSES ONLY. CONSULT A DERMATOLOGIST FOR CLINICAL DIAGNOSES.</div>', unsafe_allow_html=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSES = {
    "Skin Type": ['Dry_Skin', 'Normal_Skin', 'Oily_Skin'],
    "Acne": ['Acne', 'Normal', 'Other'],
    "Wrinkles": ['Normal', 'Other', 'Wrinkles'],
    "Blackheads": ['Blackheads', 'Normal', 'Other']
}

transforms_dict = {
    "Skin Type": transforms.Compose([
        transforms.Resize((228, 228)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "Standard": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

@st.cache_resource
def load_all_models():
    models = {}

    st_model = efficientnet_b0()
    st_model.classifier[1] = nn.Linear(st_model.classifier[1].in_features, 3)
    st_model.load_state_dict(torch.load('best_skin_type_model.pth', map_location=device))
    st_model.eval().to(device)
    models["Skin Type"] = st_model

    acne_m = mobilenet_v2()
    acne_m.classifier[1] = nn.Linear(acne_m.classifier[1].in_features, 3)
    acne_m.load_state_dict(torch.load('best_acne_detection_model.pth', map_location=device))
    acne_m.eval().to(device)
    models["Acne"] = acne_m

    wrinkle_m = mobilenet_v2()
    wrinkle_m.classifier[1] = nn.Linear(wrinkle_m.classifier[1].in_features, 3)
    wrinkle_m.load_state_dict(torch.load('best_wrinkles_detection_model.pth', map_location=device))
    wrinkle_m.eval().to(device)
    models["Wrinkles"] = wrinkle_m

    blackhead_m = mobilenet_v2()
    blackhead_m.classifier[1] = nn.Linear(blackhead_m.classifier[1].in_features, 3)
    blackhead_m.load_state_dict(torch.load('best_blackhead_detection_model.pth', map_location=device))
    blackhead_m.eval().to(device)
    models["Blackheads"] = blackhead_m

    return models

with st.spinner("Loading AI diagnostic models into memory..."):
    try:
        models = load_all_models()
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()

st.sidebar.header("⚙️ Portal Settings")
show_probs = st.sidebar.checkbox("Show Detailed Probabilities", value=True)
confidence_threshold = st.sidebar.slider("Confidence Warning Threshold (%)", 30, 80, 50)

uploaded_file = st.file_uploader("Upload a facial photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.image(image, caption="Uploaded Scan", use_container_width=True)

    if st.button("🚀 Run Full Diagnostic Pipeline", use_container_width=True):
        with st.spinner("Analyzing multi-model features..."):
            
            results = {}
            for task_name, model in models.items():
                transform = transforms_dict["Skin Type"] if task_name == "Skin Type" else transforms_dict["Standard"]
                img_t = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(img_t)
                    probs = F.softmax(outputs, dim=1)[0]
                    conf, pred = torch.max(probs, 0)
                
                label = CLASSES[task_name][pred.item()]
                results[task_name] = {
                    "label": label.replace("_", " "),
                    "confidence": conf.item() * 100,
                    "probs": probs,
                    "classes": CLASSES[task_name]
                }

        st.write("---")
        st.subheader("📊 Diagnostic Suite Results")
        
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        
        grid = [
            (row1_col1, "Skin Type"),
            (row1_col2, "Acne"),
            (row2_col1, "Wrinkles"),
            (row2_col2, "Blackheads")
        ]

        for col, task in grid:
            res = results[task]
            with col:
                st.markdown(f"### {task}")
                if res["confidence"] < confidence_threshold:
                    st.warning(f"⚠️ **Low Confidence ({res['confidence']:.1f}%)**\n\nPredicted: **{res['label']}**")
                else:
                    st.metric(
                        label="Classification",
                        value=res["label"],
                        delta=f"{res['confidence']:.1f}% Confidence"
                    )
                
                if show_probs:
                    for i, name in enumerate(res["classes"]):
                        st.progress(
                            float(res["probs"][i]),
                            text=f"{name.replace('_', ' ')}: {res['probs'][i]*100:.1f}%"
                        )
                st.write("")

        st.success("✨ Multi-Model Analysis Complete!")