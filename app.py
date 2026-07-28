import PIL.Image as Image
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import pandas as pd
from torchvision import transforms
from torchvision.models import efficientnet_b0, mobilenet_v2
from transformer_recommender import SkinCareAttentionTransformer

st.set_page_config(
    page_title="SkinSync | AI Diagnostic Portal",
    page_icon="✨",
    layout="wide"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = {
    "Skin Type": ['Dry_Skin', 'Normal_Skin', 'Oily_Skin'],
    "Acne": ['Acne', 'Normal/Other'],
    "Wrinkles": ['Normal', 'Other', 'Wrinkles'],
    "Blackheads": ['Blackheads', 'Normal', 'Other']
}

PRODUCT_DATABASE = [
    {"name": "CeraVe SA Cleanser", "brand": "CeraVe", "price_tier": "$", "ingredients": ["Salicylic Acid", "Ceramides", "Niacinamide"], "category": "Cleanser"},
    {"name": "The Ordinary Niacinamide 10% + Zinc 1%", "brand": "The Ordinary", "price_tier": "$", "ingredients": ["Niacinamide", "Zinc PCA"], "category": "Serum"},
    {"name": "Paula's Choice 2% BHA Liquid Exfoliant", "brand": "Paula's Choice", "price_tier": "$$", "ingredients": ["Salicylic Acid", "Benzoyl Peroxide"], "category": "Exfoliant"},
    {"name": "La Roche-Posay Hyalu B5 Serum", "brand": "La Roche-Posay", "price_tier": "$$", "ingredients": ["Hyaluronic Acid", "Centella Asiatica"], "category": "Serum"},
    {"name": "SkinCeuticals Retinol 0.5", "brand": "SkinCeuticals", "price_tier": "$$$", "ingredients": ["Retinol", "Ceramides"], "category": "Night Treatment"},
    {"name": "Drunk Elephant TLC Framboos Glycolic Serum", "brand": "Drunk Elephant", "price_tier": "$$$", "ingredients": ["Glycolic Acid", "Salicylic Acid"], "category": "Serum"}
]

transforms_dict = {
    "Skin Type": transforms.Compose([transforms.Resize((228, 228)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "Standard": transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}
@st.cache_resource
def load_vision_models():
    models = {}
    
    st_m = efficientnet_b0()
    st_m.classifier[1] = nn.Linear(st_m.classifier[1].in_features, 3)
    st_m.load_state_dict(torch.load('best_skin_type_model.pth', map_location=device))
    st_m.eval().to(device)
    models["Skin Type"] = st_m

    acne_m = mobilenet_v2()
    acne_m.classifier[1] = nn.Linear(acne_m.classifier[1].in_features, 2)
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

@st.cache_resource
def load_recommender_artifacts():
    encoder = joblib.load("feature_encoder.pkl")
    mlb = joblib.load("ingredient_binarizer.pkl")
    
    input_dim = len(encoder.get_feature_names_out())
    output_dim = len(mlb.classes_)
    
    recommender = SkinCareAttentionTransformer(input_dim=input_dim, output_dim=output_dim)
    recommender.load_state_dict(torch.load("transformer_recommender.pth", map_location=device))
    recommender.eval().to(device)
    
    return recommender, encoder, mlb

models = load_vision_models()
try:
    recommender, encoder, mlb = load_recommender_artifacts()
except Exception as e:
    recommender, encoder, mlb = None, None, None

st.title("✨ SkinSync Interactive Diagnostic & Recommendation Portal")
st.caption("Multi-Vision Scan + Custom Self-Attention Transformer Recommender")

st.sidebar.header("⚙️ Portal Settings")
confidence_threshold = st.sidebar.slider("Confidence Warning Threshold (%)", 30, 80, 50)

uploaded_file = st.file_uploader("Upload a clear facial photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Scan", use_container_width=True)

    if st.button("🚀 Run Diagnostic Pipeline", use_container_width=True):
        st.session_state['scan_complete'] = True
        results = {}
        for task_name in ["Skin Type", "Acne", "Wrinkles", "Blackheads"]:
            m = models[task_name]
            t = transforms_dict["Skin Type"] if task_name == "Skin Type" else transforms_dict["Standard"]
            img_t = t(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = m(img_t)
                probs = F.softmax(outputs, dim=1)[0]
                conf, pred = torch.max(probs, 0)
            
            label = CLASSES[task_name][pred.item()]
            results[task_name] = {
                "label": label.replace("_", " "),
                "confidence": conf.item() * 100,
                "probs": probs
            }
        st.session_state['results'] = results

if st.session_state.get('scan_complete', False):
    st.write("---")
    st.subheader("Step 1: Review Vision Scan & Adjust Parameters")
    st.info("💡 Review the model detections below. Adjust parameters manually if confidence is low or if you disagree.")

    results = st.session_state['results']
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown("### Skin Type")
        st.caption("Disclaimer: Visual scans only detect extreme dryness/oiliness.")
        curr_st = results["Skin Type"]["label"]
        st_choice = st.selectbox(
            "Confirm Skin Type:", 
            ["Dry Skin", "Normal Skin", "Oily Skin", "Combination"], 
            index=0 if "Dry" in curr_st else (2 if "Oily" in curr_st else 1)
        )
        if results["Skin Type"]["confidence"] < confidence_threshold:
            st.warning(f"⚠️ Low confidence ({results['Skin Type']['confidence']:.1f}%)")

    with c2:
        st.markdown("### Acne")
        acne_choice = st.checkbox("Has Acne?", value=(results["Acne"]["label"] == "Acne"))
        acne_type = st.selectbox("Acne Type:", ["Comedonal", "Inflammatory", "Nodular", "Cystic"]) if acne_choice else "General"
        if results["Acne"]["confidence"] < confidence_threshold:
            st.warning(f"⚠️ Low confidence ({results['Acne']['confidence']:.1f}%)")

    with c3:
        st.markdown("### Wrinkles")
        wrinkle_choice = st.checkbox("Has Wrinkles/Fine Lines?", value=(results["Wrinkles"]["label"] == "Wrinkles"))
        if results["Wrinkles"]["confidence"] < confidence_threshold:
            st.warning(f"⚠️ Low confidence ({results['Wrinkles']['confidence']:.1f}%)")

    with c4:
        st.markdown("### Blackheads")
        bh_choice = st.checkbox("Has Blackheads?", value=(results["Blackheads"]["label"] == "Blackheads"))
        if results["Blackheads"]["confidence"] < confidence_threshold:
            st.warning(f"⚠️ Low confidence ({results['Blackheads']['confidence']:.1f}%)")

    st.write("---")
    st.subheader("Step 2: Additional Personalization & Constraints")
    
    q1, q2, q3 = st.columns(3)
    with q1:
        age_group = st.selectbox("Age Group:", ["14-18", "19-25", "26-35", "36-50", "50+"], index=1)
    with q2:
        is_sensitive = st.selectbox("Is your skin sensitive?", ["Yes", "No"])
    with q3:
        budget = st.select_slider("Target Budget Tier:", options=["$", "$$", "$$$"], value="$$")

    all_ingredients = list(mlb.classes_) if mlb else ["Salicylic Acid", "Niacinamide", "Retinol"]
    allergies = st.multiselect("Select Ingredients to EXCLUDE (Allergies/Sensitivities):", options=all_ingredients)

    if st.button("🧪 Generate Customized Recommendations", use_container_width=True):
        st.write("---")
        st.subheader("Step 3: Transformer-Generated Recommendations")
        
        if acne_choice:
            primary_concern = "Acne"
            internal_type = acne_type
        elif bh_choice:
            primary_concern = "Whiteheads / Blackheads"
            internal_type = "General"
        elif wrinkle_choice:
            primary_concern = "Wrinkles"
            internal_type = "General"
        else:
            primary_concern = "Dullness"
            internal_type = "General"

        base_skin = st_choice.split()[0]
        subtype = "Normal to Dry" if "Dry" in base_skin else ("Normal to Oily" if "Oily" in base_skin else "Normal")

        input_data = pd.DataFrame([{
            'Age_Group': age_group,
            'Skin_Type': base_skin,
            'Skin_Subtype': subtype,
            'Sensitivity': is_sensitive,
            'Concern': primary_concern,
            'Internal_Type': internal_type
        }])
        
        encoded_input = encoder.transform(input_data)
        input_tensor = torch.tensor(encoded_input, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            scores = recommender(input_tensor)[0]
            
        ranked = []
        for idx, score in enumerate(scores):
            ing_name = mlb.classes_[idx]
            if ing_name not in allergies:
                ranked.append((ing_name, float(score) * 100))
                
        ranked.sort(key=lambda x: x[1], reverse=True)
        top_ingredients = ranked[:3]

        st.markdown("### Top Active Ingredients (Transformer Attention Matches)")
        ing_cols = st.columns(len(top_ingredients))
        for i, (ing, affinity) in enumerate(top_ingredients):
            with ing_cols[i]:
                st.metric(label=f"Active Key #{i+1}", value=ing, delta=f"{affinity:.1f}% Score")

        st.markdown("---")
        st.markdown("### Matched Products")
        target_names = [x[0] for x in top_ingredients]
        
        matched_products = [
            p for p in PRODUCT_DATABASE 
            if any(ing in p["ingredients"] for ing in target_names) 
            and not any(a in allergies for a in p["ingredients"])
            and p["price_tier"] == budget
        ]

        if matched_products:
            p_cols = st.columns(len(matched_products))
            for i, prod in enumerate(matched_products):
                with p_cols[i % len(p_cols)]:
                    st.success(f"**{prod['name']}**")
                    st.caption(f"Brand: {prod['brand']} | Category: {prod['category']}")
                    st.write(f"**Price Tier:** {prod['price_tier']}")
                    st.write(f"**Key Ingredients:** {', '.join(prod['ingredients'])}")
        else:
            st.info(f"No products found in the **{budget}** price tier matching your exact active ingredients. Try adjusting your budget slider.")