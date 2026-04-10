import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import streamlit as st
from PIL import Image
from predict import predict_image, load_model_for_inference
from utils import get_device, get_model_path


# page config
st.set_page_config(page_title="Waste Classifier", layout="centered")

st.title("♻️ Waste Image Classifier")
st.write("Upload an image and choose a model to classify it.")

# model selection
model_option = st.selectbox(
    "Select Model",
    ["ResNet18 (exp4)", "ResNet34 (exp6)"]
)
st.info(f"Using Model: {model_option}")

# load Model (Cached)
@st.cache_resource
def load_model_cached(model_name, exp_name):
    device = get_device()
    model = load_model_for_inference(
        model_name=model_name,
        model_path=get_model_path(exp_name),
        num_classes=4,
        device=device
    )
    return model, device

# model mapping -> config
if model_option == "ResNet18 (exp4)":
    model_name = "resnet18"
    exp_name = "exp4_resnet18_balanced_finetune_differential_lrs_weight_decay"
else:
    model_name = "resnet34"
    exp_name = "exp6_resnet34_config"

if "loaded_model_name" not in st.session_state or st.session_state.loaded_model_name != model_name:
    with st.spinner("Loading model..."):
        model, device = load_model_cached(model_name, exp_name)
        st.session_state.loaded_model_name = model_name
else:
    model, device = load_model_cached(model_name, exp_name)

# classes
class_names = ["glass", "metal", "paper", "plastic"]

# file upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width="content")

    # Predictions
    top_class, top_conf, results = predict_image(
        image,
        model,
        device,
        class_names
    )

    # Display Results
    st.markdown("## 🧠 Prediction")
    st.markdown(f"## 🟢 {top_class.upper()}")
    st.markdown(f"### Confidence: {top_conf*100:.2f}%")
    

    st.markdown("### 📊 Top Predictions")
    for r in results:
        st.write(f"{r['class']}: {r['confidence']*100:.2f}%")
        st.progress(float(r['confidence']))