import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load trained model
model = YOLO("runs/classify/train/weights/best.pt")

st.title("🖼️ Image Quality Detector")
st.write("Detects: Normal, Blurred, Edge-cut, Half-visible images")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict
    results = model(img)
    pred_class = results[0].names[results[0].probs.top1]
    confidence = results[0].probs.top1conf.item()

    st.success(f"Prediction: **{pred_class}** ({confidence:.2f} confidence)")
