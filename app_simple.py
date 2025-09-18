import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import time
from model import build_model
from config import DEVICE, IMG_SIZE, MODEL_PATH

# --- App Configuration ---
st.set_page_config(page_title="Simple Emotion Detector", layout="wide")
st.title("🎭 Simple Emotion Detector")
st.markdown("**Lightweight version for better performance**")
st.markdown("---")


# --- Load Trained Model ---
@st.cache_resource
def load_emotion_model():
    """Loads the pre-trained emotion detection model."""
    try:
        model = build_model()
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
        model.to(DEVICE)
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


model = load_emotion_model()
class_names = ["Angry", "Happy", "Sad"]


# --- Load Face Detection Model ---
@st.cache_resource
def load_face_detector():
    """Load OpenCV's face detection model."""
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return face_cascade
    except:
        st.error("❌ Could not load face detection model")
        return None


face_detector = load_face_detector()


# --- Image Preprocessing ---
def preprocess_image(image):
    """Preprocesses the image for the model."""
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0).to(DEVICE)


def predict_emotion_from_image(image):
    """Predict emotion from PIL image."""
    if model is None:
        return None, None

    try:
        processed_image = preprocess_image(image)
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_idx = torch.argmax(outputs, 1).item()
            confidence = probabilities[predicted_idx].item()
            emotion = class_names[predicted_idx]
            return emotion, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None


# --- Main UI ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("📊 Results")

    if uploaded_file is not None:
        if st.button("🔍 Analyze Emotion", type="primary"):
            with st.spinner("Analyzing..."):
                emotion, confidence = predict_emotion_from_image(image)

                if emotion:
                    # Emotion display with emoji
                    emotion_emojis = {"Happy": "😊", "Sad": "😢", "Angry": "😠"}
                    emoji = emotion_emojis.get(emotion, "🤔")

                    st.markdown(f"## {emoji} **{emotion}**")
                    st.progress(confidence)
                    st.write(f"**Confidence: {confidence:.1%}**")

                    # Color-coded result
                    if emotion == "Happy":
                        st.success(f"Detected: {emotion}")
                    elif emotion == "Sad":
                        st.info(f"Detected: {emotion}")
                    elif emotion == "Angry":
                        st.warning(f"Detected: {emotion}")
                else:
                    st.error("Could not analyze emotion")
    else:
        st.info("👆 Upload an image to analyze emotions")

# --- Live Camera Section (Simplified) ---
st.markdown("---")
st.subheader("📸 Take a Photo")

# Simple camera input
camera_input = st.camera_input("Take a picture for emotion detection")

if camera_input is not None:
    # Convert to PIL Image
    camera_image = Image.open(camera_input).convert("RGB")

    col_cam1, col_cam2 = st.columns([1, 1])

    with col_cam1:
        st.image(camera_image, caption="Captured Image", use_column_width=True)

    with col_cam2:
        if st.button("🔍 Analyze Camera Image", type="primary"):
            with st.spinner("Analyzing emotion..."):
                emotion, confidence = predict_emotion_from_image(camera_image)

                if emotion:
                    emotion_emojis = {"Happy": "😊", "Sad": "😢", "Angry": "😠"}
                    emoji = emotion_emojis.get(emotion, "🤔")

                    st.markdown(f"## {emoji} **{emotion}**")
                    st.progress(confidence)
                    st.write(f"**Confidence: {confidence:.1%}**")

                    if emotion == "Happy":
                        st.success(f"Detected: {emotion}")
                    elif emotion == "Sad":
                        st.info(f"Detected: {emotion}")
                    elif emotion == "Angry":
                        st.warning(f"Detected: {emotion}")
                else:
                    st.error("Could not analyze emotion")

# --- Model Info Sidebar ---
with st.sidebar:
    st.subheader("🤖 Model Information")
    st.write("**Architecture:** EfficientNet-B0")
    st.write("**Classes:** Angry, Happy, Sad")
    st.write("**Validation Accuracy:** 68.5%")
    st.write("**Training Accuracy:** 79.5%")

    st.markdown("---")
    st.subheader("💡 Tips")
    st.write("• Good lighting helps accuracy")
    st.write("• Clear facial expressions work best")
    st.write("• Try exaggerated emotions")
    st.write("• Face should be clearly visible")

    st.markdown("---")
    st.subheader("⚡ Performance")
    st.write("This simplified version:")
    st.write("• No live streaming")
    st.write("• Single image analysis")
    st.write("• Faster processing")
    st.write("• Better for slower devices")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>⚡ Lightweight Emotion Detection</p>
        <p><small>Upload images or take photos for emotion analysis!</small></p>
    </div>
    """,
    unsafe_allow_html=True,
)
