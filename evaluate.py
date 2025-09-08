# app.py
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import build_model  # Assuming your model definition is in model.py
from config import DEVICE, IMG_SIZE, MODEL_PATH

# --- App Configuration ---
st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("Human Emotion Detector 🧠")
st.write(
    "Upload an image and the model will predict the emotion: **Angry, Happy, or Sad**."
)


# --- Load Trained Model ---
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_emotion_model():
    """Loads the pre-trained emotion detection model."""
    model = build_model()
    # Load the saved model state dictionary, ensuring it's mapped to the correct device
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()  # Set the model to evaluation mode
    return model


model = load_emotion_model()
class_names = [
    "angry",
    "happy",
    "sad",
]  # Make sure this order matches your training data


# --- Image Preprocessing ---
def preprocess_image(image):
    """Preprocesses the uploaded image for the model."""
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # Add a batch dimension (B, C, H, W) and send to device
    return transform(image).unsqueeze(0).to(DEVICE)


# --- Streamlit UI ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")

    # Predict button
    if st.button("Predict Emotion"):
        with st.spinner("Analyzing the image..."):
            # Preprocess the image and get prediction
            processed_image = preprocess_image(image)
            with torch.no_grad():
                outputs = model(processed_image)
                _, predicted_idx = torch.max(outputs, 1)
                predicted_emotion = class_names[predicted_idx.item()]

            st.success(f"**Predicted Emotion: {predicted_emotion.capitalize()}**")
