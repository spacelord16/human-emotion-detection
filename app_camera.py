import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from model import build_model
from config import DEVICE, IMG_SIZE, MODEL_PATH

# --- App Configuration ---
st.set_page_config(page_title="Live Emotion Detector", layout="wide")
st.title("🎭 Live Human Emotion Detector")
st.markdown(
    """
This app uses your webcam to detect emotions in real-time! 
Make different facial expressions and see how the AI model responds.
"""
)
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


def detect_and_predict_emotion(frame):
    """Detect faces and predict emotions from the frame."""
    if model is None or face_detector is None:
        return frame, None, None

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))

    emotion_results = []

    for x, y, w, h in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract face region with padding
        padding = 10
        face_x1 = max(0, x - padding)
        face_y1 = max(0, y - padding)
        face_x2 = min(frame.shape[1], x + w + padding)
        face_y2 = min(frame.shape[0], y + h + padding)

        face_crop = frame[face_y1:face_y2, face_x1:face_x2]

        if face_crop.size > 0:
            try:
                # Convert BGR to RGB
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_rgb)

                # Preprocess and predict
                processed_image = preprocess_image(pil_image)

                with torch.no_grad():
                    outputs = model(processed_image)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    predicted_idx = torch.argmax(outputs, 1).item()
                    confidence = probabilities[predicted_idx].item()
                    emotion = class_names[predicted_idx]

                    # Draw emotion label
                    label = f"{emotion}: {confidence:.1%}"
                    cv2.putText(
                        frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    emotion_results.append((emotion, confidence))

            except Exception as e:
                st.error(f"Prediction error: {e}")

    return frame, emotion_results


# Initialize session state
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []


# --- WebRTC Video Processor ---
class VideoProcessor:
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Process every 10th frame for better performance
        if self.frame_count % 10 == 0:
            processed_img, emotions = detect_and_predict_emotion(img)

            # Store emotion results in session state
            if emotions:
                st.session_state.current_emotions = emotions
                # Keep history of last 10 detections
                st.session_state.emotion_history.extend(emotions)
                if len(st.session_state.emotion_history) > 10:
                    st.session_state.emotion_history = st.session_state.emotion_history[
                        -10:
                    ]
        else:
            processed_img = img

        self.frame_count += 1

        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


# --- Main UI Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")

    # WebRTC Configuration
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Camera stream
    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        st.success("🎥 Camera is active - make facial expressions!")
    else:
        st.info("👆 Click 'START' to begin live emotion detection")

with col2:
    st.subheader("📊 Live Results")

    # Display current emotions
    if (
        hasattr(st.session_state, "current_emotions")
        and st.session_state.current_emotions
    ):
        for emotion, confidence in st.session_state.current_emotions:
            # Emotion display with emoji
            emotion_emojis = {"Happy": "😊", "Sad": "😢", "Angry": "😠"}
            emoji = emotion_emojis.get(emotion, "🤔")

            st.markdown(f"### {emoji} {emotion}")
            st.progress(confidence)
            st.write(f"Confidence: **{confidence:.1%}**")

            # Color-coded success message
            if emotion == "Happy":
                st.success(f"Detected: {emotion}")
            elif emotion == "Sad":
                st.info(f"Detected: {emotion}")
            elif emotion == "Angry":
                st.warning(f"Detected: {emotion}")

            st.markdown("---")
    else:
        st.write("👆 Start the camera to see real-time emotion detection!")

    # Model info
    st.subheader("🤖 Model Info")
    st.write("**Architecture:** EfficientNet-B0")
    st.write("**Classes:** Angry, Happy, Sad")
    st.write("**Validation Accuracy:** 68.5%")

    # Instructions
    st.subheader("💡 How to Use")
    st.write("1. Click **START** to enable camera")
    st.write("2. Allow browser camera access")
    st.write("3. Position your face in view")
    st.write("4. Make different expressions!")
    st.write("5. See real-time predictions →")

# --- Image Upload Alternative ---
st.markdown("---")
st.subheader("📁 Alternative: Upload an Image")

col_upload1, col_upload2 = st.columns([1, 1])

with col_upload1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col_upload2:
    if uploaded_file is not None:
        if st.button("🔍 Analyze Emotion", type="primary"):
            with st.spinner("Analyzing..."):
                # Convert PIL to numpy array
                img_array = np.array(image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # Detect and predict
                processed_img, emotions = detect_and_predict_emotion(img_bgr)

                if emotions:
                    for emotion, confidence in emotions:
                        st.success(f"**{emotion}** ({confidence:.1%})")

                        emotion_emojis = {"Happy": "😊", "Sad": "😢", "Angry": "😠"}
                        st.markdown(f"## {emotion_emojis.get(emotion, '🤔')}")
                else:
                    st.warning("⚠️ No faces detected in the image")

# --- Emotion History ---
if hasattr(st.session_state, "emotion_history") and st.session_state.emotion_history:
    st.markdown("---")
    st.subheader("📈 Recent Emotion History")

    # Simple emotion counter
    from collections import Counter

    emotion_counts = Counter([e[0] for e in st.session_state.emotion_history[-10:]])

    for emotion, count in emotion_counts.most_common():
        emoji = {"Happy": "😊", "Sad": "😢", "Angry": "😠"}.get(emotion, "🤔")
        st.write(f"{emoji} **{emotion}**: {count} detections")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🚀 Built with Streamlit, PyTorch, and OpenCV</p>
        <p><small>⚠️ Make sure to allow camera access in your browser!</small></p>
    </div>
    """,
    unsafe_allow_html=True,
)
