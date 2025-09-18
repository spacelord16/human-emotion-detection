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
st.set_page_config(page_title="Live Emotion Detector", layout="wide")
st.title("🎭 Live Human Emotion Detector")
st.markdown("---")


# --- Load Trained Model ---
@st.cache_resource
def load_emotion_model():
    """Loads the pre-trained emotion detection model."""
    model = build_model()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()
    return model


model = load_emotion_model()
class_names = ["Angry", "Happy", "Sad"]


# --- Load Face Detection Model ---
@st.cache_resource
def load_face_detector():
    """Load OpenCV's face detection model."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return face_cascade


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


def detect_faces(frame):
    """Detect faces in the frame and return the largest face."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        # Get the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face

        # Add some padding around the face
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)

        face_crop = frame[y : y + h, x : x + w]
        return face_crop, (x, y, w, h)
    return None, None


def predict_emotion(face_image):
    """Predict emotion from face image."""
    if face_image is None:
        return None, None

    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_rgb)

    # Preprocess and predict
    processed_image = preprocess_image(pil_image)

    with torch.no_grad():
        outputs = model(processed_image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_idx = torch.argmax(outputs, 1).item()
        confidence = probabilities[predicted_idx].item()

        return class_names[predicted_idx], confidence


# --- UI Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")

    # Camera controls
    camera_enabled = st.checkbox("Enable Camera", value=False)

    if camera_enabled:
        # Placeholder for camera feed
        camera_placeholder = st.empty()

        # Initialize session state for camera
        if "camera_running" not in st.session_state:
            st.session_state.camera_running = False

        # Start/Stop camera buttons
        col_start, col_stop = st.columns(2)

        with col_start:
            if st.button("🎥 Start Camera", type="primary"):
                st.session_state.camera_running = True

        with col_stop:
            if st.button("⏹️ Stop Camera"):
                st.session_state.camera_running = False

        # Camera feed processing
        if st.session_state.camera_running:
            try:
                # Initialize camera
                cap = cv2.VideoCapture(0)

                if not cap.isOpened():
                    st.error(
                        "❌ Cannot access camera. Please check your camera permissions."
                    )
                    st.session_state.camera_running = False
                else:
                    # Camera loop
                    frame_count = 0
                    while st.session_state.camera_running:
                        ret, frame = cap.read()

                        if not ret:
                            st.error("❌ Failed to read from camera")
                            break

                        # Flip frame horizontally for mirror effect
                        frame = cv2.flip(frame, 1)

                        # Detect face and predict emotion every few frames
                        if (
                            frame_count % 5 == 0
                        ):  # Process every 5th frame for performance
                            face_crop, face_coords = detect_faces(frame)

                            if face_coords is not None:
                                x, y, w, h = face_coords

                                # Draw rectangle around face
                                cv2.rectangle(
                                    frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                                )

                                # Predict emotion
                                emotion, confidence = predict_emotion(face_crop)

                                if emotion is not None:
                                    # Display emotion and confidence
                                    label = f"{emotion}: {confidence:.2%}"
                                    cv2.putText(
                                        frame,
                                        label,
                                        (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 255, 0),
                                        2,
                                    )

                                    # Update session state for sidebar
                                    st.session_state.current_emotion = emotion
                                    st.session_state.current_confidence = confidence

                        # Convert frame to RGB for Streamlit
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(
                            frame_rgb, channels="RGB", use_column_width=True
                        )

                        frame_count += 1
                        time.sleep(0.1)  # Small delay to prevent overwhelming the UI

                # Release camera
                cap.release()

            except Exception as e:
                st.error(f"❌ Camera error: {str(e)}")
                st.session_state.camera_running = False

    else:
        st.info("📱 Check 'Enable Camera' to start live emotion detection")

with col2:
    st.subheader("📊 Detection Results")

    # Display current emotion if available
    if hasattr(st.session_state, "current_emotion") and hasattr(
        st.session_state, "current_confidence"
    ):
        emotion = st.session_state.current_emotion
        confidence = st.session_state.current_confidence

        # Emotion display
        st.markdown(f"### Current Emotion: **{emotion}**")
        st.progress(confidence)
        st.write(f"Confidence: **{confidence:.2%}**")

        # Emotion emoji mapping
        emotion_emojis = {"Happy": "😊", "Sad": "😢", "Angry": "😠"}

        st.markdown(f"## {emotion_emojis.get(emotion, '🤔')}")

        # Color coding based on emotion
        if emotion == "Happy":
            st.success(f"Detected: {emotion}")
        elif emotion == "Sad":
            st.info(f"Detected: {emotion}")
        elif emotion == "Angry":
            st.warning(f"Detected: {emotion}")

    else:
        st.write("👆 Start the camera to see live emotion detection results!")

    st.markdown("---")

    # Statistics section
    st.subheader("📈 Model Information")
    st.write("**Model Architecture:** EfficientNet-B0")
    st.write("**Training Accuracy:** 79.5%")
    st.write("**Validation Accuracy:** 68.5%")
    st.write("**Emotion Classes:** Angry, Happy, Sad")

    # Instructions
    st.markdown("---")
    st.subheader("💡 Instructions")
    st.write("1. Enable camera access when prompted")
    st.write("2. Click 'Start Camera' to begin")
    st.write("3. Position your face in the green rectangle")
    st.write("4. Make different expressions to test!")
    st.write("5. Click 'Stop Camera' when done")

# --- Alternative: Image Upload Section ---
st.markdown("---")
st.subheader("📁 Alternative: Upload Image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col_img, col_result = st.columns([1, 1])

    with col_img:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col_result:
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing emotion..."):
                # Convert PIL to OpenCV format for face detection
                img_array = np.array(image)

                # Detect face
                face_crop, _ = detect_faces(img_array)

                if face_crop is not None:
                    emotion, confidence = predict_emotion(face_crop)

                    if emotion:
                        st.success(f"**Predicted Emotion: {emotion}**")
                        st.write(f"**Confidence: {confidence:.2%}**")

                        # Progress bar for confidence
                        st.progress(confidence)

                        # Emoji display
                        emotion_emojis = {"Happy": "😊", "Sad": "😢", "Angry": "😠"}
                        st.markdown(f"## {emotion_emojis.get(emotion, '🤔')}")
                    else:
                        st.error("Could not analyze emotion")
                else:
                    st.warning(
                        "⚠️ No face detected in the image. Please upload an image with a clear face."
                    )

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit, PyTorch, and OpenCV</p>
        <p><small>Make sure to allow camera access for the best experience!</small></p>
    </div>
    """,
    unsafe_allow_html=True,
)
