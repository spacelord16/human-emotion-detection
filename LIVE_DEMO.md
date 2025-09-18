# Live Emotion Detection Demo 🎭

This guide will help you run the live emotion detection app with webcam support.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Live Camera App

```bash
streamlit run app_camera.py
```

### 3. Alternative Simple Version

If you have issues with WebRTC, try the simpler version:

```bash
streamlit run app_live.py
```

## 📋 Features

### 🎥 Live Camera Detection

- **Real-time emotion detection** from your webcam
- **Face detection** with bounding boxes
- **Confidence scores** for each prediction
- **Multiple face support** - detects emotions for multiple people
- **Emotion history** tracking

### 📊 Enhanced UI

- **Live preview** with emotion overlays
- **Confidence progress bars**
- **Emoji displays** for each emotion
- **Model performance metrics**
- **Alternative image upload** option

### 🎯 Supported Emotions

- **Happy** 😊
- **Sad** 😢
- **Angry** 😠

## 🔧 Technical Details

### Model Architecture

- **Base Model:** EfficientNet-B0 (pre-trained)
- **Fine-tuned** on emotion dataset
- **Validation Accuracy:** 68.5%
- **Training Accuracy:** 79.5%

### Dependencies

- **Streamlit** - Web interface
- **OpenCV** - Face detection and image processing
- **PyTorch** - Deep learning inference
- **streamlit-webrtc** - WebRTC camera streaming
- **Pillow** - Image manipulation

## 🚨 Troubleshooting

### Camera Access Issues

1. **Allow camera permissions** in your browser
2. **Use HTTPS** for WebRTC (use `streamlit run --server.enableCORS=false --server.enableXsrfProtection=false`)
3. **Try different browsers** (Chrome works best)

### Model Loading Issues

1. **Check if `emotion_detection_model.pth` exists**
2. **Verify PyTorch installation** with GPU/CPU compatibility
3. **Run `python check_gpu.py`** to verify device setup

### Performance Issues

1. **Lower frame processing rate** (modify `frame_count % 3` to higher number)
2. **Reduce image size** in config.py
3. **Close other applications** using camera

## 🎮 Usage Tips

### For Best Results

1. **Good lighting** - ensure your face is well-lit
2. **Clear view** - position face clearly in camera
3. **Stable position** - minimize camera shake
4. **Exaggerated expressions** - the model works better with clear emotions

### Testing Different Emotions

- **Happy:** Smile broadly, raise eyebrows
- **Sad:** Frown, lower eyebrows, droopy eyes
- **Angry:** Furrow brow, tense jaw, narrow eyes

## 📈 Model Performance

The model was trained on 528 images across 3 emotion categories:

- Training for 10 epochs
- Best validation accuracy: **68.5%**
- Uses transfer learning from EfficientNet-B0
- Real-time inference optimized for webcam use

## 🔮 Future Enhancements

Potential improvements for the live demo:

- **More emotion classes** (Fear, Surprise, Disgust, Neutral)
- **Age and gender detection**
- **Emotion intensity scoring**
- **Video recording** with emotion timeline
- **Multi-person emotion tracking**
- **Emotion analytics dashboard**

## 📞 Support

If you encounter issues:

1. Check the console for error messages
2. Verify all dependencies are installed
3. Test camera access in browser settings
4. Try the alternative app versions

---

**Built with ❤️ using Streamlit, PyTorch, and OpenCV**
