import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DATA_PATH = "./data"
MODEL_PATH = "./emotion_detection_model.pth"

IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NUM_CLASSES = 3
