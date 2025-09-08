import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES

def build_model():
    """Builds an EfficientNet model for emotion detection."""
    # Load a pre-trained EfficientNet-B0 model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze all the parameters in the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classifier layer for our specific task
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    
    print("Model built successfully!")
    return model