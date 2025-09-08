# Human Emotion Detection

A machine learning project for detecting human emotions from images using PyTorch.

## Project Structure

```
human_emotion_detection/
├── check_gpu.py       # GPU availability checker
├── config.py          # Configuration settings
├── data_loader.py     # Data loading utilities
├── evaluate.py        # Model evaluation
├── model.py           # Neural network model
├── train.py           # Training script
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Check GPU availability:

```bash
python check_gpu.py
```

3. Download dataset:
   The project uses a human emotions dataset. Download it manually and place it in the `data/` directory with the following structure:

```
data/
├── Angry/
├── Happy/
└── Sad/
```

## Usage

### Training

```bash
python train.py
```

### Evaluation

```bash
python evaluate.py
```

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- Other dependencies listed in requirements.txt

## Dataset

The model is trained on images categorized into three emotions:

- Angry
- Happy
- Sad

## Model

The project implements a convolutional neural network for emotion classification using PyTorch.

## Notes

- Ensure you have sufficient GPU memory for training
- The dataset is not included in the repository due to size constraints
- Configure model parameters in config.py before training
