import os
import shutil
import random


def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):

    files = os.listdir(source_dir)
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(files)

    split_idx = int(len(files) * split_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    # Ensure directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Copy files to train directory
    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))

    # Copy files to validation directory
    for file in val_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, file))

    print(
        f"Split {len(files)} files: {len(train_files)} train, {len(val_files)} validation"
    )


def split_emotion_data(emotions=["Angry", "Happy", "Sad"], split_ratio=0.8):
    """
    Split emotion data for all emotion classes.

    Args:
        emotions (list): List of emotion class names
        split_ratio (float): Ratio of data to use for training (default: 0.8)
    """
    # Create main train and validation directories
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/validation", exist_ok=True)

    # Split each emotion class
    for emotion in emotions:
        source = f"data/data/{emotion}"
        train_dest = f"data/train/{emotion}"
        val_dest = f"data/validation/{emotion}"

        if os.path.exists(source):
            print(f"Splitting {emotion} data...")
            split_data(source, train_dest, val_dest, split_ratio)
        else:
            print(f"Warning: Source directory {source} not found")


if __name__ == "__main__":
    # Set random seed for reproducible splits
    random.seed(42)

    # Split the data
    split_emotion_data()

    print("\nData splitting completed!")
