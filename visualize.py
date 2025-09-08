import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime


def create_output_folder():
    """Create output folder with timestamp for saving plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/training_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_training_history(train_losses, val_losses, train_accs, val_accs, output_dir):
    """
    Plot training and validation loss and accuracy curves.

    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        train_accs (list): Training accuracies per epoch
        val_accs (list): Validation accuracies per epoch
        output_dir (str): Directory to save plots
    """
    plt.style.use("seaborn-v0_8")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Loss
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Accuracy
    ax2.plot(epochs, train_accs, "b-", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs, val_accs, "r-", label="Validation Accuracy", linewidth=2)
    ax2.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_final_metrics(train_losses, val_losses, train_accs, val_accs, output_dir):
    """
    Plot final training metrics summary.

    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        train_accs (list): Training accuracies per epoch
        val_accs (list): Validation accuracies per epoch
        output_dir (str): Directory to save plots
    """
    plt.style.use("seaborn-v0_8")

    # Create metrics summary
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Loss comparison
    categories = ["Training", "Validation"]
    final_losses = [train_losses[-1], val_losses[-1]]
    bars1 = ax1.bar(categories, final_losses, color=["skyblue", "lightcoral"])
    ax1.set_title("Final Loss Comparison", fontweight="bold")
    ax1.set_ylabel("Loss")
    for i, v in enumerate(final_losses):
        ax1.text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom")

    # Accuracy comparison
    final_accs = [train_accs[-1], val_accs[-1]]
    bars2 = ax2.bar(categories, final_accs, color=["lightgreen", "gold"])
    ax2.set_title("Final Accuracy Comparison", fontweight="bold")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    for i, v in enumerate(final_accs):
        ax2.text(i, v + 0.02, f"{v:.4f}", ha="center", va="bottom")

    # Training progress
    epochs = range(1, len(train_losses) + 1)
    ax3.plot(epochs, train_losses, "o-", color="blue", alpha=0.7, label="Train Loss")
    ax3.plot(epochs, val_losses, "s-", color="red", alpha=0.7, label="Val Loss")
    ax3.set_title("Loss Progress", fontweight="bold")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Loss")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Accuracy progress
    ax4.plot(epochs, train_accs, "o-", color="green", alpha=0.7, label="Train Acc")
    ax4.plot(epochs, val_accs, "s-", color="orange", alpha=0.7, label="Val Acc")
    ax4.set_title("Accuracy Progress", fontweight="bold")
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("Accuracy")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_data_distribution(dataset_sizes, class_names, output_dir):
    """
    Plot dataset distribution.

    Args:
        dataset_sizes (dict): Dictionary with train/validation sizes
        class_names (list): List of class names
        output_dir (str): Directory to save plots
    """
    plt.style.use("seaborn-v0_8")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Dataset split pie chart
    sizes = [dataset_sizes["train"], dataset_sizes["validation"]]
    labels = ["Training", "Validation"]
    colors = ["lightblue", "lightcoral"]

    ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax1.set_title("Dataset Split Distribution", fontweight="bold")

    # Class distribution (assuming equal distribution)
    class_counts = [dataset_sizes["train"] // len(class_names)] * len(class_names)
    bars = ax2.bar(class_names, class_counts, color=["red", "green", "blue"])
    ax2.set_title("Class Distribution (Training)", fontweight="bold")
    ax2.set_ylabel("Number of Images")
    ax2.set_xlabel("Emotion Classes")

    for i, v in enumerate(class_counts):
        ax2.text(i, v + 1, str(v), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/data_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_training_summary(
    train_losses, val_losses, train_accs, val_accs, best_acc, training_time, output_dir
):
    """
    Save training summary to text file.

    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        train_accs (list): Training accuracies per epoch
        val_accs (list): Validation accuracies per epoch
        best_acc (float): Best validation accuracy
        training_time (float): Total training time in seconds
        output_dir (str): Directory to save summary
    """
    summary_path = f"{output_dir}/training_summary.txt"

    with open(summary_path, "w") as f:
        f.write("EMOTION DETECTION MODEL - TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write(
            f"Training completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(
            f"Total training time: {training_time // 60:.0f}m {training_time % 60:.0f}s\n"
        )
        f.write(f"Total epochs: {len(train_losses)}\n\n")

        f.write("FINAL METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Best validation accuracy: {best_acc:.4f}\n")
        f.write(f"Final training loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final validation loss: {val_losses[-1]:.4f}\n")
        f.write(f"Final training accuracy: {train_accs[-1]:.4f}\n")
        f.write(f"Final validation accuracy: {val_accs[-1]:.4f}\n\n")

        f.write("EPOCH-BY-EPOCH RESULTS:\n")
        f.write("-" * 30 + "\n")
        f.write("Epoch | Train Loss | Val Loss | Train Acc | Val Acc\n")
        f.write("-" * 55 + "\n")

        for i in range(len(train_losses)):
            f.write(
                f"{i+1:5d} | {train_losses[i]:10.4f} | {val_losses[i]:8.4f} | "
                f"{train_accs[i]:9.4f} | {val_accs[i]:7.4f}\n"
            )


def generate_all_plots(
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    dataset_sizes,
    class_names,
    best_acc,
    training_time,
):
    """
    Generate all visualization plots and save to output folder.

    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        train_accs (list): Training accuracies per epoch
        val_accs (list): Validation accuracies per epoch
        dataset_sizes (dict): Dataset sizes
        class_names (list): Class names
        best_acc (float): Best validation accuracy
        training_time (float): Training time in seconds

    Returns:
        str: Output directory path
    """
    output_dir = create_output_folder()

    print(f"\nGenerating visualizations...")
    print(f"Saving plots to: {output_dir}")

    # Generate all plots
    plot_training_history(train_losses, val_losses, train_accs, val_accs, output_dir)
    plot_final_metrics(train_losses, val_losses, train_accs, val_accs, output_dir)
    plot_data_distribution(dataset_sizes, class_names, output_dir)
    save_training_summary(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        best_acc,
        training_time,
        output_dir,
    )

    print("Visualizations saved successfully!")
    print(f"  - training_history.png")
    print(f"  - metrics_summary.png")
    print(f"  - data_distribution.png")
    print(f"  - training_summary.txt")

    return output_dir
