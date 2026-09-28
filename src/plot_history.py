from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from . import config


def plot_history():

    # =========================
    # Paths
    # =========================

    project_root = Path(__file__).resolve().parents[1]

    history_path = (
        project_root
        / "outputs"
        / "metrics"
        / config.EXPERIMENT_NAME
        / "history.csv"
    )

    figures_dir = (
        project_root
        / "outputs"
        / "figures"
        / config.EXPERIMENT_NAME
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    loss_path = figures_dir / "loss_curve.png"
    accuracy_path = figures_dir / "accuracy_curve.png"


    # =========================
    # Load training history
    # =========================

    history = pd.read_csv(history_path)

    epochs = history["epoch"]

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    train_accuracy = history["train_accuracy"]
    val_accuracy = history["val_accuracy"]


    # =========================
    # Plot loss
    # =========================

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_loss,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid()

    plt.savefig(
        loss_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    # =========================
    # Plot accuracy
    # =========================

    plt.figure(figsize=(8, 5))#weight: 8 inches , height: 5 inches

    plt.plot(
        epochs,
        train_accuracy,
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracy,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")

    plt.legend()#label the lines
    plt.grid()

    plt.savefig(
        accuracy_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {loss_path}")
    print(f"Saved: {accuracy_path}")


if __name__ == "__main__":
    plot_history()