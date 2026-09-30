import torch
import torch.nn as nn
from . import config
from .data import train_loader,val_loader
from .model import CNN
from pathlib import Path
import pandas as pd

def train():
    torch.manual_seed(config.SEED)

    model = CNN(num_classes=config.NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    scheduler = None
    if config.USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.LR_SCHEDULER_MODE,
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=config.MIN_LR,
        )

    #=========================
    # Paths
    #=========================
    history = []
    #project directory
    project_root = Path(__file__).resolve().parents[1]

    #training metrics directory
    metrics_dir = project_root / "outputs" / "metrics" / config.EXPERIMENT_NAME
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_path = metrics_dir / "history.csv"

    #checkpoint directory
    checkpoint_dir = project_root / "checkpoints" / config.EXPERIMENT_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy_path = checkpoint_dir / "best_accuracy_model.pth"
    best_loss_path = checkpoint_dir / "best_loss_model.pth"

    
    #initialize best validation accuracy and loss
    best_val_accuracy = 0.0
    best_val_loss = float("inf")

    num_epochs = config.NUM_EPOCHS
    for epoch in range(num_epochs):

        current_lr = optimizer.param_groups[0]["lr"]
        
        # =========================
        # Training
        # =========================

        model.train()#put the model in training mode

        #initialize variables to track training loss and accuracy
        train_correct = 0
        train_total_images = 0
        train_running_loss = 0.0


        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_running_loss += loss.item() * images.size(0)# Multiply by batch size to get total loss for the batch

            predicted = torch.argmax(outputs, dim=1)# Get the index of the max log-probability

            train_correct += (predicted == labels).sum().item()
            train_total_images += labels.size(0)

        train_loss = train_running_loss / train_total_images
        train_accuracy = train_correct / train_total_images

        # =========================
        # Validation
        # =========================

        model.eval()#put the model in evaluation mode

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_running_loss += (
                    loss.item() * images.size(0)
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_accuracy = val_correct / val_total

        if scheduler is not None:
            scheduler.step(val_loss)

        # =========================
        # Save model checkpoint
        # =========================

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                model.state_dict(),
                best_accuracy_path
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                best_loss_path
            )

        # =========================
        # Save history
        # =========================
        
        history.append({
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        # Save training history to CSV
        df = pd.DataFrame(history)
        df.to_csv(history_path, index=False)

        # =========================
        # Print metrics
        # =========================

        print(
            f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

if __name__ == "__main__":
    train()