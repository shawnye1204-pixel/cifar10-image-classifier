import random
import torch
import torch.nn as nn
from . import config
from .model import CNN
from .device import get_device
from pathlib import Path
import pandas as pd

#save the model weights to disk atomically, ensuring that the previous checkpoint remains intact until the new file is fully written.
def atomic_save(value, path):
    """Keep the previous checkpoint intact until the new file is complete."""
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(value, temporary_path)
    temporary_path.replace(path)

#capture the model weights on CPU, including BatchNorm buffers, to avoid device-specific issues.
def cpu_weights(model):
    """Take an independent snapshot, including BatchNorm buffers."""
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}

# Capture the RNG state for Python, CPU, and the current device.
def capture_rng(device):
    state = {"python": random.getstate(), "cpu": torch.get_rng_state(),
             "device_type": device.type, "device": None}
    if device.type == "cuda":
        state["device"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["device"] = torch.mps.get_rng_state()
    return state

# Restore the RNG state, with a warning if the device type has changed.
def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["cpu"])
    if state["device_type"] != device.type:
        print("Device changed: training can resume, but random sequences may differ.")
        return
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["device"], device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state["device"])

#write the training history, best accuracy model, and best loss model to disk
def write_outputs(history, best_accuracy_state, best_loss_state,
                  history_path, best_accuracy_path, best_loss_path):
    # These files can be rebuilt from the authoritative last checkpoint.
    atomic_save(best_accuracy_state, best_accuracy_path)
    atomic_save(best_loss_state, best_loss_path)
    temporary_path = history_path.with_name(history_path.name + ".tmp")
    pd.DataFrame(history).to_csv(temporary_path, index=False)
    temporary_path.replace(history_path)


def train():
    # Load datasets only in the main training process.
    from .data import train_loader, val_loader

    random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    device = get_device()
    non_blocking = device.type == "cuda"
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # Move the model before creating the optimizer.
    model = CNN(
        num_classes=config.NUM_CLASSES,
        use_batch_norm=config.USE_BATCH_NORM,
        use_extra_conv=config.USE_EXTRA_CONV,
        ).to(device)
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
    last_path = checkpoint_dir / "last_checkpoint.pth"

    # Total epochs may increase on resume; training settings must stay fixed.
    config_keys = (
        "BATCH_SIZE", "LEARNING_RATE", "NUM_CLASSES", "SPLIT_SEED", "SEED",
        "USE_AUGMENTATION", "USE_BATCH_NORM", "USE_EXTRA_CONV",
        "USE_LR_SCHEDULER", "LR_SCHEDULER_MODE", "LR_SCHEDULER_FACTOR",
        "LR_SCHEDULER_PATIENCE", "MIN_LR",
    )
    run_config = {key: getattr(config, key) for key in config_keys}

    
    #initialize best validation accuracy and loss
    best_val_accuracy = 0.0
    best_val_loss = float("inf")

    start_epoch = 0
    best_accuracy_state = None
    best_loss_state = None

    if config.RESUME_TRAINING:
        if not last_path.is_file():
            raise FileNotFoundError(f"No full training checkpoint: {last_path}")
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        if checkpoint["run_config"] != run_config:
            raise ValueError("Training settings differ from the saved checkpoint.")
        start_epoch = checkpoint["completed_epochs"]
        if config.NUM_EPOCHS < start_epoch:
            raise ValueError("NUM_EPOCHS is below the number of completed epochs.")

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = checkpoint["history"]
        best_val_accuracy = checkpoint["best_val_accuracy"]
        best_val_loss = checkpoint["best_val_loss"]
        best_accuracy_state = checkpoint["best_accuracy_state"]
        best_loss_state = checkpoint["best_loss_state"]

        # Repair exported files if an interruption occurred after checkpointing.
        write_outputs(history, best_accuracy_state, best_loss_state,
                      history_path, best_accuracy_path, best_loss_path)
        # Restore RNG after initialization and loading, before iterating loaders.
        restore_rng(checkpoint["rng_state"], device)
        del checkpoint
        print(f"Resumed after epoch {start_epoch}; total target: {config.NUM_EPOCHS}")
    elif any(path.exists() for path in
             (last_path, best_accuracy_path, best_loss_path, history_path)):
        raise FileExistsError("Results already exist. Resume or use a new experiment name.")

    num_epochs = config.NUM_EPOCHS
    for epoch in range(start_epoch, num_epochs):

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
            # Keep each batch on the same device as the model.
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

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
                images = images.to(device, non_blocking=non_blocking)
                labels = labels.to(device, non_blocking=non_blocking)

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

        if best_accuracy_state is None or val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_accuracy_state = cpu_weights(model)

        if best_loss_state is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_loss_state = cpu_weights(model)

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

        # Commit one complete epoch, after scheduler.step and history updates.
        checkpoint = {
            "completed_epochs": epoch + 1,
            "model_state": cpu_weights(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "best_accuracy_state": best_accuracy_state,
            "best_loss_state": best_loss_state,
            "history": history,
            "run_config": run_config,
            "rng_state": capture_rng(device),
        }
        atomic_save(checkpoint, last_path)
        del checkpoint
        write_outputs(history, best_accuracy_state, best_loss_state,
                      history_path, best_accuracy_path, best_loss_path)

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
    try:
        train()
    except KeyboardInterrupt:
        print("\nStopped. The latest complete epoch is in last_checkpoint.pth, "
              "if at least one epoch was saved. "
              "Set RESUME_TRAINING=True to continue.")