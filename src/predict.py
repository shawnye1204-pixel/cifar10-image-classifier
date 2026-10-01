"""Predict the CIFAR-10 class of one external image."""

import argparse
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torchvision import transforms

from . import config
from .device import get_device
from .model import CNN


# Keep the same class order as the CIFAR-10 dataset.
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def preprocess_image(image_path):
    # Match evaluation normalization without loading the training datasets.
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    with Image.open(Path(image_path).expanduser()) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = transform(image)

    # Convert [C, H, W] into a one-image batch [1, C, H, W].
    return tensor.unsqueeze(0)


def predict(image_path):
    """Return the predicted class and its softmax probability."""
    images = preprocess_image(image_path)
    if config.NUM_CLASSES != len(CLASSES):
        raise ValueError("Single-image prediction expects the 10 CIFAR-10 classes.")

    checkpoint_path = (
        Path(__file__).resolve().parents[1]
        / "checkpoints"
        / config.EXPERIMENT_NAME
        / "best_accuracy_model.pth"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Set EXPERIMENT_NAME to a trained experiment."
        )

    device = get_device()
    model = CNN(
        num_classes=config.NUM_CLASSES,
        use_batch_norm=config.USE_BATCH_NORM,
        use_extra_conv=config.USE_EXTRA_CONV,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint does not match the configured model. Check "
            "EXPERIMENT_NAME, USE_BATCH_NORM and USE_EXTRA_CONV."
        ) from error

    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(images.to(device))
        # Softmax is for reporting probabilities; training still uses logits.
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)

    return CLASSES[prediction.item()], confidence.item()


def main():
    parser = argparse.ArgumentParser(
        description="Classify one image using the configured CIFAR-10 model."
    )
    parser.add_argument("image", type=Path, help="Path to an image file")
    args = parser.parse_args()
    try:
        label, confidence = predict(args.image)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.1%}")


if __name__ == "__main__":
    main()
