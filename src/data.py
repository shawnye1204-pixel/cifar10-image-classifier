from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch
from torch.utils.data import random_split
from . import config
from torch.utils.data import DataLoader
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Define the transformation to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5), 
        (0.5, 0.5, 0.5)
    )
])


# Load the CIFAR-10 dataset
full_train_dataset = CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)


# Split the full training dataset into training and validation datasets
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [45000, 5000],
    generator=torch.Generator().manual_seed(config.SEED)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


if __name__ == "__main__":

    print("classes:", train_dataset.dataset.classes)
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))

    for loader in [train_loader, val_loader, test_loader]:
        images, labels = next(iter(loader))
        print(images.shape, labels.shape)

    images, labels = next(iter(train_loader))
    print(images.shape)
    print(labels.shape)
    for image, label in zip(images, labels):
        print(image.shape, train_dataset.dataset.classes[label])