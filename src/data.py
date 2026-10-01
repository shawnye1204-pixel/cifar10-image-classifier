from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch
from torch.utils.data import Subset
from . import config
from .device import get_device
from torch.utils.data import DataLoader
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Fixed preprocessing for validation and test images
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
    ),
])

# Optional augmentation for training images
if config.USE_AUGMENTATION:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ])
else:
    train_transform = eval_transform

# Load the CIFAR-10 dataset
# Same training images, but separate transforms
full_train_dataset = CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=train_transform,
)

full_val_dataset = CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=eval_transform,
)

test_dataset = CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=eval_transform,
)

# Keep the same split across experiments
generator = torch.Generator().manual_seed(config.SPLIT_SEED)
indices = torch.randperm(
    len(full_train_dataset),
    generator=generator,
).tolist()

train_indices = indices[:45000]
val_indices = indices[45000:]

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_val_dataset, val_indices)


# Keep CPU and MPS data loading unchanged.
use_cuda = get_device().type == "cuda"
train_workers = config.CUDA_NUM_WORKERS if use_cuda else 0

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=train_workers,
    pin_memory=use_cuda,
    persistent_workers=train_workers > 0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=use_cuda,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=use_cuda,
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