'''
input: B: batch size, C: channels, H: height, W: width)
output: (B, num_classes)
'''

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
            self, 
            num_classes=10, 
            use_batch_norm=False,
            use_extra_conv=False,
        ):
        super().__init__()
        self.use_extra_conv = use_extra_conv

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        # Use an identity operation to preserve the original CNN when disabled.
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()

        # Preserve the feature-map dimensions and the existing classifier.
        if self.use_extra_conv:
            self.conv3 = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            )
            self.bn3 = (
                nn.BatchNorm2d(64)
                if use_batch_norm
                else nn.Identity()
            )

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        #Apply the extra convolutional layer if enabled
        if self.use_extra_conv:
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)

        x = self.pool(x)
        
        x = torch.flatten(x, start_dim=1)  # Flatten all dimensions except batch
        
        x = self.fc1(x)
        
        x = self.relu(x)
        x = self.fc2(x)
        

        return x


if __name__ == "__main__":
    model = CNN()

    x = torch.randn(8, 3, 32, 32)

    output = model(x)
    predictions = torch.argmax(output, dim=1)

    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    print("Output logits:", output)
    for i, pred in enumerate(predictions):
        print(f"Sample {i}: Predicted class: {classes[pred]}")