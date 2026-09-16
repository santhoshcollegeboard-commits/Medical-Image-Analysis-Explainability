"""
Deep convolutional neural network for pulmonary radiograph analysis.
Includes residual blocks, batch normalization, dropout, and hooked feature maps for Grad-CAM explainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return F.relu(out)


class MedicalResNet(nn.Module):
    """
    Residual architecture for pulmonary medical imaging.
    Exposes final convolutional feature maps directly for Grad-CAM visual attribution.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super(MedicalResNet, self).__init__()

        self.prep = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = ResidualBlock(32, 64, stride=2)   # 28x28 -> 14x14
        self.layer2 = ResidualBlock(64, 128, stride=2)  # 14x14 -> 7x7
        self.layer3 = ResidualBlock(128, 256, stride=2) # 7x7 -> 4x4

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(256, num_classes)

        # Gradients and activation placeholders for Grad-CAM
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Retain feature activations for Grad-CAM
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        self.activations = x

        pooled = self.pool(x)
        flattened = torch.flatten(pooled, 1)
        dropped = self.dropout(flattened)
        out = self.fc(dropped)
        return out

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations
