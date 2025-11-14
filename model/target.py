"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Target CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.target import Target
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["Target"]


class Target(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        # TODO: 2(b) - define each layer
        self.conv1 = nn.Conv2d(3, 16, 5, 2, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 64, 5, 2, padding=2)
        self.conv3 = nn.Conv2d(64, 8, 5, 2, padding=2)
        self.fc_1 = nn.Linear(32, 2)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            # TODO: 2(b) - initialize the parameters for the convolutional layers
            input_channels = conv.in_channels
            kH, kW = conv.kernel_size
            fan_in = kH * kW * input_channels
            std = (1.0 / fan_in) ** 0.5
            conv.weight.data.normal_(mean=0.0, std=std)
            conv.bias.data.fill_(0.0)


        # TODO: 2(b) - initialize the parameters for [self.fc_1]
        input_size = self.fc_1.in_features
        fc_std = (1.0 / input_size) ** 0.5
        self.fc_1.weight.data.normal_(mean=0.0, std=fc_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.fc_1(x)

        return x