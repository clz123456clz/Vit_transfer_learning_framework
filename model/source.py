"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Source CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.source import Source
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_random_seed


__all__ = ["Source"]


class Source(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        # TODO: 3(a) - define each layer
        self.conv1 = nn.Conv2d(3, 16, 5, 2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 64, 5, 2, 2)
        self.conv3 = nn.Conv2d(64, 8, 5, 2, 2)
        self.fc1 = nn.Linear(32, 8)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        set_random_seed()

        for conv in [self.conv1, self.conv2, self.conv3]:
            # TODO: 3(a) - initialize the parameters for the convolutional layers
            input_channels = conv.in_channels
            kH, kW = conv.kernel_size
            fan_in = kH * kW * input_channels
            std = (1.0 / fan_in) ** 0.5
            conv.weight.data.normal_(mean=0.0, std=std)
            conv.bias.data.fill_(0.0)
        
        # TODO: 3(a) - initialize the parameters for [self.fc1]
        input_size = self.fc1.in_features
        fc_std = (1.0 / input_size) ** 0.5
        self.fc1.weight.data.normal_(mean=0.0, std=fc_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward propagation for a batch of input examples. Pass the input array
        through layers of the model and return the output after the final layer.

        Args:
            x: array of shape (N, C, H, W) 
                N = number of samples
                C = number of channels
                H = height
                W = width

        Returns:
            z: array of shape (1, # output classes)
        """
        N, C, H, W = x.shape

        # TODO: 3(a) - forward pass
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x
