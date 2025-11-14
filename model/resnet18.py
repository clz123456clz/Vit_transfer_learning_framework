from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["Resnet18"]


def conv3x3(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    """3x3 conv with padding, no bias (BN follows)."""
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """ResNet BasicBlock: 3x3 conv -> BN -> ReLU -> 3x3 conv -> BN -> add skip -> ReLU."""
    expansion = 1

    def __init__(self, in_c: int, out_c: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_c, out_c, 1)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class Resnet18(nn.Module):
    """
    Custom ResNet-18 (CIFAR-stem, no pretrained).
    - Stem: 3x3, stride=1, no initial MaxPool  —— better for 64x64.
    - Stages: [2, 2, 2, 2] BasicBlocks with channel dims [64,128,256,512].
    - Head: AdaptiveAvgPool -> FC(num_classes).
    """
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        # --- Stem (CIFAR-style, keep more spatial detail for 64x64) ---
        self.in_c = 64
        self.conv1 = nn.Conv2d(3, self.in_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(self.in_c)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()

        # --- Residual stages ---
        self.layer1 = self._make_layer(64,  blocks=2, stride=1)  # 64x64
        self.layer2 = self._make_layer(128, blocks=2, stride=2)  # 32x32
        self.layer3 = self._make_layer(256, blocks=2, stride=2)  # 16x16
        self.layer4 = self._make_layer(512, blocks=2, stride=2)  # 8x8

        # --- Head ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self.init_weights()

    def _make_layer(self, out_c: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_c != out_c * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_c, out_c * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c * BasicBlock.expansion),
            )

        layers = [BasicBlock(self.in_c, out_c, stride=stride, downsample=downsample)]
        self.in_c = out_c * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def init_weights(self) -> None:
        """Kaiming init for conv, constant for BN, and fan-in for FC."""
        torch.manual_seed(42)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                fan_in = m.in_features
                std = (1.0 / fan_in) ** 0.5
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stages
        x = self.layer1(x)  # 64x64
        x = self.layer2(x)  # 32x32
        x = self.layer3(x)  # 16x16
        x = self.layer4(x)  # 8x8

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        return x