import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple

from dataclasses import dataclass

__all__ = ["ResNet18", "ResNet34"]

class CIFARStem(nn.Module):
    def __init__(self, in_planes=3, planes=64):
        super(CIFARStem, self).__init__()

        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ImageNetStem(nn.Module):
    def __init__(self, in_planes=3, planes=64):
        super(ImageNetStem, self).__init__()

        self.conv = nn.Conv2d(in_planes, planes, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(planes)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, layer_config):
        super(BasicBlock, self).__init__()

        in_channels = layer_config["in_channels"]
        out_channels = layer_config["out_channels"]
        conv = layer_config["conv"]
        stride = layer_config["stride"]

        self.conv1 = conv(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            args=layer_config,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            args=layer_config,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, configs=None, num_classes=10, stem=CIFARStem):
        super(ResNet, self).__init__()

        ###### cache details to make model loading easy
        self.model_class = "resnet"
        self.block = block
        self.num_blocks = num_blocks
        self.configs = configs
        self.num_classes = num_classes
        self.stem_type = stem
        ##############################################

        self.in_planes = 64

        self.stem = stem(3, self.in_planes)

        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], configs[0]
        )
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], configs[1]
        )
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], configs[2]
        )
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], configs[3]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, configs):
        # strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for layer_config in configs:
            layers.append(block(layer_config))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
