import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class ResNetConfig:
    channels: List[int]
    blocks: List[int]
    strides: List[int]

resnet_configs = {
    'resnet18': ResNetConfig([64,128,256,512], [2,2,2,2], [1,2,2,2]),
    'resnet34': ResNetConfig([64,128,256,512], [3,4,6,3], [1,2,2,2]),
}

def generate_random_config(channels, blocks, strides, op_registry):

    # configs is a list of lists of operations for each stage of the ResNet
    # ResNets always have four stages
    # each stage has a variable number of layers (for instance, in ResNet18 they all have 2 layers)
    configs = []

    in_channels = channels[0]

    for out_channels, block, stride in zip(channels, blocks, strides):

        # in a ResNet18 we have [2,2,2,2] blocks (i.e. 2 blocks per stage)
        # so in this case stage_config would be a list of 2 op configs
        stage_config = []

        for i, layer in enumerate(range(block)):
            op_config = {}

            conv = np.random.choice(op_registry)

            op_config["conv"] = conv
            op_config["stride"] = stride if i == 0 else 1

            op_config["in_channels"] = in_channels
            op_config["out_channels"] = out_channels

            # e.g. if group conv, generate random number of groups
            extra_args = conv.generate_random_args(in_channels, out_channels)

            op_config.update(extra_args)
            stage_config.append(op_config)

            in_channels = out_channels

        configs.append(stage_config)

    return configs
