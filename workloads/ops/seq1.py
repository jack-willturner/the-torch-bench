import math
import torch.nn as nn

from .op import Op
from typing import Union, Tuple, Dict

class Seq1(ConvModule):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, bias, padding=1, args=None
    ):
        super(Seq1, self).__init__()
        convs = []
        sf = args["split_factor"]
        for i, layer in enumerate(range(sf)):
            g = args["groups"][i]
            convs.append(
                nn.Conv2d(
                    in_channels,
                    out_channels // sf,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias,
                    padding=padding,
                    groups=g,
                )
            )
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        return torch.cat(outs, dim=1)

"""

            if conv == Seq1:
                sf = np.random.choice([1, 2, 4, 8])
                op_config["split_factor"] = sf
                op_config["groups"] = np.random.choice([1, 2, 4, 8], sf)

            elif conv == Seq2:
                op_config["unroll_factor"] = np.random.choice([1, 2, 4, 8, 16])
                op_config["unrollconv_groups"] = np.random.choice([1, 2, 3, 4])

            elif conv == Seq3:
                sf = np.random.choice([1, 2, 4, 8])
                op_config["split_factor"] = sf
"""
