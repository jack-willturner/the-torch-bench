import math
import torch
import torch.nn as nn
import numpy as np

from .op import Op
from typing import Union, Tuple, Dict


class Seq1(Op):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding=1, args=None
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
                    padding=padding,
                    groups=g,
                )
            )
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        return torch.cat(outs, dim=1)

    def generate_random_args(
        in_channels, out_channels, img_shape=None
    ) -> Dict[str, int]:

        """
        Step 1: choose a split factor
        """
        valid_split_factor = False

        while not valid_split_factor:
            split_factor = np.random.randint(1, min(in_channels, out_channels))
            valid_split_factor = (in_channels % split_factor == 0) and (
                out_channels % split_factor == 0
            )

        """
        Step 2: associate grouping to each split
        """
        # e.g. if split factor is 2 and out_channels is 64 then there are two groups of 32
        channels_per_split = out_channels // split_factor
        groups_per_split = []

        # now generate a grouping level for each split§
        for split in range(split_factor):
            valid_group_factor = False

            while not valid_group_factor:
                groups = np.random.randint(1, channels_per_split)

                valid_group_factor = channels_per_split % groups == 0

            groups_per_split.append(groups)

        return {"split_factor": split_factor, "groups": groups_per_split}


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
