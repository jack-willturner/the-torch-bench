import math
import torch
import torch.nn as nn
import numpy as np

from .op import Op
from typing import Union, Tuple, Dict


class Seq3(Op):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding=1, args=None
    ):
        super(Seq3, self).__init__()
        self.split_factor = args["split_factor"]
        self.inferred_image_shape = args["inferred_image_shape"]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
                for i in range(args["split_factor"])
            ]
        )

    def forward(self, x):

        assert x.size()[2] == self.inferred_image_shape[0]

        H = x.shape[2]
        Hg = H // self.split_factor

        outs = []
        for i, conv in enumerate(self.convs):
            x_ = x[:, :, i * Hg : (i + 1) * Hg, :]
            outs.append(conv(x_))

        return torch.cat(outs, 2)

    def generate_random_args(in_channels, out_channels, img_shape=None):

        assert img_shape is not None
        split_factor = img_shape[0] + 1

        while img_shape[0] % split_factor != 0:
            split_factor = np.random.randint(1, min(img_shape[0], img_shape[1]))

        return {"split_factor": split_factor, "inferred_image_shape": img_shape}
