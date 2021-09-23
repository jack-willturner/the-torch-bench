import math
import torch
import torch.nn as nn
import numpy as np

from .op import Op
from typing import Union, Tuple, Dict


class Seq2(Op):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding=1, args=None
    ):
        super(Seq2, self).__init__()
        self.unroll_factor = args["unroll_factor"]
        g = args["unrollconv_groups"]
        self.conv1 = nn.Conv2d(
            in_channels,
            self.unroll_factor,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.convg1 = nn.Conv2d(
            (in_channels - self.unroll_factor),
            (out_channels - self.unroll_factor),
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=g,
        )

    def forward(self, x):
        l_slice = x
        r_slice = x[:, self.unroll_factor :, :, :]

        l_out = self.conv1(l_slice)
        r_out = self.convg1(r_slice)

        return torch.cat((l_out, r_out), 1)

    def generate_random_args(in_channels, out_channels, img_shape=None):

        # step 1: choose an unroll factor
        unroll_factor = np.random.randint(0, min(in_channels, out_channels))

        remaining_in_channels = in_channels - unroll_factor
        remaining_out_channels = out_channels - unroll_factor

        if remaining_in_channels > 1 and remaining_out_channels > 1:

            # step 2: choose a grouping factor for the rest
            valid_number_of_groups = False
            while(not valid_number_of_groups):

                groups = np.random.randint(1, min(remaining_in_channels, remaining_out_channels))
                valid_number_of_groups = (remaining_in_channels % groups == 0) and (remaining_out_channels % groups == 0)

        else:
            groups = 1

        return {"unroll_factor": unroll_factor, "unrollconv_groups": groups}
