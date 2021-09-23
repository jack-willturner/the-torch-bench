import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple


class Seq4(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding=1, args=None
    ):
        super(Seq4, self).__init__()

        kernel_sizes = args["kernel_sizes"]
        groups = args["groups"]

        assert len(kernel_sizes) == groups

        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups

        self.convs = []

        for kernel_size in kernel_sizes:
            self.convs.append(
                nn.Conv2d(
                    self.in_channels_per_group,
                    self.out_channels_per_group,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                )
            )

    def forward(self, x):

        outs = []
        for i, conv in enumerate(self.convs):
            outs.append(
                conv(
                    x[
                        :,
                        (i * self.in_channels_per_group) : (i + 1)
                        * self.in_channels_per_group,
                        :,
                        :,
                    ]
                )
            )
        return torch.cat(outs, 1)

    def generate_random_args(
        in_channels: int, out_channels: int, img_shape:Tuple[int, int]=None
    ) -> Dict:

        # choose n_groups to be something divisble by both in and out_channels
        def find_common_factors(num1: int, num2: int) -> List[int]:
            common_factors = []
            for i in range(1, min(num1, num2) + 1):
                if num1 % i == num2 % i == 0:
                    common_factors.append(i)
            return common_factors

        groups: int = np.random.choice(find_common_factors(in_channels, out_channels))

        kernel_sizes: List[int] = []

        for _ in range(groups):
            kernel_sizes.append(np.random.choice([1, 3, 5, 7, 11]))

        return {"groups": groups, "kernel_sizes": kernel_sizes}
