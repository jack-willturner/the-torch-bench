import torch.nn as nn

from .op import Op
from typing import Dict, Union, Tuple
import numpy as np


class Conv(Op):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        stride: Union[Tuple[int, int], int],
        padding: int,
        args: Dict,
    ) -> None:
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )

    def get_param_count(self) -> int:
        return math.prod(self.conv.weight.size())

    def get_flop_count(self) -> int:
        return super().get_flop_count()

    def generate_random_args(in_channels, out_channels, img_shape=None):
        return {}
