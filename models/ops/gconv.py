import math
import torch.nn as nn
import numpy as np
from .op import Op
from typing import Union, Tuple, Dict


class GConv(Op):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        stride: Union[Tuple[int, int], int],
        padding: int,
        args: Dict[str, int],
    ) -> None:
        super(GConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            groups=args['groups'],
            stride=stride,
            padding=padding,
        )

    def get_param_count(self) -> int:
        return math.prod(self.conv.weight.size())

    def get_flop_count(self) -> int:
        return super().get_flop_count()

    def generate_random_args(in_channels, out_channels, img_shape=None) -> Dict[str, int]:

        valid_number_of_groups = False

        while(not valid_number_of_groups):
            groups = np.random.randint(1, min(in_channels, out_channels))
            valid_number_of_groups = (in_channels % groups == 0) and (out_channels % groups == 0)

        return {'groups': groups}
