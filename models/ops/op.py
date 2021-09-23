import torch
import torch.nn as nn


class Op(nn.Module):
    def get_param_count(self) -> int:
        pass

    def get_flop_count(self) -> int:
        pass

    def forward(self, x) -> torch.Tensor:
        return self.conv(x)
