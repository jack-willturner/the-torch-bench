from .conv import Conv
from .gconv import GConv
from .seq1 import Seq1
from .seq2 import Seq2
from .seq4 import Seq4

__all__ = ["Conv", "GConv", "Seq1", "Seq2", "Seq4"]

OP_REGISTRY = [Conv, GConv, Seq1, Seq2, Seq4]
