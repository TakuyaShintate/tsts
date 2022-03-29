from typing import Callable, List, Optional, Tuple

from torch import Tensor

__all__ = ["Batch", "MaybeRawDataset", "RawBatch", "RawDataset", "Frame"]


Batch = Tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Optional[Tensor],
    List[Callable],
    List[Callable],
]
MaybeRawDataset = List[Optional[Tensor]]
MaybeTensor = Optional[Tensor]
RawBatch = Tuple[
    Tuple[
        Tensor,
        Tensor,
        Tensor,
        Optional[Tensor],
        Callable,
        Callable,
    ]
]
RawDataset = List[Tensor]
# X, y, bias, time_stamps
# NOTE: y and time_stamps (& bias) are None during test
Frame = Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]
