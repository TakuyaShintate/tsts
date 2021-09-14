from typing import Callable, List, Optional, Tuple

from torch import Tensor

__all__ = ["Batch", "MaybeRawDataset", "RawBatch", "RawDataset"]


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
