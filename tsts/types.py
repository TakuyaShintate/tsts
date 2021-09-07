from typing import List, Optional, Tuple

from torch import Tensor

__all__ = ["Batch", "MaybeRawDataset", "RawBatch", "RawDataset"]


Batch = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]
MaybeRawDataset = List[Optional[Tensor]]
RawBatch = Tuple[Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]]
RawDataset = List[Tensor]
