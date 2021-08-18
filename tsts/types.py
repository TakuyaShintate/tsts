from typing import List, Tuple, Union

from torch import Tensor

__all__ = ["Batch", "RawBatch", "RawDataset"]

# (X, y, X_mask, y_mask)
Batch = Tuple[Tensor, Tensor, Tensor, Tensor]
RawBatch = Tuple[Tuple[Tensor, Tensor]]
RawDataset = Union[Tensor, List[Tensor], Tuple[Tensor]]
