from typing import List, Tuple, Union

from torch import Tensor

__all__ = ["Batch", "RawBatch", "RawDataset"]


Batch = Tuple[Tensor, Tensor, Tensor, Tensor]
RawBatch = Tuple[Tuple[Tensor, Tensor]]
RawDataset = Union[Tensor, List[Tensor]]
