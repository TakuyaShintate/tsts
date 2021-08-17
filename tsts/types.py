from typing import List, Tuple, Union

from torch import Tensor

__all__ = ["RawDataset"]

RawDataset = Union[Tensor, List[Tensor], Tuple[Tensor]]
