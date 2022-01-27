from typing import List, Optional

from torch import Tensor
from tsts.types import Frame

from .transform import Transform

__all__ = ["Pipeline"]


class Pipeline(object):
    def __init__(self, transforms: List[Transform]) -> None:
        self.transforms = transforms

    @property
    def num_transforms(self) -> int:
        return len(self.transforms)

    def apply(
        self,
        X: Tensor,
        y: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        time_stamps: Optional[Tensor] = None,
    ) -> Frame:
        """Apply a series of transforms to given input.

        Parameters
        ----------
        X : Tensor
            Input time series

        y : Optional[Tensor]
            Ground truth for input time series

        time_stamps : Optional[Tensor]
            Timestamps for each step of input time series
        """
        for t in self.transforms:
            (X, y, bias, time_stamps) = t.apply(X, y, bias, time_stamps)
        return (X, y, bias, time_stamps)
