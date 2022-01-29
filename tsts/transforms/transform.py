import warnings
from typing import Optional

from torch import Tensor
from tsts.types import Frame

__all__ = ["Transform"]


class Transform(object):
    """Base transform class."""

    def apply(
        self,
        X: Tensor,
        y: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        time_stamps: Optional[Tensor] = None,
    ) -> Frame:
        warnings.warn("Base transform is called. Be sure your pipeline is correct!")
        return (X, y, bias, time_stamps)
