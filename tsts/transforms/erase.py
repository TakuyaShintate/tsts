import random
from typing import Optional

from torch import Tensor
from tsts.core import TRANSFORMS
from tsts.types import Frame

from .transform import Transform


@TRANSFORMS.register()
class RandomErase(Transform):
    def __init__(
        self,
        min_size: int = 0,
        max_size: int = 5,
        p: float = 0.7,
    ) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self.p = p

    def apply(
        self,
        X: Tensor,
        y: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        time_stamps: Optional[Tensor] = None,
    ) -> Frame:
        if random.random() < self.p:
            size = random.randint(self.min_size, self.max_size)
            X_steps = len(X)
            X_end = random.randint(0, X_steps)
            X[max(0, X_end - size) : X_end] = X.mean(0)
        return (X, y, bias, time_stamps)
