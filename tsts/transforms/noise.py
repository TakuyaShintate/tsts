from typing import Optional

from torch import Tensor
from torch.autograd import Variable
from tsts.core import TRANSFORMS
from tsts.types import Frame

from .transform import Transform


@TRANSFORMS.register()
class GaussianNoise(Transform):
    """Add gaussian noise to input."""

    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def apply(
        self,
        X: Tensor,
        y: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        time_stamps: Optional[Tensor] = None,
    ) -> Frame:
        data = X.data.new(X.size()).normal_(self.mean, self.std)
        noise = Variable(data)
        X = X + noise
        return (X, y, bias, time_stamps)
