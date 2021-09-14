from typing import List

from torch import Tensor
from tsts.cfg import CfgNode as CN

__all__ = ["Scaler"]


class Scaler(object):
    @classmethod
    def from_cfg(cls, cfg: CN) -> "Scaler":
        raise NotImplementedError

    def fit(self, X_or_y: Tensor) -> None:
        raise NotImplementedError

    def fit_batch(self, X_or_ys: List[Tensor]) -> None:
        raise NotImplementedError

    def transform(self, X_or_y: Tensor) -> Tensor:
        raise NotImplementedError

    def inv_transform(self, X_or_y: Tensor) -> Tensor:
        raise NotImplementedError
