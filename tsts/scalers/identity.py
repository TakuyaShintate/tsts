from typing import Any, Dict, List

from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import SCALERS

from .scaler import Scaler

__all__ = ["IdentityScaler"]


@SCALERS.register()
class IdentityScaler(Scaler):
    def __init__(self, cfg: CN) -> None:
        super(IdentityScaler, self).__init__()
        self.cfg = cfg

    def fit(self, X_or_y: Tensor) -> None:
        pass

    def fit_batch(self, X_or_ys: List[Tensor]) -> None:
        pass

    @property
    def meta_info(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_cfg(cls, cfg: CN) -> "IdentityScaler":
        scaler = cls(cfg)
        return scaler

    def transform(self, X_or_y: Tensor) -> Tensor:
        return X_or_y

    def inv_transform(self, X_or_y: Tensor) -> Tensor:
        return X_or_y
