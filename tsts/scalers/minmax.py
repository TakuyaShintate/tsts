from typing import Any, Dict

from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import SCALERS

from .scaler import Scaler

__all__ = ["MinMaxScaler"]


@SCALERS.register()
class MinMaxScaler(Scaler):
    def __init__(self, cfg: CN) -> None:
        super(MinMaxScaler, self).__init__()
        self.cfg = cfg

    def fit(self, X_or_y: Tensor) -> None:
        self.min_v = X_or_y.min(0)[0]
        self.max_v = X_or_y.max(0)[0]

    @property
    def meta_info(self) -> Dict[str, Any]:
        return {
            "min_v": self.min_v.tolist(),
            "max_v": self.max_v.tolist(),
        }

    @classmethod
    def from_cfg(cls, cfg: CN) -> "MinMaxScaler":
        scaler = cls(cfg)
        return scaler

    def transform(self, X_or_y: Tensor) -> Tensor:
        device = X_or_y.device
        self.min_v = self.min_v.to(device)
        self.max_v = self.max_v.to(device)
        X_or_y_new = (X_or_y - self.min_v) / (self.max_v - self.min_v)
        return X_or_y_new

    def inv_transform(self, X_or_y: Tensor) -> Tensor:
        device = X_or_y.device
        self.min_v = self.min_v.to(device)
        self.max_v = self.max_v.to(device)
        X_or_y_new = X_or_y * (self.max_v - self.min_v) + self.min_v
        return X_or_y_new
