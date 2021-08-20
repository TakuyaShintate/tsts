import copy
from typing import Any, Dict

from tsts.cfg import CfgNode as CN
from tsts.core import SCALERS
from tsts.types import RawDataset

from .scaler import Scaler

__all__ = ["MinMaxScaler"]


@SCALERS.register()
class MinMaxScaler(Scaler):
    def __init__(
        self,
        min_v: float,
        max_v: float,
        cfg: CN,
    ) -> None:
        super(MinMaxScaler, self).__init__()
        self.min_v = min_v
        self.max_v = max_v
        self.cfg = cfg

    @property
    def meta_info(self) -> Dict[str, Any]:
        return {
            "min_v": self.min_v,
            "max_v": self.max_v,
        }

    @classmethod
    def from_cfg(cls, X_or_y: RawDataset, cfg: CN) -> "MinMaxScaler":
        num_instances = len(X_or_y)
        min_v = float("inf")
        max_v = -float("inf")
        for i in range(num_instances):
            current_min_v = X_or_y[i].min().item()
            current_max_v = X_or_y[i].max().item()
            if current_min_v < min_v:
                min_v = current_min_v
            if current_max_v > max_v:
                max_v = current_max_v
        scaler = cls(min_v, max_v, cfg)
        return scaler

    def transform(self, X_or_y: RawDataset) -> RawDataset:
        num_instances = len(X_or_y)
        X_or_y_new = copy.deepcopy(X_or_y)
        for i in range(num_instances):
            X_or_y_new[i] = (X_or_y[i] - self.min_v) / (self.max_v - self.min_v)
        return X_or_y_new

    def inv_transform(self, X_or_y: RawDataset) -> RawDataset:
        num_instances = len(X_or_y)
        X_or_y_new = copy.deepcopy(X_or_y)
        for i in range(num_instances):
            X_or_y_new[i] = X_or_y[i] * (self.max_v - self.min_v) + self.min_v
        return X_or_y_new
