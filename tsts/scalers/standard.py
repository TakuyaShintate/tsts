from typing import Any, Dict, List

import torch
from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import SCALERS

from .scaler import Scaler

__all__ = ["StandardScaler"]

EPSILON = 1e-8


@SCALERS.register()
class StandardScaler(Scaler):
    def __init__(self, cfg: CN) -> None:
        super(StandardScaler, self).__init__()
        self.cfg = cfg

    @property
    def meta_info(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    def fit(self, X_or_y: Tensor) -> None:
        self.mean = X_or_y.mean(0)
        self.std = X_or_y.std(0)

    def fit_batch(self, X_or_ys: List[Tensor]) -> None:
        num_feats = X_or_ys[0].size(-1)
        mean = torch.zeros(num_feats)
        std = torch.zeros(num_feats)
        num_instances = 0.0
        for i in range(len(X_or_ys)):
            mean += X_or_ys[i].sum(0)
            num_instances += float(X_or_ys[i].size(0))
        self.mean = mean / num_instances
        for i in range(len(X_or_ys)):
            std += ((X_or_ys[i] - mean) ** 2).sum(0)
        std /= num_instances
        self.std = std.sqrt()

    @classmethod
    def from_cfg(cls, cfg: CN) -> "StandardScaler":
        scaler = cls(cfg)
        return scaler

    def transform(self, X_or_y: Tensor) -> Tensor:
        device = X_or_y.device
        X_or_y_new = (X_or_y - self.mean.to(device)) / self.std.to(device)
        return X_or_y_new

    def inv_transform(self, X_or_y: Tensor) -> Tensor:
        device = X_or_y.device
        X_or_y_new = X_or_y * self.std.to(device) + self.mean.to(device)
        return X_or_y_new
