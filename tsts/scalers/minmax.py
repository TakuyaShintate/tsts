from typing import List

import torch
from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import SCALERS

from .scaler import Scaler

__all__ = ["MinMaxScaler"]


@SCALERS.register()
class MinMaxScaler(Scaler):
    def __init__(self) -> None:
        super(MinMaxScaler, self).__init__()

    @classmethod
    def from_cfg(cls, cfg: CN) -> "MinMaxScaler":
        scaler = cls()
        return scaler

    def fit(self, X_or_y: Tensor) -> None:
        self.min_v = X_or_y.min(0)[0]
        self.max_v = X_or_y.max(0)[0]

    def fit_batch(self, X_or_ys: List[Tensor]) -> None:
        num_feats = X_or_ys[0].size(-1)
        min_v = torch.zeros(num_feats) + float("inf")
        max_v = torch.zeros(num_feats) - float("inf")
        for i in range(len(X_or_ys)):
            min_v = torch.minimum(min_v, X_or_ys[i].min(0)[0])
            max_v = torch.maximum(max_v, X_or_ys[i].max(0)[0])
        self.min_v = min_v
        self.max_v = max_v

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
