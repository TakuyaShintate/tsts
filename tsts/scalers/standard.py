import copy
import typing
from typing import Any, Dict, List, Union

import torch
from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import SCALERS
from tsts.types import RawDataset

from .scaler import Scaler

__all__ = ["StandardScaler"]

EPSILON = 1e-8


@SCALERS.register()
class StandardScaler(Scaler):
    def __init__(
        self,
        mean: Union[Tensor, List[List[float]]],
        std: Union[Tensor, List[List[float]]],
        cfg: CN,
    ) -> None:
        super(StandardScaler, self).__init__()
        if isinstance(mean, Tensor) is False:
            mean = torch.tensor(mean)
        if isinstance(std, Tensor) is False:
            std = torch.tensor(std)
        self.mean = typing.cast(Tensor, mean)
        self.std = typing.cast(Tensor, std)
        self.cfg = cfg

    @property
    def meta_info(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_cfg(cls, X_or_y: RawDataset, cfg: CN) -> "StandardScaler":
        num_feats = X_or_y[0].size(-1)
        mean = torch.zeros(num_feats)
        std = torch.zeros(num_feats)
        num_instances = 0.0
        for i in range(len(X_or_y)):
            mean += X_or_y[i].sum(0)
            num_instances += float(X_or_y[i].size(0))
        mean /= num_instances
        for i in range(len(X_or_y)):
            std += ((X_or_y[i] - mean) ** 2).sum(0)
        std /= num_instances
        std = std.sqrt()
        scaler = cls(mean, std, cfg)
        return scaler

    def transform(self, X_or_y: RawDataset) -> RawDataset:
        num_instances = len(X_or_y)
        device = X_or_y[0].device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        X_or_y_new = copy.deepcopy(X_or_y)
        for i in range(num_instances):
            X_or_y_new[i] = (X_or_y[i] - self.mean) / (self.std + EPSILON)
        return X_or_y_new

    def inv_transform(self, X_or_y: RawDataset) -> RawDataset:
        num_instances = len(X_or_y)
        device = X_or_y[0].device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        X_or_y_new = copy.deepcopy(X_or_y)
        for i in range(num_instances):
            X_or_y_new[i] = X_or_y[i] * (self.std + EPSILON) + self.mean
        return X_or_y_new
