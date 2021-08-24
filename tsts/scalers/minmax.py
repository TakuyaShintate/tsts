import copy
import typing
from typing import Any, Dict, List, Union

import torch
from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import SCALERS
from tsts.types import RawDataset

from .scaler import Scaler

__all__ = ["MinMaxScaler"]


@SCALERS.register()
class MinMaxScaler(Scaler):
    def __init__(
        self,
        min_v: Union[Tensor, List[List[float]]],
        max_v: Union[Tensor, List[List[float]]],
        cfg: CN,
    ) -> None:
        super(MinMaxScaler, self).__init__()
        if isinstance(min_v, Tensor) is False:
            min_v = torch.tensor(min_v)
        if isinstance(max_v, Tensor) is False:
            max_v = torch.tensor(max_v)
        self.min_v = typing.cast(Tensor, min_v)
        self.max_v = typing.cast(Tensor, max_v)
        self.cfg = cfg

    @property
    def meta_info(self) -> Dict[str, Any]:
        return {
            "min_v": self.min_v.tolist(),
            "max_v": self.max_v.tolist(),
        }

    @classmethod
    def from_cfg(cls, X_or_y: RawDataset, cfg: CN) -> "MinMaxScaler":
        X_or_y_new = torch.cat(X_or_y)
        min_v = X_or_y_new.min(0)[0]
        max_v = X_or_y_new.max(0)[0]
        scaler = cls(min_v, max_v, cfg)
        return scaler

    def transform(self, X_or_y: RawDataset) -> RawDataset:
        num_instances = len(X_or_y)
        device = X_or_y[0].device
        self.min_v = self.min_v.to(device)
        self.max_v = self.max_v.to(device)
        X_or_y_new = copy.deepcopy(X_or_y)
        for i in range(num_instances):
            X_or_y_new[i] = (X_or_y[i] - self.min_v) / (self.max_v - self.min_v)
        return X_or_y_new

    def inv_transform(self, X_or_y: RawDataset) -> RawDataset:
        num_instances = len(X_or_y)
        device = X_or_y[0].device
        self.min_v = self.min_v.to(device)
        self.max_v = self.max_v.to(device)
        X_or_y_new = copy.deepcopy(X_or_y)
        for i in range(num_instances):
            X_or_y_new[i] = X_or_y[i] * (self.max_v - self.min_v) + self.min_v
        return X_or_y_new
