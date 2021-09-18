from typing import Optional

import torch
from torch import Tensor
from torch.nn import Linear
from tsts.cfg import CfgNode as CN
from tsts.core import MODELS

from .module import Module

__all__ = ["HistricalInertia"]


@MODELS.register()
class HistricalInertia(Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
    ) -> None:
        assert lookback >= horizon
        super(HistricalInertia, self).__init__()
        self.lookback = lookback
        self.horizon = horizon
        self._init_dummy_module()

    def _init_dummy_module(self) -> None:
        self.dummy_module = Linear(1, 1)

    @classmethod
    def from_cfg(
        cls,
        num_in_feats: int,
        num_out_feats: int,
        cfg: CN,
    ) -> "HistricalInertia":
        lookback = cfg.IO.LOOKBACK
        horizon = cfg.IO.HORIZON
        model = cls(lookback, horizon)
        return model

    def forward(
        self,
        X: Tensor,
        X_mask: Tensor,
        time_stamps: Optional[Tensor] = None,
    ) -> Tensor:
        Z = torch.zeros_like(X, requires_grad=True)
        Z = Z[:, -self.horizon :] + X[:, -self.horizon :]
        return Z[:, -self.horizon :]
