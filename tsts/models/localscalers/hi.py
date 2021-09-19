import torch
from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import LOCALSCALERS
from tsts.models import Module

__all__ = ["HistricalInertia"]


@LOCALSCALERS.register()
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
        bias: Tensor,
    ) -> Tensor:
        Z = torch.zeros_like(bias, requires_grad=True)
        Z = Z[:, -self.horizon :] + bias[:, -self.horizon :]
        return Z[:, -self.horizon :]
