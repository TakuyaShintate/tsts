import torch.nn.functional as F
from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import LOSSES

from .loss import Loss

__all__ = ["MSE"]


@LOSSES.register()
class MSE(Loss):
    @classmethod
    def from_cfg(cls, cfg: CN) -> "MSE":
        loss = cls()
        return loss

    def forward(self, Z: Tensor, y: Tensor) -> Tensor:
        loss_val = F.mse_loss(Z, y)
        return loss_val
