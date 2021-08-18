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

    def forward(self, Z: Tensor, y: Tensor, y_mask: Tensor) -> Tensor:
        loss_val = F.mse_loss(Z, y, reduction="none")
        loss_val *= y_mask
        loss_val = loss_val.sum()
        loss_val /= y_mask.sum()
        return loss_val