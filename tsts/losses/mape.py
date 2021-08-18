from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import LOSSES

from .loss import Loss

__all__ = ["MAPE"]


@LOSSES.register()
class MAPE(Loss):
    @classmethod
    def from_cfg(cls, cfg: CN) -> "MAPE":
        loss = cls()
        return loss

    def forward(self, Z: Tensor, y: Tensor, y_mask: Tensor) -> Tensor:
        Z = Z[y_mask.bool()]
        y = y[y_mask.bool()]
        # Remove invalid elements
        Z = Z[y != 0.0]
        y = y[y != 0.0]
        loss_val = ((Z - y) / y).abs()
        loss_val = loss_val.sum()
        loss_val /= y_mask.sum()
        return loss_val