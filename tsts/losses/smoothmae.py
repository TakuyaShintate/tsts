from typing import Any

import torch.nn.functional as F
from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import LOSSES

from .loss import Loss

__all__ = ["SmoothMAE"]


@LOSSES.register()
class SmoothMAE(Loss):
    """Smooth MAE (smooth l1 loss) implementation.

    Example
    -------
    .. code-block:: yaml

        LOSSES:
          NAMES: ["SmoothMAE"]
          ARGS: [{"beta": 1.0 / 9.0}]
    """

    def __init__(self, beta: float = 1.0) -> None:
        super(SmoothMAE, self).__init__()
        self.beta = beta

    @classmethod
    def from_cfg(cls, cfg: CN, **loss_args: Any) -> "SmoothMAE":
        loss = cls(**loss_args)
        return loss

    def forward(self, Z: Tensor, y: Tensor, y_mask: Tensor) -> Tensor:
        """Return loss value.

        Parameters
        ----------
        Z : Tensor
            Prediction

        y : Tensor
            Target time series

        y_mask : Tensor
            Target time series mask

        Returns
        -------
        Tensor
            Loss value
        """
        loss_val = F.smooth_l1_loss(Z, y, reduction="none", beta=self.beta)
        loss_val *= y_mask
        loss_val = loss_val.sum()
        loss_val /= y_mask.sum()
        return loss_val
