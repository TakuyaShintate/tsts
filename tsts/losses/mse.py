from typing import Any

import torch.nn.functional as F
from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import LOSSES

from .loss import Loss

__all__ = ["MSE"]


@LOSSES.register()
class MSE(Loss):
    """MSE implementation.

    Example
    -------
    .. code-block:: yaml

        LOSSES:
          NAMES: ["MSE"]
    """

    @classmethod
    def from_cfg(cls, cfg: CN, **loss_args: Any) -> "MSE":
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
        loss_val = F.mse_loss(Z, y, reduction="none")
        loss_val *= y_mask
        loss_val = loss_val.sum()
        loss_val /= y_mask.sum()
        return loss_val
