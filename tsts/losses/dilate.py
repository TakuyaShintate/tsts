from typing import Any

import torch
from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import LOSSES
from tsts.thirdparty.dilate import PathDTWBatch, SoftDTWBatch, pairwise_distances

from .loss import Loss

__all__ = ["DILATE"]


@LOSSES.register()
class DILATE(Loss):
    """DILATE implementation.

    Example
    -------
    .. code-block:: yaml

        LOSSES:
          NAMES: ["DILATE"]
          ARGS: [{"alpha": 0.5, "gamma": 0.001}]

    Parameters
    ----------
    alpha: float, optional
        Balancing parameter of shape and temporal losses, by default 0.5

    gamma: float, optional
        Smoothing parameter of softmin, by default 0.001
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.001,
    ) -> None:
        super(DILATE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @classmethod
    def from_cfg(cls, cfg: CN, **loss_args: Any) -> "DILATE":
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
        (batch_size, horizon, num_out_feats) = Z.size()
        device = Z.device
        pair_dist_mat = torch.zeros((batch_size, horizon, horizon))
        pair_dist_mat = pair_dist_mat.to(device)
        for i in range(batch_size):
            Z_new = y_mask[i] * Z[i]
            y_new = y_mask[i] * y[i]
            pair_dist_mat[i] = pairwise_distances(Z_new, y_new)
        loss_v_shape = SoftDTWBatch.apply(pair_dist_mat, self.gamma)
        omega = pairwise_distances(torch.arange(1, horizon + 1).view(horizon, 1))
        omega = omega.to(device)
        loss_v_temp = torch.sum(PathDTWBatch.apply(pair_dist_mat, self.gamma) * omega)
        loss_v_temp /= y_mask.sum() / num_out_feats
        loss_v = self.alpha * loss_v_shape + (1.0 - self.alpha) * loss_v_temp
        return loss_v
