import torch.nn.functional as F
from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import METRICS

from .metric import Metric

__all__ = ["MSE"]


@METRICS.register()
class MSE(Metric):
    """MSE implementation.

    Example
    -------
    .. code-block:: yaml

        METRICS:
          NAMES: ["MSE"]
    """

    def __init__(self) -> None:
        super(MSE, self).__init__()

    @classmethod
    def from_cfg(cls, cfg: CN) -> "MSE":
        metric = cls()
        return metric

    def update(self, Z: Tensor, y: Tensor, y_mask: Tensor) -> None:
        mse = F.mse_loss(Z, y, reduction="none")
        mse *= y_mask
        score = mse.sum().item()
        self.total_score += score
        instances = y_mask.sum().item()
        self.total_instances += instances

    def forward(self, reset: bool = True) -> float:
        """Return averaged score.

        Returns
        -------
        float
            Averaged score
        """
        ave_score = self.total_score / self.total_instances
        if reset is True:
            self._reset_internal_state()
        return ave_score
