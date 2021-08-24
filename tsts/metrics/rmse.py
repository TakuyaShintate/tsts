import torch.nn.functional as F
from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import METRICS

from .metric import Metric

__all__ = ["RMSE"]


@METRICS.register()
class RMSE(Metric):
    """RMSE implementation.

    Example
    -------
    .. code-block:: yaml

        METRICS:
          NAMES: ["RMSE"]
    """

    def __init__(self) -> None:
        super(RMSE, self).__init__()

    @classmethod
    def from_cfg(cls, cfg: CN) -> "RMSE":
        metric = cls()
        return metric

    def update(self, Z: Tensor, y: Tensor, y_mask: Tensor) -> None:
        mse = F.mse_loss(Z, y, reduction="none")
        rmse = mse.sqrt()
        rmse *= y_mask
        score = rmse.sum().item()
        self.total_score += score
        instances = y_mask.sum().item()
        self.total_instances += instances

    def forward(self) -> float:
        """Return averaged score.

        Returns
        -------
        float
            Averaged score
        """
        ave_score = self.total_score / self.total_instances
        self._reset_internal_state()
        return ave_score
