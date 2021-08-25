import numpy as np
import torch
from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import METRICS

from .metric import Metric

__all__ = ["MAPE"]


@METRICS.register()
class MAPE(Metric):
    """MAPE implementation.

    Example
    -------
    .. code-block:: yaml

        METRICS:
          NAMES: ["MAPE"]
    """

    def __init__(self) -> None:
        super(MAPE, self).__init__()

    @classmethod
    def from_cfg(cls, cfg: CN) -> "MAPE":
        metric = cls()
        return metric

    def update(self, Z: Tensor, y: Tensor, y_mask: Tensor) -> None:
        c = y_mask / y
        c[c != c] = 0.0
        c[c == np.inf] = 0.0
        mape = torch.abs((Z - y) * c)
        score = mape.sum().item()
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
        ave_score = 100.0 * self.total_score / self.total_instances
        self._reset_internal_state()
        return ave_score
