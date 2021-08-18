from torch import Tensor
from torch.nn.modules.loss import _Loss
from tsts.cfg import CfgNode as CN

__all__ = ["Metric"]


class Metric(_Loss):
    def __init__(self) -> None:
        super(Metric, self).__init__()
        self._reset_internal_state()

    @classmethod
    def from_cfg(cls, cfg: CN) -> "Metric":
        raise NotImplementedError

    def _reset_internal_state(self) -> None:
        self.total_score = 0.0
        self.total_instances = 0.0

    def update(self, Z: Tensor, y: Tensor, y_mask: Tensor) -> None:
        raise NotImplementedError
