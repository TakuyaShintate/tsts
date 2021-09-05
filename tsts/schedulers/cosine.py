from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR
from tsts.cfg import CfgNode as CN
from tsts.core import SCHEDULERS
from tsts.optimizers import Optimizer

from .scheduler import Scheduler

__all__ = ["CosineAnnealing"]


@SCHEDULERS.register()
class CosineAnnealing(Scheduler):
    """Cosine annealing scheduler implementation.

    Example
    -------
    .. code-block:: python

        SCHEDULER:
          NAME: "CosineAnnealing"
          T_MAX: 10

    Parameters
    ----------
    optimizer : Optimizer
        Target optimizer

    T_max : int
        Maximum number of iterations (from pytorch)

    eta_min : float, optional
        Minimum learning rate (from pytorch), default 0.0
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self._init_scheduler()

    @classmethod
    def from_cfg(cls, optimizer: Optimizer, cfg: CN) -> "CosineAnnealing":
        T_max = cfg.SCHEDULER.T_MAX
        eta_min = cfg.SCHEDULER.ETA_MIN
        scheduler = cls(optimizer, T_max, eta_min)
        return scheduler

    def _init_scheduler(self) -> None:
        self.scheduler = _CosineAnnealingLR(
            self.optimizer,
            self.T_max,
            self.eta_min,
        )

    def step(self) -> None:
        self.scheduler.step()
