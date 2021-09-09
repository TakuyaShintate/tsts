from torch.optim.lr_scheduler import StepLR
from tsts.cfg import CfgNode as CN
from tsts.core import SCHEDULERS
from tsts.optimizers import Optimizer

from .scheduler import Scheduler

__all__ = ["StepScheduler"]


@SCHEDULERS.register()
class StepScheduler(Scheduler):
    """Step scheduler implementation.

    Example
    -------
    .. code-block:: python

        SCHEDULER:
          NAME: "StepScheduler"
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
        step_size: int,
        gamma: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self._init_scheduler()

    @classmethod
    def from_cfg(cls, optimizer: Optimizer, cfg: CN) -> "StepScheduler":
        step_size = cfg.SCHEDULER.STEP_SIZE
        gamma = cfg.SCHEDULER.GAMMA
        scheduler = cls(optimizer, step_size, gamma)
        return scheduler

    def _init_scheduler(self) -> None:
        self.scheduler = StepLR(
            self.optimizer,
            self.step_size,
            self.gamma,
        )

    def step(self) -> None:
        self.scheduler.step()
