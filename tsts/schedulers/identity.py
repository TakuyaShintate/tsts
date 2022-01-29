from tsts.cfg import CfgNode as CN
from tsts.core import SCHEDULERS
from tsts.optimizers import Optimizer

from .scheduler import Scheduler

__all__ = ["IdentityScheduler"]


@SCHEDULERS.register()
class IdentityScheduler(Scheduler):
    """Dummy scheduler.

    Example
    -------
    .. code-block:: python

        SCHEDULER:
          NAME: "IdentityScheduler"

    Parameters
    ----------
    optimizer : Optimizer
        Target optimizer
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        warmup_steps: int = 0,
    ) -> None:
        super(IdentityScheduler, self).__init__(
            optimizer,
            base_lr,
            warmup_steps,
        )

    @classmethod
    def from_cfg(
        cls,
        optimizer: Optimizer,
        iters_per_epoch: int,
        cfg: CN,
    ) -> "IdentityScheduler":
        base_lr = cfg.OPTIMIZER.LR
        warmup_steps = cfg.SCHEDULER.WARMUP_STEPS
        scheduler = cls(optimizer, base_lr, warmup_steps)
        return scheduler

    def step(self) -> None:
        self.warmup()
        self.T += 1.0
