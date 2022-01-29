from torch.optim.lr_scheduler import \
    CosineAnnealingWarmRestarts as _CosineAnnealingWarmRestarts
from tsts.cfg import CfgNode as CN
from tsts.core import SCHEDULERS
from tsts.optimizers import Optimizer

from .scheduler import Scheduler

__all__ = ["CosineAnnealingWarmRestarts"]


@SCHEDULERS.register()
class CosineAnnealingWarmRestarts(Scheduler):
    """Cosine annealing scheduler implementation.

    Example
    -------
    .. code-block:: python

        SCHEDULER:
          NAME: "CosineAnnealingWithRestarts"
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
        base_lr: float,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        warmup_steps: int = 0,
    ) -> None:
        super(CosineAnnealingWarmRestarts, self).__init__(
            optimizer,
            base_lr,
            warmup_steps,
        )
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self._init_scheduler()

    @classmethod
    def from_cfg(
        cls,
        optimizer: Optimizer,
        iters_per_epoch: int,
        cfg: CN,
    ) -> "CosineAnnealingWarmRestarts":
        base_lr = cfg.OPTIMIZER.LR
        T_0 = cfg.SCHEDULER.T_0
        T_mult = cfg.SCHEDULER.T_MULT
        eta_min = cfg.SCHEDULER.ETA_MIN
        warmup_steps = cfg.SCHEDULER.WARMUP_STEPS
        scheduler = cls(
            optimizer,
            base_lr,
            T_0,
            T_mult,
            eta_min,
            warmup_steps,
        )
        return scheduler

    def _init_scheduler(self) -> None:
        self.scheduler = _CosineAnnealingWarmRestarts(
            self.optimizer,
            self.T_0,
            self.T_mult,
            self.eta_min,  # type: ignore
        )

    def step(self) -> None:
        self.scheduler.step()
