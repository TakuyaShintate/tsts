from torch.optim.lr_scheduler import LambdaLR
from tsts.cfg import CfgNode as CN
from tsts.core import SCHEDULERS
from tsts.optimizers import Optimizer

from .scheduler import Scheduler

__all__ = ["ExponentialDecay"]


@SCHEDULERS.register()
class ExponentialDecay(Scheduler):
    """ExponentialDecay scheduler.

    It works in the same way to ExponentialDecay scheduler defined in Keras.

    Example
    -------
    .. code-block:: python

        TRAINING:
          NUM_EPOCHS: 100
        SCHEDULER:
          NAME: "ExponentialDecay"
          DECAY_RATE: 0.96

    Parameters
    ----------
    optimizer : Optimizer
        Target optimizer

    num_epochs : int
        Total number of epochs

    decay_rate : float, optional
        Decaying parameter, by default 0.96
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        decay_steps: float,
        decay_rate: float = 0.96,
        warmup_steps: int = 0,
    ) -> None:
        super(ExponentialDecay, self).__init__(
            optimizer,
            base_lr,
            warmup_steps,
        )
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self._init_scheduler()

    @classmethod
    def from_cfg(
        cls,
        optimizer: Optimizer,
        iters_per_epoch: int,
        cfg: CN,
    ) -> "ExponentialDecay":
        base_lr = cfg.OPTIMIZER.LR
        decay_steps = cfg.SCHEDULER.DECAY_STEPS
        decay_rate = cfg.SCHEDULER.DECAY_RATE
        warmup_steps = cfg.SCHEDULER.WARMUP_STEPS
        scheduler = cls(
            optimizer,
            base_lr,
            decay_steps,
            decay_rate,
            warmup_steps,
        )
        return scheduler

    def _init_scheduler(self) -> None:
        self.scheduler = LambdaLR(
            self.optimizer,
            lambda step: self.decay_rate ** (step / self.decay_steps),
        )

    def step(self) -> None:
        self.scheduler.step()
