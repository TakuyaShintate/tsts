from tsts.cfg import CfgNode as CN
from tsts.core import SCHEDULERS
from tsts.optimizers import Optimizer

from .scheduler import Scheduler

__all__ = ["IdentityScheduler"]


@SCHEDULERS.register()
class IdentityScheduler(Scheduler):
    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    @classmethod
    def from_cfg(cls, optimizer: Optimizer, cfg: CN) -> "IdentityScheduler":
        scheduler = cls(optimizer)
        return scheduler

    def step(self) -> None:
        pass
