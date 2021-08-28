from tsts.cfg import CfgNode as CN
from tsts.core import SCHEDULERS
from tsts.optimizers import Optimizer

from .scheduler import Scheduler


def build_scheduler(optimizer: Optimizer, cfg: CN) -> Scheduler:
    scheduler_name = cfg.SCHEDULER.NAME
    cls = SCHEDULERS[scheduler_name]
    scheduler = cls.from_cfg(optimizer, cfg)
    return scheduler
