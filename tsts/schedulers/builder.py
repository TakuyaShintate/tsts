from tsts.cfg import CfgNode as CN
from tsts.core import SCHEDULERS
from tsts.optimizers import Optimizer

from .scheduler import Scheduler


def build_scheduler(optimizer: Optimizer, cfg: CN) -> Scheduler:
    """Build learning rate scheduler.

    Parameters
    ----------
    optimizer : Optimizer
        Target optimizer

    cfg : CN
        Global config

    Returns
    -------
    Scheduler
        Built learning rate scheduler
    """
    scheduler_name = cfg.SCHEDULER.NAME
    cls = SCHEDULERS[scheduler_name]
    scheduler = cls.from_cfg(optimizer, cfg)
    return scheduler
