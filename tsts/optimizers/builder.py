from typing import Iterable

from torch.optim import Optimizer
from tsts.cfg import CfgNode as CN
from tsts.core import OPTIMIZERS

__all__ = ["build_optimizer"]


def build_optimizer(params: Iterable, cfg: CN) -> Optimizer:
    optim_name = cfg.OPTIMIZER.NAME
    cls = OPTIMIZERS[optim_name]
    optim = cls.from_cfg(params, cfg)
    return optim
