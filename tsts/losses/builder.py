from typing import List

from tsts.cfg import CfgNode as CN
from tsts.core import LOSSES

from .loss import Loss

__all__ = ["build_losses"]


def build_losses(cfg: CN) -> List[Loss]:
    """Build a list of loss functions.

    Parameters
    ----------
    cfg : CN
        Global configuration

    Returns
    -------
    List[Loss]
        List of loss functions
    """
    losses = []
    num_losses = len(cfg.LOSSES.NAMES)
    for i in range(num_losses):
        loss_name = cfg.LOSSES.NAMES[i]
        cls = LOSSES[loss_name]
        loss_args = cfg.LOSSES.ARGS[i]
        loss = cls.from_cfg(cfg, **loss_args)
        losses.append(loss)
    return losses
