from typing import Iterable

from torch.optim import SGD as _SGD
from tsts.cfg import CfgNode as CN
from tsts.core import OPTIMIZERS

__all__ = ["SGD"]


@OPTIMIZERS.register()
class SGD(_SGD):
    """SGD implementation.

    Example
    -------
    .. code-block:: python

        OPTIMIZER:
          NAME: "SGD"
          LR: 0.01
          MOMENTUM: 0.9
    """

    @classmethod
    def from_cfg(cls, params: Iterable, cfg: CN) -> "SGD":
        lr = cfg.OPTIMIZER.LR
        momentum = cfg.OPTIMIZER.MOMENTUM
        weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY
        optim = cls(
            params,
            lr,
            momentum,
            weight_decay=weight_decay,
        )
        return optim
