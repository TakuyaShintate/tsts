from typing import Iterable

from torch.optim import Adam as _Adam
from tsts.cfg import CfgNode as CN
from tsts.core import OPTIMIZERS

__all__ = ["Adam"]


@OPTIMIZERS.register()
class Adam(_Adam):
    """Adam implementation.

    Example
    -------
    .. code-block:: python

        OPTIMIZER:
          NAME: "Adam"
          LR: 0.001
    """

    @classmethod
    def from_cfg(cls, params: Iterable, cfg: CN) -> "Adam":
        lr = cfg.OPTIMIZER.LR
        weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY
        optim = cls(
            params,
            lr,
            weight_decay=weight_decay,
        )
        return optim
