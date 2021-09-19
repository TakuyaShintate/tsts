from typing import Iterable

from torch.optim import AdamW as _AdamW
from tsts.cfg import CfgNode as CN
from tsts.core import OPTIMIZERS

__all__ = ["AdamW"]


@OPTIMIZERS.register()
class AdamW(_AdamW):
    """AdamW implementation.

    Example
    -------
    .. code-block:: python

        OPTIMIZER:
          NAME: "AdamW"
          LR: 0.001
    """

    @classmethod
    def from_cfg(cls, params: Iterable, cfg: CN) -> "AdamW":
        lr = cfg.OPTIMIZER.LR
        weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY
        optim = cls(
            params,
            lr,
            weight_decay=weight_decay,
        )
        return optim

    @property
    def require_second_step(self) -> bool:
        return False
