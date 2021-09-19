from typing import Iterable

from tsts.cfg import CfgNode as CN
from tsts.core import OPTIMIZERS
from tsts.thirdparty.sam import SAM as _SAM

__all__ = ["SAM"]


@OPTIMIZERS.register()
class SAM(_SAM):
    """SAM implementation.

    Example
    -------
    .. code-block:: python

        OPTIMIZER:
          NAME: "SAM"
    """

    @classmethod
    def from_cfg(cls, params: Iterable, cfg: CN) -> "SAM":
        base_optimizer_name = cfg.OPTIMIZER.BASE_OPTIMIZER_NAME
        base_optimizer = OPTIMIZERS[base_optimizer_name].from_cfg(params, cfg)
        defaults = base_optimizer.defaults
        rho = cfg.OPTIMIZER.RHO
        optim = cls(params, OPTIMIZERS[base_optimizer_name], rho, **defaults)
        return optim

    @property
    def require_second_step(self) -> bool:
        return True
