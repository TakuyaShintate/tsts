from torch.nn.modules.loss import _Loss
from tsts.cfg import CfgNode as CN

__all__ = ["Loss"]


class Loss(_Loss):
    @classmethod
    def from_cfg(cls, cfg: CN) -> "Loss":
        raise NotImplementedError
