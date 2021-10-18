from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import LOCALSCALERS
from tsts.models import Module

__all__ = ["NOOP"]


@LOCALSCALERS.register()
class NOOP(Module):
    def __init__(
        self,
    ) -> None:
        super(NOOP, self).__init__()

    @classmethod
    def from_cfg(
        cls,
        num_in_feats: int,
        num_out_feats: int,
        cfg: CN,
    ) -> "NOOP":
        local_scaler = cls()
        return local_scaler

    def forward(self, bias: Tensor) -> Tensor:
        return bias.new_zeros((1,))
