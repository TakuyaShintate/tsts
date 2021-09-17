from torch import Tensor
from tsts.cfg import CfgNode as CN
from tsts.core import LOCALSCALERS
from tsts.models import Module

__all__ = ["LastStep"]


@LOCALSCALERS.register()
class LastStep(Module):
    def __init__(
        self,
    ) -> None:
        super(LastStep, self).__init__()

    @classmethod
    def from_cfg(
        cls,
        num_in_feats: int,
        num_out_feats: int,
        cfg: CN,
    ) -> "LastStep":
        local_scaler = cls()
        return local_scaler

    def forward(self, bias: Tensor) -> Tensor:
        return bias[:, -1].unsqueeze(1)
