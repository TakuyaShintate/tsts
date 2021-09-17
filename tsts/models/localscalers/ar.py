from torch import Tensor
from torch.nn import Linear
from tsts.cfg import CfgNode as CN
from tsts.core import LOCALSCALERS
from tsts.models import Module

__all__ = ["AutoRegressiveModel"]


@LOCALSCALERS.register()
class AutoRegressiveModel(Module):
    def __init__(
        self,
        num_in_feats: int,
        num_out_feats: int,
        num_steps: int,
    ) -> None:
        super(AutoRegressiveModel, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.num_steps = num_steps
        self._init_linear()

    @classmethod
    def from_cfg(
        cls,
        num_in_feats: int,
        num_out_feats: int,
        cfg: CN,
    ) -> "AutoRegressiveModel":
        num_steps = cfg.LOCALSCALER.NUM_STEPS
        local_scaler = cls(num_in_feats, num_out_feats, num_steps)
        return local_scaler

    def _init_linear(self) -> None:
        self.linear = Linear(self.num_steps, 1)

    def forward(self, bias: Tensor) -> Tensor:
        bias = bias[:, -self.num_steps :]
        bias = bias.permute(0, 2, 1).reshape(-1, self.num_steps)
        bias = self.linear(bias)
        bias = bias.view(-1, 1, self.num_out_feats)
        return bias
