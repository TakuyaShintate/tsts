from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import (
    Conv1d,
    Dropout,
    LeakyReLU,
    Linear,
    ModuleList,
    ReplicationPad1d,
    Sequential,
    Tanh,
)
from tsts.cfg import CfgNode as CN
from tsts.core import MODELS

from .module import Module

__all__ = ["SCINet"]


class ConvModule(Module):
    def __init__(
        self,
        num_in_feats: int,
        kernel_size: int,
        expansion_rate: float,
        dropout_rate: float = 0.5,
    ) -> None:
        super(ConvModule, self).__init__()
        self.num_in_feats = num_in_feats
        self.kernel_size = kernel_size
        self.expansion_rate = expansion_rate
        self.dropout_rate = dropout_rate
        self._init_conv_module()

    def _init_conv_module(self) -> None:
        self.conv_module = Sequential(
            ReplicationPad1d(self.kernel_size // 2),
            Conv1d(
                self.num_in_feats,
                int(self.expansion_rate * self.num_in_feats),
                self.kernel_size,
            ),
            LeakyReLU(),
            Dropout(self.dropout_rate),
            ReplicationPad1d(self.kernel_size // 2),
            Conv1d(
                int(self.expansion_rate * self.num_in_feats),
                self.num_in_feats,
                self.kernel_size,
            ),
            Tanh(),
        )

    def forward(self, mb_feats: Tensor) -> Tensor:
        mb_feats = mb_feats.transpose(-2, -1).contiguous()
        mb_feats = self.conv_module(mb_feats)
        mb_feats = mb_feats.transpose(-2, -1).contiguous()
        return mb_feats


class SCIBlock(Module):
    def __init__(
        self,
        num_in_feats: int,
        kernel_size: int,
        expansion_rate: float,
        dropout_rate: float = 0.5,
    ) -> None:
        super(SCIBlock, self).__init__()
        self.num_in_feats = num_in_feats
        self.kernel_size = kernel_size
        self.expansion_rate = expansion_rate
        self.dropout_rate = dropout_rate
        self._init_ops()

    def _init_ops(self) -> None:
        self.phi = ConvModule(
            self.num_in_feats,
            self.kernel_size,
            self.expansion_rate,
            self.dropout_rate,
        )
        self.psi = ConvModule(
            self.num_in_feats,
            self.kernel_size,
            self.expansion_rate,
            self.dropout_rate,
        )
        self.rho = ConvModule(
            self.num_in_feats,
            self.kernel_size,
            self.expansion_rate,
            self.dropout_rate,
        )
        self.eta = ConvModule(
            self.num_in_feats,
            self.kernel_size,
            self.expansion_rate,
            self.dropout_rate,
        )

    def _split_mb_feats(self, mb_feats: Tensor) -> Tuple[Tensor, Tensor]:
        # Padding
        (batch_size, num_steps, _) = mb_feats.size()
        if num_steps % 2 == 1:
            padding = mb_feats.new_zeros((batch_size, num_steps, 1))
            mb_feats = torch.cat([mb_feats, padding], dim=2)
        f_even = mb_feats[:, ::2]
        f_odd = mb_feats[:, 1::2]
        return (f_even, f_odd)

    def _scale_and_shift(self, f_even: Tensor, f_odd: Tensor) -> Tuple[Tensor, Tensor]:
        # Escape f_even and f_odd
        _f_even = f_even
        _f_odd = f_odd
        f_even = f_even * self.psi(_f_odd).exp()
        f_odd = f_odd * self.phi(_f_even).exp()
        _f_even = f_even
        _f_odd = f_odd
        f_even = f_even - self.eta(_f_odd)
        f_odd = f_odd + self.rho(_f_even)
        return (f_even, f_odd)

    def forward(self, mb_feats: Tensor) -> Tuple[Tensor, Tensor]:
        (f_even, f_odd) = self._split_mb_feats(mb_feats)
        (f_even, f_odd) = self._scale_and_shift(f_even, f_odd)
        return (f_even, f_odd)


@MODELS.register()
class SCINet(Module):
    def __init__(
        self,
        num_in_feats: int,
        num_out_feats: int,
        lookback: int,
        horizon: int,
        depth: int = 3,
        kernel_size: int = 5,
        expansion_rate: float = 4.0,
        dropout_rate: float = 0.5,
        use_regressor_across_feats: bool = False,
    ) -> None:
        super(SCINet, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.lookback = lookback
        self.horizon = horizon
        self.depth = depth
        self.kernel_size = kernel_size
        self.expansion_rate = expansion_rate
        self.dropout_rate = dropout_rate
        self.use_regressor_across_feats = use_regressor_across_feats
        self._init_sciblocks()
        self._init_regressor()

    @classmethod
    def from_cfg(cls, num_in_feats: int, num_out_feats: int, cfg: CN) -> "SCINet":
        lookback = cfg.IO.LOOKBACK
        horizon = cfg.IO.HORIZON
        depth = cfg.MODEL.DEPTH
        kernel_size = cfg.MODEL.KERNEL_SIZE
        expansion_rate = cfg.MODEL.EXPANSION_RATE
        dropout_rate = cfg.MODEL.DROPOUT_RATE
        use_regressor_across_feats = cfg.MODEL.USE_REGRESSOR_ACROSS_TIME
        model = cls(
            num_in_feats,
            num_out_feats,
            lookback,
            horizon,
            depth,
            kernel_size,
            expansion_rate,
            dropout_rate,
            use_regressor_across_feats,
        )
        return model

    def _init_sciblocks(self) -> None:
        sciblock = SCIBlock(
            self.num_in_feats,
            self.kernel_size,
            self.expansion_rate,
            self.dropout_rate,
        )
        self.sciblocks = ModuleList([sciblock])
        for d in range(1, self.depth):
            for _ in range(2 ** d):
                sciblock = SCIBlock(
                    self.num_in_feats,
                    self.kernel_size,
                    self.expansion_rate,
                    self.dropout_rate,
                )
                self.sciblocks.append(sciblock)

    def _init_regressor(self) -> None:
        self.regressor_across_time = Linear(self.lookback, self.horizon)
        if self.use_regressor_across_feats is True:
            self.regressor_across_feats = Linear(self.num_in_feats, self.num_out_feats)

    def _merge_results(self, current_state: List[Tensor]) -> Tensor:
        mb_feats = torch.stack(current_state, dim=1)
        (batch_size, num_f, num_steps, num_feats) = mb_feats.size()
        mb_feats = mb_feats.transpose(1, 2).contiguous()
        mb_feats = mb_feats.view(batch_size, num_steps * num_f, num_feats)
        return mb_feats

    def _run_regressor(self, mb_feats: Tensor) -> Tensor:
        mb_feats = mb_feats.transpose(-2, -1).contiguous()
        mb_feats = self.regressor_across_time(mb_feats)
        mb_feats = mb_feats.transpose(-2, -1).contiguous()
        if self.use_regressor_across_feats is True:
            mb_feats = self.regressor_across_feats(mb_feats)
        return mb_feats

    def forward(
        self,
        X: Tensor,
        X_mask: Tensor,
        time_stamps: Optional[Tensor] = None,
    ) -> Tensor:
        current_state = [X]
        for d in range(self.depth):
            for i in range(2 ** d):
                f = current_state.pop(0)
                (f_even, f_odd) = self.sciblocks[i](f)
                current_state.append(f_odd)
                current_state.append(f_even)
        mb_feats = self._merge_results(current_state)
        mb_feats = mb_feats + X
        mb_feats = self._run_regressor(mb_feats)
        return mb_feats
