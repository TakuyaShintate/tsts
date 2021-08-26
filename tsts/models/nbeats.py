from typing import Tuple, Type, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Linear, Module, ModuleList, ReLU
from tsts.cfg import CfgNode as CN
from tsts.core import MODELS

__all__ = ["NBeats"]


class IdentityBasis(Module):
    def __init__(
        self,
        backcast_size: int,
        forecast_size: int,
        num_in_feats: int,
        num_out_feats: int,
        degree: int = 2,
    ) -> None:
        super(IdentityBasis, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, mb_feats: Tensor) -> Tuple[Tensor, Tensor]:
        return (
            mb_feats[:, : self.backcast_size],
            mb_feats[:, -self.forecast_size :],
        )


class TrendBasis(Module):
    def __init__(
        self,
        backcast_size: int,
        forecast_size: int,
        num_in_feats: int,
        num_out_feats: int,
        degree: int = 2,
    ) -> None:
        super(TrendBasis, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.degree = degree
        self.p = degree + 1
        self._init_t()

    def _init_t(self) -> None:
        actual_backcast_size = self.backcast_size // self.num_in_feats
        actual_forecast_size = self.forecast_size // self.num_out_feats
        backcast_T = []
        forecast_T = []
        for i in range(3):
            backcast_t = np.arange(actual_backcast_size, dtype=np.float32)
            backcast_t = backcast_t / actual_backcast_size
            backcast_t = np.power(backcast_t, i)
            backcast_t = np.concatenate([backcast_t for _ in range(self.num_in_feats)])
            backcast_T.append(backcast_t[None, :])
            forecast_t = np.arange(actual_forecast_size, dtype=np.float32)
            forecast_t = forecast_t / actual_forecast_size
            forecast_t = np.power(forecast_t, i)
            forecast_t = np.concatenate([forecast_t for _ in range(self.num_out_feats)])
            forecast_T.append(forecast_t[None, :])
        _backcast_T = torch.tensor(np.concatenate(backcast_T))
        _forecast_T = torch.tensor(np.concatenate(forecast_T))
        self.register_buffer("backcast_T", _backcast_T)
        self.register_buffer("forecast_T", _forecast_T)

    def forward(self, mb_feats: Tensor) -> Tuple[Tensor, Tensor]:
        device = mb_feats.device
        bt = self.backcast_T[None, :, :].to(device)  # type: ignore
        ft = self.forecast_T[None, :, :].to(device)  # type: ignore
        backcast = (mb_feats[:, None, : self.backcast_size] * bt).sum(1)
        forecast = (mb_feats[:, None, -self.forecast_size :] * ft).sum(1)
        return (backcast, forecast)


class Block(Module):
    """Main component of N-Beats."""

    def __init__(
        self,
        num_in_steps: int,
        num_out_steps: int,
        num_h_units: int,
        depth: int,
        block_type: str,
        num_in_feats: int,
        num_out_feats: int,
        degree: int = 2,
    ) -> None:
        assert depth > 0
        super(Block, self).__init__()
        self.num_in_steps = num_in_steps
        self.num_out_steps = num_out_steps
        self.num_h_units = num_h_units
        self.depth = depth
        self.block_type = block_type
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.degree = degree
        self._init_layers()
        self._init_basis()

    def _init_layers(self) -> None:
        self.layers = ModuleList(
            [
                Linear(
                    self.num_in_steps,
                    self.num_h_units,
                ),
                ReLU(),
            ]
        )
        for _ in range(self.depth - 1):
            self.layers.append(
                Linear(
                    self.num_h_units,
                    self.num_h_units,
                )
            )
            self.layers.append(ReLU())
        self.layers.append(
            Linear(
                self.num_h_units,
                self.num_in_steps + self.num_out_steps,
            )
        )

    def _init_basis(self) -> None:
        if self.block_type == "identity":
            basis: Union[Type[IdentityBasis], Type[TrendBasis]] = IdentityBasis
        elif self.block_type == "trend":
            basis = TrendBasis
        else:
            ValueError("Currently, NBeats supports IdentityBasis and TrendBasis only")
        self.basis_fn = basis(
            self.num_in_steps,
            self.num_out_steps,
            self.num_in_feats,
            self.num_out_feats,
            self.degree,
        )

    def forward(self, mb_feats: Tensor) -> Tuple[Tensor, Tensor]:
        for i in range(self.depth):
            mb_feats = self.layers[i](mb_feats)
        mb_feats = self.layers[-1](mb_feats)
        (mb_feats, mb_preds) = self.basis_fn(mb_feats)
        return (mb_feats, mb_preds)


@MODELS.register()
class NBeats(Module):
    """N-Beats implementation.

    Example
    -------
    .. code-block:: yaml

        MODEL:
          NAME: "NBeats"
          NUM_H_UNITS: 512
          NUM_STACKS: 30

    Parameters
    ----------
    num_in_feats : int
        Number of input features

    num_out_feats : int
        Number of output features

    lookback : int
        Number of input time steps

    horizon : int. optional
        Indicate how many steps it predicts by default 1

    num_h_units : int
        Number of hidden units

    depth : int
        Number of hidden layers per block

    stack_size : int
        Number of blocks
    """

    def __init__(
        self,
        num_in_feats: int,
        num_out_feats: int,
        lookback: int,
        horizon: int,
        num_h_units: int = 512,
        depth: int = 2,
        stack_size: int = 30,
        block_type: str = "identity",
        degree: int = 2,
    ) -> None:
        super(NBeats, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.lookback = lookback
        self.horizon = horizon
        self.num_h_units = num_h_units
        self.depth = depth
        self.stack_size = stack_size
        self.block_type = block_type
        self.degree = degree
        self._init_stack()

    @classmethod
    def from_cfg(
        cls,
        num_in_feats: int,
        num_out_feats: int,
        cfg: CN,
    ) -> "NBeats":
        lookback = cfg.IO.LOOKBACK
        horizon = cfg.IO.HORIZON
        num_h_units = cfg.MODEL.NUM_H_UNITS
        depth = cfg.MODEL.DEPTH
        stack_size = cfg.MODEL.STACK_SIZE
        block_type = cfg.MODEL.BLOCK_TYPE
        degree = cfg.MODEL.DEGREE
        model = cls(
            num_in_feats,
            num_out_feats,
            lookback,
            horizon,
            num_h_units,
            depth,
            stack_size,
            block_type,
            degree,
        )
        return model

    def _init_stack(self) -> None:
        self.stack = ModuleList()
        for _ in range(self.stack_size):
            self.stack.append(
                Block(
                    self.num_in_feats * self.lookback,
                    self.num_out_feats * self.horizon,
                    self.num_h_units,
                    self.depth,
                    self.block_type,
                    self.num_in_feats,
                    self.num_out_feats,
                    self.degree,
                )
            )

    def forward(self, X: Tensor, X_mask: Tensor) -> Tensor:
        """Return prediction.

        Parameters
        ----------
        X : Tensor
            Input time series

        X_mask : Tensor
            Input time series mask

        Returns
        -------
        Tensor
            Prediction
        """
        batch_size = X.size(0)
        X = X.reshape(batch_size, -1)
        X_mask = X_mask.reshape(batch_size, -1)
        X_mask = X_mask.flip(dims=(1,))
        # Predict offset
        mb_total_preds = X[:, -1:]
        mb_feats = X.flip(dims=(1,))
        for i in range(self.stack_size):
            (current_mb_feats, mb_preds) = self.stack[i](mb_feats)
            mb_feats = (mb_feats - current_mb_feats) * X_mask
            mb_total_preds = mb_total_preds + mb_preds
        mb_total_preds = mb_total_preds.reshape(
            batch_size,
            self.horizon,
            self.num_out_feats,
        )
        return mb_total_preds
