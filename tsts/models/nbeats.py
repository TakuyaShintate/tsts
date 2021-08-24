from typing import Tuple

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
    ) -> None:
        super(IdentityBasis, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, mb_feats: Tensor) -> Tuple[Tensor, Tensor]:
        return (
            mb_feats[:, : self.backcast_size],
            mb_feats[:, -self.forecast_size :],
        )


class Block(Module):
    """Main component of N-Beats."""

    def __init__(
        self,
        num_in_steps: int,
        num_out_steps: int,
        num_h_units: int,
        depth: int,
    ) -> None:
        assert depth > 0
        super(Block, self).__init__()
        self.num_in_steps = num_in_steps
        self.num_out_steps = num_out_steps
        self.num_h_units = num_h_units
        self.depth = depth
        self.basis_fn = IdentityBasis(
            num_in_steps,
            num_out_steps,
        )
        self._init_layers()

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
    ) -> None:
        super(NBeats, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.lookback = lookback
        self.horizon = horizon
        self.num_h_units = num_h_units
        self.depth = depth
        self.stack_size = stack_size
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
        model = cls(
            num_in_feats,
            num_out_feats,
            lookback,
            horizon,
            num_h_units,
            depth,
            stack_size,
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
