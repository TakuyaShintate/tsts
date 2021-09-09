from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Linear, LSTMCell, ModuleList, ReLU
from torch.nn.modules.container import Sequential
from tsts.cfg import CfgNode as CN
from tsts.core import MODELS

from .module import Module


@MODELS.register()
class Seq2Seq(Module):
    """Seq2Seq implementation.

    Example
    -------
    Add following section to use Seq2Seq.

    .. code-block:: yaml

        MODEL:
          NAME: "Seq2Seq"
          NUM_H_UNITS: 64

    Parameters
    ----------
    num_in_feats : int
        Number of input features

    num_out_feats : int
        Number of output features

    horizon : int. optional
        Indicate how many steps it predicts by default 1

    num_h_units : int, optional
        Number of hidden units, by default 64

    depth : int, optional
        Number of hidden layers, bu default 2

    add_last_step_val : bool, optional
        If True, Add x_t (the last value of input time series) to every output, by default False
    """

    def __init__(
        self,
        num_in_feats: int,
        num_out_feats: int,
        horizon: int = 1,
        num_h_units: int = 64,
        depth: int = 2,
        add_last_step_val: bool = False,
    ) -> None:
        super(Seq2Seq, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.horizon = horizon
        self.num_h_units = num_h_units
        self.depth = depth
        self.add_last_step_val = add_last_step_val
        self._init_hidden_layers()
        self._init_regressor()

    @classmethod
    def from_cfg(cls, num_in_feats: int, num_out_feats: int, cfg: CN) -> "Seq2Seq":
        horizon = cfg.IO.HORIZON
        num_h_units = cfg.MODEL.NUM_H_UNITS
        depth = cfg.MODEL.DEPTH
        add_last_step_val = cfg.MODEL.ADD_LAST_STEP_VAL
        model = cls(
            num_in_feats,
            num_out_feats,
            horizon=horizon,
            num_h_units=num_h_units,
            depth=depth,
            add_last_step_val=add_last_step_val,
        )
        return model

    def _init_hidden_layers(self) -> None:
        self.encoder = ModuleList([LSTMCell(self.num_in_feats, self.num_h_units)])
        for _ in range(self.depth - 1):
            self.encoder.append(LSTMCell(self.num_h_units, self.num_h_units))
        self.decoder = ModuleList()
        for _ in range(self.depth):
            self.decoder.append(LSTMCell(self.num_h_units, self.num_h_units))

    def _init_regressor(self) -> None:
        self.regressor = Sequential(
            Linear(self.num_h_units, self.num_h_units),
            ReLU(),
            Linear(self.num_h_units, self.num_out_feats),
        )

    def _init_memory(
        self, batch_size: int, device: torch.device
    ) -> Tuple[List[Tensor], ...]:
        h = []
        c = []
        for _ in range(self.depth):
            h.append(torch.zeros((batch_size, self.num_h_units), device=device))
            c.append(torch.zeros((batch_size, self.num_h_units), device=device))
        return (h, c)

    def _run_encoder(self, X: Tensor) -> List[Tensor]:
        batch_size = X.size(0)
        device = X.device
        (h, c) = self._init_memory(batch_size, device)
        num_in_steps = X.size(1)
        for t in range(num_in_steps):
            h_t = X[:, t]
            for i in range(self.depth):
                (h_t, c_t) = self.encoder[i](h_t, (h[i], c[i]))
                h[i] = h_t
                c[i] = c_t
        return h

    def _run_decoder(self, h: List[Tensor], bias: Tensor) -> Tensor:
        batch_size = bias.size(0)
        device = bias.device
        (_, c) = self._init_memory(batch_size, device)
        mb_preds = []
        h_t = h[-1]
        for _ in range(self.horizon):
            for i in range(self.depth):
                (h_t, c_t) = self.decoder[i](h_t, (h[i], c[i]))
                h[i] = h_t
                c[i] = c_t
            y_t = self.regressor(h_t)
            if self.add_last_step_val is True:
                y_t = y_t + bias[:, -1]
            mb_preds.append(y_t.unsqueeze(1))
        return torch.cat(mb_preds, dim=1)

    def forward(
        self,
        X: Tensor,
        bias: Tensor,
        X_mask: Tensor,
        time_stamps: Optional[Tensor] = None,
    ) -> Tensor:
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
        h = self._run_encoder(X)
        mb_preds = self._run_decoder(h, bias)
        return mb_preds
