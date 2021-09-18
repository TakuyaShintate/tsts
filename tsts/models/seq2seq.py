from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Linear, LSTMCell, ModuleList
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

    num_h_feats : int, optional
        Number of hidden units, by default 64

    depth : int, optional
        Number of hidden layers, bu default 2
    """

    def __init__(
        self,
        num_in_feats: int,
        num_out_feats: int,
        horizon: int = 1,
        num_h_feats: int = 64,
        num_encoders: int = 1,
        num_decoders: int = 1,
    ) -> None:
        super(Seq2Seq, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.horizon = horizon
        self.num_h_feats = num_h_feats
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self._init_embedding()
        self._init_hidden_layers()
        self._init_regressor()

    @classmethod
    def from_cfg(cls, num_in_feats: int, num_out_feats: int, cfg: CN) -> "Seq2Seq":
        horizon = cfg.IO.HORIZON
        num_h_feats = cfg.MODEL.NUM_H_FEATS
        num_encoders = cfg.MODEL.NUM_ENCODERS
        num_decoders = cfg.MODEL.NUM_DECODERS
        model = cls(
            num_in_feats,
            num_out_feats,
            horizon=horizon,
            num_h_feats=num_h_feats,
            num_encoders=num_encoders,
            num_decoders=num_decoders,
        )
        return model

    def _init_embedding(self) -> None:
        self.token_embedding = Linear(self.num_in_feats, self.num_h_feats)

    def _init_hidden_layers(self) -> None:
        self.encoders = ModuleList()
        for _ in range(self.num_encoders):
            self.encoders.append(LSTMCell(self.num_h_feats, self.num_h_feats))
        self.decoders = ModuleList()
        for _ in range(self.num_decoders):
            self.decoders.append(LSTMCell(self.num_h_feats, self.num_h_feats))

    def _init_regressor(self) -> None:
        self.regressor = Linear(self.num_h_feats, self.num_out_feats)

    def _init_memory(
        self,
        batch_size: int,
        device: torch.device,
        depth: int,
    ) -> Tuple[List[Tensor], ...]:
        h = []
        c = []
        for _ in range(depth):
            h.append(torch.zeros((batch_size, self.num_h_feats), device=device))
            c.append(torch.zeros((batch_size, self.num_h_feats), device=device))
        return (h, c)

    def _run_encoder(self, X: Tensor) -> List[Tensor]:
        batch_size = X.size(0)
        device = X.device
        (h, c) = self._init_memory(batch_size, device, self.num_encoders)
        num_in_steps = X.size(1)
        for t in range(num_in_steps):
            h_t = self.token_embedding(X[:, t])
            for i in range(self.num_encoders):
                (h_t, c_t) = self.encoders[i](h_t, (h[i], c[i]))
                h[i] = h_t
                c[i] = c_t
        return h

    def _run_decoder(self, h: List[Tensor]) -> Tensor:
        batch_size = h[0].size(0)
        device = h[0].device
        (_, c) = self._init_memory(batch_size, device, self.num_decoders)
        mb_preds = []
        h_t = h[-1]
        for _ in range(self.horizon):
            for i in range(self.num_decoders):
                (h_t, c_t) = self.decoders[i](h_t, (h[i], c[i]))
                h[i] = h_t
                c[i] = c_t
            y_t = self.regressor(h_t)
            mb_preds.append(y_t.unsqueeze(1))
        return torch.cat(mb_preds, dim=1)

    def forward(
        self,
        X: Tensor,
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
        mb_preds = self._run_decoder(h)
        return mb_preds
