from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Linear, LSTMCell, Module, ModuleList

__all__ = ["seq2seq"]


class Seq2Seq(Module):
    """Seq2Seq implementation.

    Parameters
    ----------
    num_in_feats : int
        Number of input features

    num_out_feats : int
        Number of output features

    horizon : int, optional
        Indicate how many steps it predicts, by default 1

    num_h_units : int, optional
        Number of hidden units, by default 64

    depth : int, optional
        Number of hidden layers, bu default 2
    """

    def __init__(
        self,
        num_in_feats: int,
        num_out_feats: int,
        horizon: int = 1,
        num_h_units: int = 64,
        depth: int = 2,
    ) -> None:
        super(Seq2Seq, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.horizon = horizon
        self.num_h_units = num_h_units
        self.depth = depth
        self._init_hidden_layers()
        self._init_regressor()

    def _init_hidden_layers(self) -> None:
        self.encoder = ModuleList([LSTMCell(self.num_in_feats, self.num_h_units)])
        for _ in range(self.depth - 1):
            self.encoder.append(LSTMCell(self.num_h_units, self.num_h_units))
        self.decoder = ModuleList()
        for _ in range(self.depth):
            self.decoder.append(LSTMCell(self.num_h_units, self.num_h_units))

    def _init_regressor(self) -> None:
        self.regressor = Linear(
            self.num_h_units,
            self.num_out_feats,
        )

    def _init_memory(
        self, batch_size: int, device: torch.device
    ) -> Tuple[List[Tensor], ...]:
        h = []
        c = []
        for _ in range(self.depth):
            h.append(torch.randn(batch_size, self.num_h_units, device=device))
            c.append(torch.randn(batch_size, self.num_h_units, device=device))
        return (h, c)

    def _run_encoder(self, mb_feats: Tensor) -> Tensor:
        batch_size = mb_feats.size(0)
        device = mb_feats.device
        (h, c) = self._init_memory(batch_size, device)
        num_in_steps = mb_feats.size(1)
        for t in range(num_in_steps):
            h_t = mb_feats[:, t]
            for i in range(self.depth):
                (h_t, c_t) = self.encoder[i](h_t, (h[i], c[i]))
                h[i] = h_t
                c[i] = c_t
        return h_t

    def _run_decoder(self, mb_feats: Tensor) -> Tensor:
        batch_size = mb_feats.size(0)
        device = mb_feats.device
        (h, c) = self._init_memory(batch_size, device)
        hs = [mb_feats.unsqueeze(1)]
        for _ in range(self.horizon):
            h_t = hs[-1].squeeze(1)
            for i in range(self.depth):
                (h_t, c_t) = self.decoder[i](h_t, (h[i], c[i]))
                h[i] = h_t
                c[i] = c_t
            hs.append(h_t.unsqueeze(1))
        mb_feats = torch.cat(hs[1:], dim=1)
        return mb_feats

    def _run_regressor(self, mb_feats: Tensor) -> Tensor:
        mb_preds = self.regressor(mb_feats)
        mb_preds = mb_preds
        return mb_preds

    def forward(self, mb_feats: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        mb_feats = self._run_encoder(mb_feats)
        mb_feats = self._run_decoder(mb_feats)
        mb_preds = self._run_regressor(mb_feats)
        return mb_preds


def seq2seq(
    num_in_feats: int,
    num_out_feats: int,
    lookback: int = 1,
    horizon: int = 1,
    num_h_units: int = 64,
    depth: int = 2,
) -> Seq2Seq:
    return Seq2Seq(
        num_in_feats,
        num_out_feats,
        horizon,
        num_h_units,
        depth,
    )
