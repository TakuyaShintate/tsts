import typing

import numpy as np
import torch
from torch import Tensor
from torch.nn import Conv1d
from tsts.cfg import CfgNode as CN
from tsts.core import MODELS

from .module import Module


class TokenEmbedding(Module):
    def __init__(self, num_in_feats: int, num_out_feats: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self._init_embedding()

    def _init_embedding(self) -> None:
        self.embedding = Conv1d(
            self.num_in_feats,
            self.num_out_feats,
            3,
            padding=1,
            padding_mode="circular",
        )

    def forward(self, X: Tensor) -> Tensor:
        # X: (B, T, F)
        mb_feats = self.embedding(X.permute(0, 2, 1)).transpose(1, 2)
        return mb_feats


class PositionEmbedding(Module):
    def __init__(self, num_out_feats: int, lookback: int) -> None:
        super(PositionEmbedding, self).__init__()
        self.num_out_feats = num_out_feats
        self.lookback = lookback
        self._init_embedding()

    def _init_embedding(self) -> None:
        embedding = torch.zeros((self.lookback, self.num_out_feats))
        embedding.requires_grad = False
        x = torch.arange(0.0, self.lookback)
        x = x.unsqueeze(1)
        m = torch.arange(0.0, self.num_out_feats, 2.0)
        m = m * -(np.log(10000.0) / self.num_out_feats)
        m = m.exp()
        embedding[:, 0::2] = torch.cos(x * m)
        embedding[:, 1::2] = torch.sin(x * m)
        embedding = embedding.unsqueeze(0)
        self.register_buffer("embedding", embedding)

    def forward(self) -> Tensor:
        return typing.cast(Tensor, self.embedding)


@MODELS.register()
class Informer(Module):
    """Informer implementation."""

    def __init__(
        self,
        num_in_feats: int,
        num_out_feats: int,
        lookback: int,
        horizon: int,
    ) -> None:
        super(Informer, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.lookback = lookback
        self.horizon = horizon
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        self.token_embedding = TokenEmbedding(
            self.num_in_feats,
            self.num_out_feats,
        )
        self.position_embedding = PositionEmbedding(
            self.num_out_feats,
            self.lookback,
        )

    @classmethod
    def from_cfg(
        cls,
        num_in_feats: int,
        num_out_feats: int,
        cfg: CN,
    ) -> "Informer":
        pass

    def forward(self, X: Tensor, X_mask: Tensor) -> Tensor:
        mb_feats = self.token_embedding(X)
        mb_feats = mb_feats + self.position_embedding()
        print(mb_feats)
        pass
