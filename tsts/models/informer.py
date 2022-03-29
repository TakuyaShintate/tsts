import typing
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Conv1d,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    MaxPool1d,
    ModuleList,
    Parameter,
)
from tsts.cfg import CfgNode as CN
from tsts.core import MODELS
from tsts.types import MaybeTensor

from .module import Module

MIN_SIZE = 4
HOUR_SIZE = 24
DAY_SIZE = 32
WEEKDAY_SIZE = 7
MONTH_SIZE = 13


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
        for m in self.modules():
            if isinstance(m, Conv1d):
                init.kaiming_normal_(
                    m.weight,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )

    def forward(self, X: Tensor) -> Tensor:
        # X: (B, T, F)
        mb_feats = self.embedding(X.permute(0, 2, 1)).transpose(1, 2)
        return mb_feats


class PositionEmbedding(Module):
    def __init__(self, num_out_feats: int, lookback: int) -> None:
        super(PositionEmbedding, self).__init__()
        self.num_out_feats = num_out_feats
        self.lookback = 5000  # lookback
        self._init_embedding()

    def _init_embedding(self) -> None:
        embedding = torch.zeros((self.lookback, self.num_out_feats))
        embedding.requires_grad = False
        x = torch.arange(0.0, float(self.lookback))
        x = x.unsqueeze(1)
        m = torch.arange(0.0, float(self.num_out_feats), 2.0)
        m = m * -(np.log(10000.0) / self.num_out_feats)
        m = m.exp()
        embedding[:, 0::2] = torch.sin(x * m)
        embedding[:, 1::2] = torch.cos(x * m)
        embedding = embedding.unsqueeze(0)
        self.register_buffer("embedding", embedding)

    def forward(self, mb_feats: Tensor) -> Tensor:
        t = mb_feats.size(1)
        return typing.cast(Tensor, self.embedding[:, :t])  # type: ignore


class FixedEmbedding(Module):
    def __init__(self, num_in_feats: int, num_out_feats: int) -> None:
        super(FixedEmbedding, self).__init__()
        embedding = torch.zeros(num_in_feats, num_out_feats).float()
        x = torch.arange(0.0, float(num_in_feats))
        x = x.unsqueeze(1)
        m = torch.arange(0.0, float(num_out_feats), 2.0)
        m = m * -(np.log(10000.0) / num_out_feats)
        m = m.exp()
        embedding[:, 0::2] = torch.sin(x * m)
        embedding[:, 1::2] = torch.cos(x * m)
        self.embedding = Embedding(num_in_feats, num_out_feats)
        self.embedding.weight = Parameter(embedding, requires_grad=False)

    def forward(self, mb_feats: Tensor) -> Tensor:
        return self.embedding(mb_feats).detach()


class TemporalEmbedding(Module):
    def __init__(self, num_out_feats: int) -> None:
        super(TemporalEmbedding, self).__init__()
        self.num_out_feats = num_out_feats
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        self.embeddings = ModuleList()
        self.embeddings.append(FixedEmbedding(MONTH_SIZE, self.num_out_feats))
        self.embeddings.append(FixedEmbedding(WEEKDAY_SIZE, self.num_out_feats))
        self.embeddings.append(FixedEmbedding(DAY_SIZE, self.num_out_feats))
        self.embeddings.append(FixedEmbedding(HOUR_SIZE, self.num_out_feats))
        self.embeddings.append(FixedEmbedding(MIN_SIZE, self.num_out_feats))

    def forward(self, time_stamps: Tensor) -> Tensor:
        mb_feats = time_stamps.new_tensor(0.0)
        for i in range(time_stamps.size(2)):
            mb_feats = mb_feats + self.embeddings[i](time_stamps[:, :, i])
        return mb_feats


class ProbMask(Module):
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        num_queries: int,
        m_top: Tensor,
        scores: Tensor,
        device: torch.device = torch.device("cpu"),
    ):
        mask = torch.ones(
            num_queries,
            scores.shape[-1],
            dtype=torch.bool,
        )
        mask = mask.to(device)
        mask = mask.triu(1)
        mask = mask[None, None, :].expand(
            batch_size,
            num_heads,
            num_queries,
            scores.shape[-1],
        )
        b = torch.arange(batch_size)[:, None, None]
        h = torch.arange(num_heads)[None, :, None]
        indicator = mask[b, h, m_top, :].to(device)
        self.mask = indicator.view(scores.shape).to(device)


class SelfAttention(Module):
    def __init__(
        self,
        num_in_feats: int,
        num_out_feats: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
    ) -> None:
        super(SelfAttention, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self._init_qkv()
        self._init_dropout()
        self._init_projector()

    def _init_qkv(self) -> None:
        num_out_feats = (self.num_out_feats // self.num_heads) * self.num_heads
        self.q_projector = Linear(self.num_in_feats, num_out_feats)
        self.k_projector = Linear(self.num_in_feats, num_out_feats)
        self.v_projector = Linear(self.num_in_feats, num_out_feats)

    def _init_dropout(self) -> None:
        self.dropout = Dropout(self.dropout_rate)

    def _init_projector(self) -> None:
        num_out_feats = (self.num_out_feats // self.num_heads) * self.num_heads
        self.projector = Linear(num_out_feats, self.num_in_feats)

    def _apply_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        num_feats = q.size(-1)
        scale = 1.0 / np.sqrt(num_feats)
        scores = torch.einsum("blhe,bshe->bhls", q, k)
        scores = self.dropout(torch.softmax(scale * scores, dim=-1))
        v_new = torch.einsum("bhls,bshd->blhd", scores, v)
        return v_new

    def _process_qkv(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        (b_q, n_q, u_q) = q.size()
        (b_k, n_k, u_k) = k.size()
        (b_v, n_v, u_v) = v.size()
        q = self.q_projector(q)
        k = self.k_projector(k)
        v = self.v_projector(v)
        q = q.reshape(b_q, n_q, self.num_heads, u_q // self.num_heads)
        k = k.reshape(b_k, n_k, self.num_heads, u_k // self.num_heads)
        v = v.reshape(b_v, n_v, self.num_heads, u_v // self.num_heads)
        return (q, k, v)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        (q, k, v) = self._process_qkv(q, k, v)
        mb_feats = self._apply_attention(q, k, v)
        (batch_size, num_queries, _, _) = q.size()
        mb_feats = mb_feats.contiguous()
        mb_feats = mb_feats.view(batch_size, num_queries, -1)
        mb_feats = self.projector(mb_feats)
        return mb_feats


class ProbSparseSelfAttention(Module):
    def __init__(
        self,
        num_in_feats: int,
        num_out_feats: int,
        is_decoding: bool,
        mix: bool,
        num_heads: int = 8,
        contraction_factor: int = 5,
    ) -> None:
        super(ProbSparseSelfAttention, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.is_decoding = is_decoding
        self.mix = mix
        self.num_heads = num_heads
        self.contraction_factor = contraction_factor
        self._init_qkv()
        self._init_projector()

    def _init_qkv(self) -> None:
        num_out_feats = (self.num_out_feats // self.num_heads) * self.num_heads
        self.q_projector = Linear(self.num_in_feats, num_out_feats)
        self.k_projector = Linear(self.num_in_feats, num_out_feats)
        self.v_projector = Linear(self.num_in_feats, num_out_feats)

    def _init_projector(self) -> None:
        num_out_feats = (self.num_out_feats // self.num_heads) * self.num_heads
        self.projector = Linear(num_out_feats, self.num_in_feats)

    def _get_num_samples(self, q: Tensor, k: Tensor) -> Tuple[int, int]:
        """Return the number of query and key samples.

        Parameters
        ----------
        q : Tensor (N, H, L_q, F)
            Queries

        k : Tensor (N, H, L_k, F)
            Keys

        Returns
        -------
        Tuple[int, int]
            Number of query and key samples.
        """
        num_queries = q.size(2)
        num_keys = k.size(2)
        num_q_samples = np.ceil(np.log(num_queries))
        num_q_samples *= self.contraction_factor
        num_q_samples = int(num_q_samples)
        num_k_samples = np.ceil(np.log(num_keys))
        num_k_samples *= self.contraction_factor
        num_k_samples = int(num_k_samples)
        # NOTE: When the length is too small, num_query_samples could be larger than num_queries
        if num_q_samples > num_queries:
            num_q_samples = num_queries
        if num_k_samples > num_keys:
            num_k_samples = num_keys
        return (num_q_samples, num_k_samples)

    def _get_random_keys(self, q: Tensor, k: Tensor, num_key_samples: int) -> Tensor:
        """Randomly select num_key_samples for each query.

        Parameters
        ----------
        q : Tensor (N, H, L_q, F)
            Queries

        k : Tensor (N, H, L_k, F)
            Keys

        num_k_samples : int
            Number of keys selected (=L_k')

        Notes
        -----
        Note L_k' <= L_k.

        Returns
        -------
        Tensor (N, H, L_q, L_k', F)
            Selected keys for each query
        """
        (num_batches, num_heads, num_queries, num_feats) = q.size()
        num_keys = k.size(2)
        k_new = k.unsqueeze(-3)
        k_new = k_new.expand(
            num_batches,
            num_heads,
            num_queries,
            num_keys,
            num_feats,
        )
        i = torch.randint(num_keys, (num_queries, num_key_samples))
        k_new = k_new[:, :, torch.arange(num_queries).unsqueeze(1), i, :]
        return k_new

    def _get_topk_queries(
        self,
        q: Tensor,
        k_new: Tensor,
        num_keys: int,
        num_q_samples: int,
    ) -> Tensor:
        """Get k queries which have higher KL div values.

        Parameters
        ----------
        q : Tensor (N, H, L_q, F)
            Querys

        k_new : Tensor (N, H, L_q, L_k', F)
            Selected keys for each query

        num_keys: int
            Number of keys (=L_k)

        num_q_samples: int
            Number of top-k queries (=L_q')

        Returns
        -------
        Tensor (N, H, L_q')
            Top-k queries
        """
        q_new = torch.matmul(q.unsqueeze(-2), k_new.transpose(-2, -1))
        # q_new: (N, H, L_q, L_k')
        q_new = q_new.squeeze()
        m = q_new.max(-1)[0] - torch.div(q_new.sum(-1), num_keys)
        m_top = m.topk(num_q_samples, sorted=False)[1]
        return m_top

    def _get_attention_scores(
        self,
        q: Tensor,
        k: Tensor,
        num_q_samples: int,
        num_k_samples: int,
    ) -> Tuple[Tensor, Tensor]:
        """Return attention scores between reduced q and k.

        Parameters
        ----------
        q : Tensor
            Queries

        k : Tensor
            Keys

        num_q_samples : int
            Number of top-k queries (=L_q')

        num_k_samples : int
            Number of keys selected (=L_k')

        Returns
        -------
        Tuple[Tensor, Tensor]
            [description]
        """
        k_new = self._get_random_keys(q, k, num_k_samples)
        num_keys = k.size(2)
        m_top = self._get_topk_queries(q, k_new, num_keys, num_q_samples)
        (batch_size, num_heads, _, _) = q.size()
        n = torch.arange(batch_size)[:, None, None]
        h = torch.arange(num_heads)[None, :, None]
        q_new = q[n, h, m_top]
        scores = torch.matmul(q_new, k.transpose(-2, -1))
        return (scores, m_top)

    def _apply_attention(
        self,
        v: Tensor,
        scores: Tensor,
        m_top: Tensor,
        num_queries: int,
    ) -> Tensor:
        # TODO: Add masking feature
        (batch_size, num_heads, _, num_feats) = v.size()
        if self.is_decoding is True:
            v_new = v.cumsum(dim=-2)
            attn_mask = ProbMask(
                batch_size,
                num_heads,
                num_queries,
                m_top,
                scores,
                device=v.device,
            )
            scores.masked_fill_(attn_mask.mask, -np.inf)
        else:
            v_new = v.mean(dim=-2, keepdim=True)
            v_new = v_new.expand(batch_size, num_heads, num_queries, num_feats)
            v_new = v_new.clone()
        scores = torch.softmax(scores, dim=-1)
        n = torch.arange(batch_size)[:, None, None]
        h = torch.arange(num_heads)[None, :, None]
        v_new[n, h, m_top] = torch.matmul(scores, v)
        v_new = v_new.contiguous()
        return v_new

    def _process_qkv(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        (b_q, n_q, u_q) = q.size()
        (b_k, n_k, u_k) = k.size()
        (b_v, n_v, u_v) = v.size()
        q = self.q_projector(q)
        k = self.k_projector(k)
        v = self.v_projector(v)
        q = q.reshape(b_q, n_q, self.num_heads, u_q // self.num_heads)
        k = k.reshape(b_k, n_k, self.num_heads, u_k // self.num_heads)
        v = v.reshape(b_v, n_v, self.num_heads, u_v // self.num_heads)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        return (q, k, v)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        (q, k, v) = self._process_qkv(q, k, v)
        (scores, m_top) = self._get_attention_scores(q, k, *self._get_num_samples(q, k))
        num_feats = q.size(-1)
        scale = 1.0 / np.sqrt(num_feats)
        scores = scale * scores
        num_queries = q.size(2)
        mb_feats = self._apply_attention(v, scores, m_top, num_queries)
        (batch_size, _, num_vals, _) = v.size()
        if self.mix is False:
            mb_feats = mb_feats.transpose(2, 1).contiguous()
        mb_feats = mb_feats.view(batch_size, num_vals, -1)
        mb_feats = self.projector(mb_feats)
        return mb_feats


class AttentionBlock(Module):
    def __init__(
        self,
        num_in_feats: int,
        num_out_feats: int,
        num_heads: int = 8,
        contraction_factor: int = 5,
        dropout_rate: float = 0.1,
        expansion_rate: float = 1.0,
    ) -> None:
        # NOTE: Take num_in_feats and num_out_feats separately for compatibility
        super(AttentionBlock, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.num_heads = num_heads
        self.contraction_factor = contraction_factor
        self.dropout_rate = dropout_rate
        self.expantion_rate = expansion_rate
        self._init_attention()
        self._init_dropout()
        self._init_norms()
        self._init_convs()

    def _init_attention(self) -> None:
        self.attention = ProbSparseSelfAttention(
            self.num_in_feats,
            self.num_out_feats,
            False,
            False,
            self.num_heads,
            self.contraction_factor,
        )

    def _init_dropout(self) -> None:
        self.dropout = Dropout(self.dropout_rate)

    def _init_norms(self) -> None:
        self.norm1 = LayerNorm(self.num_in_feats)
        self.norm2 = LayerNorm(self.num_in_feats)

    def _init_convs(self) -> None:
        num_h_feats = int(self.expantion_rate * self.num_out_feats)
        self.conv1 = Conv1d(self.num_in_feats, num_h_feats, 1)
        self.conv2 = Conv1d(num_h_feats, self.num_out_feats, 1)

    def forward(self, mb_feats: Tensor) -> Tensor:
        mb_feats = self.attention(mb_feats, mb_feats, mb_feats)
        mb_feats = mb_feats + self.dropout(mb_feats)
        mb_feats = skip_feats = self.norm1(mb_feats)
        mb_feats = self.conv1(mb_feats.transpose(-2, -1))
        mb_feats = F.gelu(mb_feats)
        mb_feats = self.dropout(mb_feats)
        mb_feats = self.conv2(mb_feats).transpose(-2, -1)
        mb_feats = self.dropout(mb_feats)
        mb_feats = self.norm2(skip_feats + mb_feats)
        return mb_feats


class CrossAttentionBlock(Module):
    def __init__(
        self,
        num_in_feats: int,
        num_out_feats: int,
        num_heads: int = 8,
        contraction_factor: int = 5,
        dropout_rate: float = 0.1,
        expansion_rate: float = 1.0,
    ) -> None:
        # NOTE: Take num_in_feats and num_out_feats separately for compatibility
        super(CrossAttentionBlock, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.num_heads = num_heads
        self.contraction_factor = contraction_factor
        self.dropout_rate = dropout_rate
        self.expantion_rate = expansion_rate
        self._init_qkv()
        self._init_attention()
        self._init_dropout()
        self._init_norms()
        self._init_convs()

    def _init_qkv(self) -> None:
        num_out_feats = (self.num_out_feats // self.num_heads) * self.num_heads
        self.qkv = Linear(
            self.num_in_feats,
            num_out_feats,
        )

    def _init_attention(self) -> None:
        self.attention = ProbSparseSelfAttention(
            self.num_in_feats,
            self.num_out_feats,
            True,
            True,
            self.num_heads,
            self.contraction_factor,
        )
        self.cross_attention = SelfAttention(
            self.num_in_feats,
            self.num_out_feats,
            self.num_heads,
            self.dropout_rate,
        )

    def _init_dropout(self) -> None:
        self.dropout = Dropout(self.dropout_rate)

    def _init_norms(self) -> None:
        self.norm1 = LayerNorm(self.num_in_feats)
        self.norm2 = LayerNorm(self.num_in_feats)
        self.norm3 = LayerNorm(self.num_in_feats)

    def _init_convs(self) -> None:
        num_h_feats = int(self.expantion_rate * self.num_out_feats)
        self.conv1 = Conv1d(self.num_in_feats, num_h_feats, 1)
        self.conv2 = Conv1d(num_h_feats, self.num_out_feats, 1)

    def forward(self, mb_feats: Tensor, mb_enc_feats: Tensor) -> Tensor:
        attn = self.attention(mb_feats, mb_feats, mb_feats)
        mb_feats = mb_feats + self.dropout(attn)
        mb_feats = self.norm1(mb_feats)
        attn = self.cross_attention(mb_feats, mb_enc_feats, mb_enc_feats)
        mb_feats = mb_feats + self.dropout(attn)
        mb_feats = skip_feats = self.norm2(mb_feats)
        mb_feats = self.conv1(mb_feats.transpose(-2, -1))
        mb_feats = F.gelu(mb_feats)
        mb_feats = self.dropout(mb_feats)
        mb_feats = self.conv2(mb_feats).transpose(-2, -1)
        mb_feats = self.dropout(mb_feats)
        mb_feats = self.norm3(skip_feats + mb_feats)
        return mb_feats


class DistilModule(Module):
    def __init__(self, num_in_feats: int, num_out_feats: int) -> None:
        super(DistilModule, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self._init_conv()
        self._init_norm()
        self._init_pooling_module()

    def _init_conv(self) -> None:
        self.conv = Conv1d(
            self.num_in_feats,
            self.num_out_feats,
            3,
            padding=1,
            padding_mode="circular",
        )

    def _init_norm(self) -> None:
        self.norm = BatchNorm1d(self.num_out_feats)

    def _init_pooling_module(self) -> None:
        self.pooling_module = MaxPool1d(3, 2, 1)

    def forward(self, mb_feats: Tensor) -> Tensor:
        mb_feats = mb_feats.transpose(-2, -1)
        mb_feats = self.conv(mb_feats)
        mb_feats = self.norm(mb_feats)
        mb_feats = F.elu(mb_feats)
        mb_feats = self.pooling_module(mb_feats)
        mb_feats = mb_feats.transpose(-2, -1)
        return mb_feats


@MODELS.register()
class Informer(Module):
    """Informer implementation.

    Example
    -------
    Add following section to use Informer.

    .. code-block:: yaml

        MODEL:
          NAME: "Informer"
          NUM_H_UNITS: 512

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

    num_h_feats : int, optional
        Number of hidden features, by default 512

    num_encoders : int, optional
        Number of encoders, by default 2

    num_decoders : int, optional
        Number of decoders, by default 1

    num_heads : int, optional
        Number of heads of multi-head self attention, by default 8

    contraction_factor : int, optional
        Factor which detemines the number of samples of queries and keys in
        ProbSparseSelfAttention, by default 5

    dropout_rate : int, optional
        Dropout rate, by default 0.05

    expansion_rate : int, optional
        Expansion rate which determines the number of filters in conv layers after attention, by
        default 4.0

    distil : bool, optional
        Flag if use distillation module after each encoder except the last one, by default True

    dec_in_size : int, optional
        Size of input to decoder (last dec_in_size values of input to encoder are used), by
        default 24

    add_last_step_val : bool, optional
        If True, Add x_t (the last value of input time series) to every output, by default False
    """

    def __init__(
        self,
        num_in_feats: int,
        num_out_feats: int,
        lookback: int,
        horizon: int,
        num_h_feats: int = 512,
        num_encoders: int = 2,
        num_decoders: int = 1,
        num_heads: int = 8,
        contraction_factor: int = 5,
        dropout_rate: float = 0.05,
        expansion_rate: float = 4.0,
        distil: bool = True,
        dec_in_size: int = 24,
    ) -> None:
        super(Informer, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.lookback = lookback
        self.horizon = horizon
        self.num_h_feats = num_h_feats
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self.num_heads = num_heads
        self.contraction_factor = contraction_factor
        self.dropout_rate = dropout_rate
        self.expansion_rate = expansion_rate
        self.distil = distil
        self.dec_in_size = dec_in_size
        self._init_embeddings()
        self._init_dropout()
        self._init_encoders()
        self._init_decoders()
        self._init_projector()

    @classmethod
    def from_cfg(
        cls,
        num_in_feats: int,
        num_out_feats: int,
        cfg: CN,
    ) -> "Informer":
        lookback = cfg.IO.LOOKBACK
        horizon = cfg.IO.HORIZON
        num_h_feats = cfg.MODEL.NUM_H_FEATS
        num_encoders = cfg.MODEL.NUM_ENCODERS
        num_decoders = cfg.MODEL.NUM_DECODERS
        num_heads = cfg.MODEL.NUM_HEADS
        contraction_factor = cfg.MODEL.CONTRACTION_FACTOR
        dropout_rate = cfg.MODEL.DROPOUT_RATE
        expansion_rate = cfg.MODEL.EXPANSION_RATE
        dec_in_size = cfg.MODEL.DECODER_IN_LENGTH
        model = cls(
            num_in_feats,
            num_out_feats,
            lookback,
            horizon,
            num_h_feats,
            num_encoders,
            num_decoders,
            num_heads,
            contraction_factor,
            dropout_rate,
            expansion_rate,
            True,
            dec_in_size,
        )
        return model

    def _init_embeddings(self) -> None:
        self.token_embedding_enc = TokenEmbedding(
            self.num_in_feats,
            self.num_h_feats,
        )
        self.token_embedding_dec = TokenEmbedding(
            self.num_in_feats,
            self.num_h_feats,
        )
        self.position_embedding_enc = PositionEmbedding(
            self.num_h_feats,
            self.lookback,
        )
        self.position_embedding_dec = PositionEmbedding(
            self.num_h_feats,
            self.lookback + self.horizon,
        )
        self.temporal_embedding_enc = TemporalEmbedding(
            self.num_h_feats,
        )
        self.temporal_embedding_dec = TemporalEmbedding(
            self.num_h_feats,
        )

    def _init_dropout(self) -> None:
        self.dropout = Dropout(self.dropout_rate)

    def _init_encoders(self) -> None:
        self.encoders = ModuleList()
        for _ in range(self.num_encoders):
            enc = AttentionBlock(
                self.num_h_feats,
                self.num_h_feats,
                self.num_heads,
                self.contraction_factor,
                self.dropout_rate,
                self.expansion_rate,
            )
            self.encoders.append(enc)
        if self.distil is True:
            self.distil_modules = ModuleList()
            for _ in range(self.num_encoders - 1):
                dm = DistilModule(self.num_h_feats, self.num_h_feats)
                self.distil_modules.append(dm)
        self.norm_enc = LayerNorm(self.num_h_feats)

    def _init_decoders(self) -> None:
        self.decoders = ModuleList()
        for _ in range(self.num_decoders):
            dec = CrossAttentionBlock(
                self.num_h_feats,
                self.num_h_feats,
                self.num_heads,
                self.contraction_factor,
                self.dropout_rate,
                self.expansion_rate,
            )
            self.decoders.append(dec)
        self.norm_dec = LayerNorm(self.num_h_feats)

    def _init_projector(self) -> None:
        self.projector = Linear(self.num_h_feats, self.num_out_feats)

    def _run_encoders(self, mb_feats: Tensor) -> Tensor:
        if self.distil is True:
            for i in range(self.num_encoders - 1):
                mb_feats = self.encoders[i](mb_feats)
                mb_feats = self.distil_modules[i](mb_feats)
            mb_feats = self.encoders[-1](mb_feats)
        else:
            for i in range(self.num_encoders):
                mb_feats = self.encoders[i](mb_feats)
        mb_feats = self.norm_enc(mb_feats)
        return mb_feats

    def _run_decoders(self, mb_feats: Tensor, mb_enc_feats: Tensor) -> Tensor:
        for i in range(self.num_decoders):
            mb_feats = self.decoders[i](mb_feats, mb_enc_feats)
        mb_feats = self.norm_dec(mb_feats)
        return mb_feats

    def forward(
        self,
        X: Tensor,
        X_mask: Tensor,
        bias: MaybeTensor = None,
        time_stamps: MaybeTensor = None,
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
        mb_enc_feats = self.token_embedding_enc(X)
        mb_enc_feats += self.position_embedding_enc(X)
        if time_stamps is not None:
            time_stamps_X = time_stamps[:, : self.lookback]  # type: ignore
            mb_enc_feats += self.temporal_embedding_enc(time_stamps_X)
        mb_enc_feats = self.dropout(mb_enc_feats)
        mb_enc_feats = self._run_encoders(mb_enc_feats)
        if bias is not None:
            X_dec1 = bias[:, -self.dec_in_size :]
        else:
            X_dec1 = X[:, -self.dec_in_size :]
        (b, _, u) = X_dec1.size()
        X_dec2 = torch.zeros(b, self.horizon, u)
        device = X_dec1.device
        X_dec2 = X_dec2.to(device)
        X_dec = torch.cat([X_dec1, X_dec2], dim=1)
        mb_dec_feats = self.token_embedding_dec(X_dec)
        mb_dec_feats += self.position_embedding_dec(X_dec)
        if time_stamps is not None:
            time_stamps_y = time_stamps[:, -self.dec_in_size - self.horizon :]  # type: ignore
            mb_dec_feats += self.temporal_embedding_dec(time_stamps_y)
        mb_dec_feats = self.dropout(mb_dec_feats)
        mb_dec_feats = self._run_decoders(mb_dec_feats, mb_enc_feats)
        mb_feats = self.projector(mb_dec_feats)
        return mb_feats[:, -self.horizon :]
