import typing
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Conv1d,
    Dropout,
    LayerNorm,
    Linear,
    MaxPool1d,
    ModuleList,
)
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

    def forward(self, mb_feats: Tensor) -> Tensor:
        t = mb_feats.size(1)
        return typing.cast(Tensor, self.embedding[:, :t])  # type: ignore


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
    ) -> None:
        super(SelfAttention, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.num_heads = num_heads
        self._init_projector()

    def _init_projector(self) -> None:
        num_out_feats = (self.num_out_feats // self.num_heads) * self.num_heads
        self.projector = Linear(num_out_feats, self.num_in_feats)

    def _apply_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        num_feats = q.size(-1)
        scale = 1.0 / np.sqrt(num_feats)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = torch.softmax(scale * scores, dim=-1)
        v_new = (scores[..., None] * v[:, :, None, :, :]).sum(-2)
        (batch_size, _, num_vals, _) = v_new.size()
        v_new = v_new.view(batch_size, num_vals, -1)
        return v_new

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        mb_feats = self._apply_attention(q, k, v)
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
        self._init_projector()

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

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        (scores, m_top) = self._get_attention_scores(q, k, *self._get_num_samples(q, k))
        num_feats = q.size(-1)
        scale = 1.0 / np.sqrt(num_feats)
        scores = scale * scores
        num_queries = q.size(2)
        mb_feats = self._apply_attention(v, scores, m_top, num_queries)
        (batch_size, _, num_vals, _) = v.size()
        if self.mix is True:
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
        self._init_qkv()
        self._init_attention()
        self._init_dropout()
        self._init_norms()
        self._init_convs()

    def _init_qkv(self) -> None:
        num_out_feats = (self.num_out_feats // self.num_heads) * self.num_heads
        self.qkv = Linear(
            self.num_in_feats,
            3 * num_out_feats,
        )

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

    def _get_qkv(self, mb_feats: Tensor) -> Tuple[Tensor, ...]:
        (b, n, u) = mb_feats.shape
        qkv = self.qkv(mb_feats)
        qkv = qkv.reshape(b, n, 3, self.num_heads, u // self.num_heads)
        # q, k, v: (batch size, number of heads, number of tokens, number of features)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        (q, k, v) = (qkv[0], qkv[1], qkv[2])
        return (q, k, v)

    def forward(self, mb_feats: Tensor) -> Tensor:
        (q, k, v) = self._get_qkv(mb_feats)
        mb_feats = self.attention(q, k, v)
        mb_feats = self.dropout(mb_feats)
        mb_feats = skip_feats = self.norm1(mb_feats)
        mb_feats = self.conv1(mb_feats.transpose(-2, -1))
        mb_feats = torch.relu(mb_feats)
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

    def _reshape_qkv(self, qkv: Tensor, start: int, end: int) -> Tensor:
        batch_size = qkv.size(0)
        n = qkv[:, start:end]
        n = n.reshape(batch_size, n.size(1), self.num_heads, -1)
        n = n.permute(0, 2, 1, 3)
        return n

    def _get_qkv(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, ...]:
        mb_feats = torch.cat([q, k, v], dim=1)
        (b, _, _) = mb_feats.size()
        qkv = self.qkv(mb_feats)
        q = self._reshape_qkv(qkv, 0, q.size(1))
        k = self._reshape_qkv(qkv, q.size(1), q.size(1) + k.size(1))
        v = self._reshape_qkv(
            qkv, q.size(1) + k.size(1), q.size(1) + k.size(1) + v.size(1)
        )
        return (q, k, v)

    def forward(self, mb_feats: Tensor, mb_enc_feats: Tensor) -> Tensor:
        (q, k, v) = self._get_qkv(
            mb_feats,
            mb_enc_feats,
            mb_enc_feats,
        )
        mb_feats = mb_feats + self.dropout(self.attention(q, q, q))
        mb_feats = self.norm1(mb_feats)
        mb_feats = mb_feats + self.dropout(self.cross_attention(q, k, v))
        mb_feats = skip_feats = self.norm2(mb_feats)
        mb_feats = self.conv1(mb_feats.transpose(-2, -1))
        mb_feats = torch.relu(mb_feats)
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
    """Informer implementation."""

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
        dec_in_size: int = 48,
        add_last_step_val: bool = False,
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
        self.add_last_step_val = add_last_step_val
        self._init_embeddings()
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
        add_last_step_val = cfg.MODEL.ADD_LAST_STEP_VAL
        model = cls(
            num_in_feats,
            num_out_feats,
            lookback,
            horizon,
            add_last_step_val=add_last_step_val,
        )
        return model

    def _init_embeddings(self) -> None:
        self.token_embedding_enc = TokenEmbedding(
            self.num_in_feats,
            self.num_h_feats,
        )
        self.position_embedding_enc = PositionEmbedding(
            self.num_h_feats,
            self.lookback,
        )
        self.token_embedding_dec = TokenEmbedding(
            self.num_in_feats,
            self.num_h_feats,
        )
        self.position_embedding_dec = PositionEmbedding(
            self.num_h_feats,
            self.lookback,
        )

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

    def forward(self, X: Tensor, bias: Tensor, X_mask: Tensor) -> Tensor:
        mb_enc_feats = self.token_embedding_enc(X)
        mb_enc_feats = mb_enc_feats + self.position_embedding_enc(mb_enc_feats)
        mb_enc_feats = self._run_encoders(mb_enc_feats)
        X_dec = X[:, -self.dec_in_size :]
        mb_dec_feats = self.token_embedding_dec(X_dec)
        mb_dec_feats = mb_dec_feats + self.position_embedding_dec(mb_dec_feats)
        mb_dec_feats = self._run_decoders(mb_dec_feats, mb_enc_feats)
        mb_feats = self.projector(mb_dec_feats)
        if self.add_last_step_val is True:
            mb_feats = mb_feats + bias[:, -1:]
        return mb_feats[:, -self.horizon :]
