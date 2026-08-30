"""Scaled dot-product attention, Multi-Head Attention (MHA), and
Grouped-Query Attention (GQA), built from raw linear algebra (no
nn.MultiheadAttention)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import apply_rotary_pos_emb


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """q,k,v: (batch, heads, seq_len, head_dim). mask: bool, True = keep."""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, rotary=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rotary = rotary  # RotaryEmbedding module, or None -> self-attn only

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        q, k, v = self._split(self.w_q(query)), self._split(self.w_k(key)), self._split(self.w_v(value))

        if self.rotary is not None:
            cos, sin = self.rotary(q.size(2))
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        out, _ = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)
        out = out.transpose(1, 2).contiguous().view(query.size(0), query.size(1), -1)
        return self.w_o(out)


class GroupedQueryAttention(nn.Module):
    """Like MHA but K/V use fewer heads than Q; each KV head is shared by
    n_heads // n_kv_heads query heads (repeat_interleave before attention)."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0, rotary=None):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.rotary = rotary

        self.w_q = nn.Linear(d_model, n_heads * self.head_dim)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.w_o = nn.Linear(n_heads * self.head_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        b, tq, _ = query.shape
        tk = key.shape[1]

        q = self.w_q(query).view(b, tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(key).view(b, tk, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(value).view(b, tk, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.rotary is not None:
            cos, sin = self.rotary(tq)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        out, _ = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)
        out = out.transpose(1, 2).contiguous().view(b, tq, self.n_heads * self.head_dim)
        return self.w_o(out)
