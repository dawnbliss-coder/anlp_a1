"""Full Encoder-Decoder Transformer, assembled from the from-scratch
attention/norm/positional modules and driven by a config so that swapping
one ablation axis (pos encoding / attention / norm) only changes a flag,
not the layer wiring. Used directly by C1-C4; C5 (BLT) wraps the same
EncoderLayer/DecoderLayer stack as its "global" transformer in blt.py."""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, GroupedQueryAttention
from .ffn import PositionwiseFeedForward
from .norm import LayerNorm, RMSNorm, PreNormResidual
from .positional import SinusoidalPositionalEncoding, RotaryEmbedding

NORM_REGISTRY = {"layernorm": LayerNorm, "rmsnorm": RMSNorm}


def get_causal_mask(seq_len: int, device) -> torch.Tensor:
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask.view(1, 1, seq_len, seq_len)


def get_padding_mask(pad_mask_1d: torch.Tensor) -> torch.Tensor:
    """pad_mask_1d: (batch, seq_len) bool, True = real token, False = pad."""
    return pad_mask_1d[:, None, None, :]


def make_attention(attn_type, d_model, n_heads, n_kv_heads, dropout, rotary):
    if attn_type == "mha":
        return MultiHeadAttention(d_model, n_heads, dropout=dropout, rotary=rotary)
    if attn_type == "gqa":
        return GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout=dropout, rotary=rotary)
    raise ValueError(f"Unknown attention type: {attn_type}")


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, dropout, norm_cls, attn_type, rotary):
        super().__init__()
        self.self_attn = make_attention(attn_type, d_model, n_heads, n_kv_heads, dropout, rotary)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.attn_block = PreNormResidual(d_model, norm_cls, dropout)
        self.ffn_block = PreNormResidual(d_model, norm_cls, dropout)

    def forward(self, x, mask):
        x = self.attn_block(x, lambda h: self.self_attn(h, h, h, mask=mask))
        x = self.ffn_block(x, self.ffn)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, dropout, norm_cls, attn_type, rotary):
        super().__init__()
        self.self_attn = make_attention(attn_type, d_model, n_heads, n_kv_heads, dropout, rotary)
        # cross-attention keys/values come from the encoder domain, so RoPE (a
        # relative self-attention scheme) is not applied here regardless of config.
        self.cross_attn = make_attention(attn_type, d_model, n_heads, n_kv_heads, dropout, None)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.self_block = PreNormResidual(d_model, norm_cls, dropout)
        self.cross_block = PreNormResidual(d_model, norm_cls, dropout)
        self.ffn_block = PreNormResidual(d_model, norm_cls, dropout)

    def forward(self, x, memory, self_mask, cross_mask):
        x = self.self_block(x, lambda h: self.self_attn(h, h, h, mask=self_mask))
        x = self.cross_block(x, lambda h: self.cross_attn(h, memory, memory, mask=cross_mask))
        x = self.ffn_block(x, self.ffn)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, layers, final_norm):
        super().__init__()
        self.layers = layers
        self.final_norm = final_norm

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)


class TransformerDecoder(nn.Module):
    def __init__(self, layers, final_norm):
        super().__init__()
        self.layers = layers
        self.final_norm = final_norm

    def forward(self, x, memory, self_mask, cross_mask):
        for layer in self.layers:
            x = layer(x, memory, self_mask, cross_mask)
        return self.final_norm(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        tgt_pad_idx: int = 0,
        d_model: int = 256,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 1024,
        pos_encoding: str = "sinusoidal",
        attention: str = "mha",
        norm: str = "layernorm",
    ):
        super().__init__()
        self.tgt_pad_idx = tgt_pad_idx
        self.d_model = d_model
        norm_cls = NORM_REGISTRY[norm]

        # No padding_idx on the source embedding: cipher bytes span the full
        # 0-255 range, so byte value 0 can be a real (non-pad) source token.
        # Padded source positions are masked out via src_pad_mask instead.
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.embed_scale = d_model ** 0.5
        self.dropout = nn.Dropout(dropout)

        self.use_rope = pos_encoding == "rope"
        if self.use_rope:
            self.rotary = RotaryEmbedding(d_model // n_heads, max_len=max_len)
            self.pos_enc = None
        else:
            self.rotary = None
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        self.encoder = TransformerEncoder(
            nn.ModuleList([
                EncoderLayer(d_model, n_heads, n_kv_heads, d_ff, dropout, norm_cls, attention, self.rotary)
                for _ in range(n_layers)
            ]),
            norm_cls(d_model),
        )
        self.decoder = TransformerDecoder(
            nn.ModuleList([
                DecoderLayer(d_model, n_heads, n_kv_heads, d_ff, dropout, norm_cls, attention, self.rotary)
                for _ in range(n_layers)
            ]),
            norm_cls(d_model),
        )
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_pad_mask):
        x = self.src_embed(src) * self.embed_scale
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.dropout(x)
        return self.encoder(x, get_padding_mask(src_pad_mask))

    def decode(self, tgt, memory, tgt_pad_mask, src_pad_mask):
        x = self.tgt_embed(tgt) * self.embed_scale
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.dropout(x)
        causal = get_causal_mask(tgt.size(1), tgt.device)
        self_mask = get_padding_mask(tgt_pad_mask) & causal
        cross_mask = get_padding_mask(src_pad_mask)
        x = self.decoder(x, memory, self_mask, cross_mask)
        return self.out_proj(x)

    def forward(self, src, tgt, src_pad_mask, tgt_pad_mask):
        memory = self.encode(src, src_pad_mask)
        return self.decode(tgt, memory, tgt_pad_mask, src_pad_mask)
