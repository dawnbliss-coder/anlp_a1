"""Simplified Byte Latent Transformer (BLT) components for C5.

Real BLT uses entropy-based dynamic patching; this is a coursework-scale
simplification that uses FIXED-size byte patches instead, as permitted by
the assignment ("simplified Byte Latent Transformer approach").

Pipeline: raw bytes -> Patcher -> LocalEncoder (byte-level, produces one
vector per patch) -> global Transformer (reused EncoderLayer/DecoderLayer
from transformer.py, operating on patch embeddings) -> LocalDecoder
(byte-level, autoregressively expands a patch context vector back into
patch_size bytes).

Dataset note: for this assignment's cipher (8 bits/plaintext-char), the
number of source bytes after chunking into 8-bit groups equals the number
of target plaintext characters exactly, which bounds generation length
without needing an explicit EOS byte.
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .ffn import PositionwiseFeedForward
from .norm import LayerNorm, PreNormResidual
from .positional import SinusoidalPositionalEncoding
from .transformer import (
    EncoderLayer,
    DecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    get_causal_mask,
    get_padding_mask,
)

PAD_BYTE = 0  # never occurs in this dataset's ASCII letters/space alphabet


class Patcher:
    """Groups a byte sequence into fixed-size, non-overlapping patches."""

    def __init__(self, patch_size: int, pad_byte: int = PAD_BYTE):
        self.patch_size = patch_size
        self.pad_byte = pad_byte

    def to_patches(self, byte_ids: torch.Tensor, byte_mask: torch.Tensor):
        b, n = byte_ids.shape
        remainder = n % self.patch_size
        if remainder != 0:
            pad_len = self.patch_size - remainder
            byte_ids = torch.cat([byte_ids, byte_ids.new_full((b, pad_len), self.pad_byte)], dim=1)
            byte_mask = torch.cat([byte_mask, byte_mask.new_zeros((b, pad_len))], dim=1)
        n_patches = byte_ids.size(1) // self.patch_size
        patches = byte_ids.view(b, n_patches, self.patch_size)
        patch_byte_mask = byte_mask.view(b, n_patches, self.patch_size)
        patch_mask = patch_byte_mask.any(dim=-1)
        return patches, patch_byte_mask, patch_mask


class LocalEncoder(nn.Module):
    """Byte embedding + small non-causal transformer within each patch,
    pooled (via a learned attention query) into a single patch vector."""

    def __init__(self, d_model, patch_size, n_local_layers=1, n_heads=4, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 2
        self.byte_embed = nn.Embedding(256, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, patch_size, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, n_heads, d_ff, dropout, LayerNorm, "mha", None) for _ in range(n_local_layers)]
        )
        self.final_norm = LayerNorm(d_model)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)

    def forward(self, patches: torch.Tensor, patch_byte_mask: torch.Tensor) -> torch.Tensor:
        b, p, k = patches.shape
        x = self.byte_embed(patches.view(b * p, k)) + self.pos_embed
        mask = patch_byte_mask.reshape(b * p, k).clone()
        mask[mask.sum(-1) == 0] = True  # avoid empty softmax rows on fully-padded patches
        attn_mask = mask[:, None, None, :]

        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.final_norm(x)

        query = self.pool_query.expand(b * p, -1, -1)
        pooled = self.pool_attn(query, x, x, mask=attn_mask)
        return pooled.view(b, p, -1)


class LocalDecoder(nn.Module):
    """Given one global context vector per patch, autoregressively predicts
    (teacher-forced during training) the patch_size bytes of that patch."""

    def __init__(self, d_model, patch_size, n_heads=4, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 2
        self.patch_size = patch_size
        self.byte_embed = nn.Embedding(256, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, patch_size, d_model) * 0.02)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.self_block = PreNormResidual(d_model, LayerNorm, dropout)
        self.cross_block = PreNormResidual(d_model, LayerNorm, dropout)
        self.ffn_block = PreNormResidual(d_model, LayerNorm, dropout)
        self.final_norm = LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 256)

    def forward(self, patch_context: torch.Tensor, local_input_bytes: torch.Tensor) -> torch.Tensor:
        """patch_context: (b, p, d_model). local_input_bytes: (b, p, k) bytes
        right-shifted-by-one within each patch (position 0 = BOS = 0)."""
        b, p, k = local_input_bytes.shape
        x = self.byte_embed(local_input_bytes.view(b * p, k)) + self.pos_embed
        causal = get_causal_mask(k, x.device)
        memory = patch_context.reshape(b * p, 1, -1)

        x = self.self_block(x, lambda h: self.self_attn(h, h, h, mask=causal))
        x = self.cross_block(x, lambda h: self.cross_attn(h, memory, memory, mask=None))
        x = self.ffn_block(x, self.ffn)
        x = self.final_norm(x)
        return self.out_proj(x).view(b, p, k, 256)


class BLTSeq2Seq(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 2048,
        local_layers: int = 1,
        local_heads: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patcher = Patcher(patch_size)

        self.src_local_encoder = LocalEncoder(d_model, patch_size, local_layers, local_heads, dropout=dropout)
        self.tgt_local_encoder = LocalEncoder(d_model, patch_size, local_layers, local_heads, dropout=dropout)
        self.local_decoder = LocalDecoder(d_model, patch_size, local_heads, dropout=dropout)
        self.start_patch_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.global_encoder = TransformerEncoder(
            nn.ModuleList([EncoderLayer(d_model, n_heads, n_heads, d_ff, dropout, LayerNorm, "mha", None) for _ in range(n_layers)]),
            LayerNorm(d_model),
        )
        self.global_decoder = TransformerDecoder(
            nn.ModuleList([DecoderLayer(d_model, n_heads, n_heads, d_ff, dropout, LayerNorm, "mha", None) for _ in range(n_layers)]),
            LayerNorm(d_model),
        )

    def encode(self, src_bytes, src_byte_mask):
        patches, patch_byte_mask, patch_mask = self.patcher.to_patches(src_bytes, src_byte_mask)
        x = self.pos_enc(self.src_local_encoder(patches, patch_byte_mask))
        memory = self.global_encoder(x, get_padding_mask(patch_mask))
        return memory, patch_mask

    def forward(self, src_bytes, src_byte_mask, tgt_bytes, tgt_byte_mask):
        memory, src_patch_mask = self.encode(src_bytes, src_byte_mask)

        tgt_patches, tgt_patch_byte_mask, tgt_patch_mask = self.patcher.to_patches(tgt_bytes, tgt_byte_mask)
        b, p, k = tgt_patches.shape

        tgt_patch_embed = self.tgt_local_encoder(tgt_patches, tgt_patch_byte_mask)
        start = self.start_patch_embed.expand(b, 1, -1)
        decoder_input = self.pos_enc(torch.cat([start, tgt_patch_embed[:, :-1]], dim=1))

        causal = get_causal_mask(p, decoder_input.device)
        self_mask = get_padding_mask(tgt_patch_mask) & causal
        cross_mask = get_padding_mask(src_patch_mask)
        patch_context = self.global_decoder(decoder_input, memory, self_mask, cross_mask)

        byte_bos = tgt_patches.new_zeros((b, p, 1))
        local_input_bytes = torch.cat([byte_bos, tgt_patches[:, :, :-1]], dim=2)
        logits = self.local_decoder(patch_context, local_input_bytes)
        return logits, tgt_patches, tgt_patch_byte_mask

    @torch.no_grad()
    def generate(self, src_bytes, src_byte_mask, max_patches: int):
        device = src_bytes.device
        b = src_bytes.size(0)
        k = self.patch_size
        memory, src_patch_mask = self.encode(src_bytes, src_byte_mask)

        generated = torch.zeros((b, 0, k), dtype=torch.long, device=device)
        for _ in range(max_patches):
            p = generated.size(1)
            start = self.start_patch_embed.expand(b, 1, -1)
            if p > 0:
                patch_embed = self.tgt_local_encoder(generated, torch.ones_like(generated, dtype=torch.bool))
                decoder_input = torch.cat([start, patch_embed], dim=1)
            else:
                decoder_input = start
            decoder_input = self.pos_enc(decoder_input)

            causal = get_causal_mask(decoder_input.size(1), device)
            tgt_mask = torch.ones(b, decoder_input.size(1), dtype=torch.bool, device=device)
            self_mask = get_padding_mask(tgt_mask) & causal
            cross_mask = get_padding_mask(src_patch_mask)
            dec_out = self.global_decoder(decoder_input, memory, self_mask, cross_mask)
            patch_context = dec_out[:, -1:, :]

            local_input = torch.zeros((b, 1, k), dtype=torch.long, device=device)
            output_patch = torch.zeros((b, k), dtype=torch.long, device=device)
            for j in range(k):
                logits = self.local_decoder(patch_context, local_input)
                next_byte = logits[:, 0, j, :].argmax(dim=-1)
                output_patch[:, j] = next_byte
                if j + 1 < k:
                    local_input[:, 0, j + 1] = next_byte
            generated = torch.cat([generated, output_patch.unsqueeze(1)], dim=1)

        return generated.view(b, -1)
