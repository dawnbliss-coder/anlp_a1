"""Normalization modules built from raw tensor ops (no nn.LayerNorm)."""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)


class PreNormResidual(nn.Module):
    """x + sublayer(norm(x)). norm_cls is injected so the C1<->C4 ablation
    (LayerNorm vs RMSNorm) only changes a config flag, not layer wiring."""

    def __init__(self, d_model: int, norm_cls, dropout: float = 0.0):
        super().__init__()
        self.norm = norm_cls(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer, **kwargs) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x), **kwargs))
