"""The five architectural configurations from Table 1 of the assignment.
C2-C5 each flip exactly one axis relative to the base C1; every other field
is held fixed so the ablation isolates a single component's effect."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    pos_encoding: str = "sinusoidal"  # "sinusoidal" | "rope"
    attention: str = "mha"            # "mha" | "gqa"
    norm: str = "layernorm"           # "layernorm" | "rmsnorm"
    tokenization: str = "subword"     # "subword" | "blt"

    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int = 2  # only used when attention == "gqa"
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 320  # positional-encoding capacity; must exceed max sequence length

    # BLT-only (tokenization == "blt")
    patch_size: int = 4
    local_layers: int = 1
    local_heads: int = 4


CONFIGS = {
    "C1": ModelConfig(name="C1"),
    "C2": ModelConfig(name="C2", pos_encoding="rope"),
    "C3": ModelConfig(name="C3", attention="gqa"),
    "C4": ModelConfig(name="C4", norm="rmsnorm"),
    "C5": ModelConfig(name="C5", tokenization="blt"),
}


def get_config(name: str) -> ModelConfig:
    key = name.upper()
    if key not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Choose from {sorted(CONFIGS)}.")
    return CONFIGS[key]
