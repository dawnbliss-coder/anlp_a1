#!/usr/bin/env bash
# Trains C1-C5 sequentially with identical hyperparameters (as required by
# the assignment's controlled-ablation setup) and produces a cross-config
# comparison plot at the end. Run from anywhere; paths are resolved
# relative to this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WANDB_PROJECT="${WANDB_PROJECT:-anlp-a1}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"
MAX_CHARS="${MAX_CHARS:-256}"
VOCAB_SIZE="${VOCAB_SIZE:-1000}"

for CFG in C1 C2 C3 C4 C5; do
  echo "=== Training $CFG ==="
  python src/train.py \
    --config "$CFG" \
    --wandb_project "$WANDB_PROJECT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_chars "$MAX_CHARS" \
    --vocab_size "$VOCAB_SIZE"
done

python - <<'PY'
import json
import os
import sys

sys.path.insert(0, "src")
from utils import plot_config_comparison

with open("outputs/results.json") as f:
    results = json.load(f)

flat = {name: r["test"] | {"peak_gpu_mem_mb": r["peak_gpu_mem_mb"]} for name, r in results.items()}
os.makedirs("outputs/plots", exist_ok=True)
for metric in ["bit_level_accuracy", "sequence_accuracy", "levenshtein", "peak_gpu_mem_mb"]:
    if all(metric in v for v in flat.values()):
        plot_config_comparison(flat, metric, f"outputs/plots/compare_{metric}.png")
print("Wrote comparison plots to outputs/plots/")
PY

echo "All configs trained. See outputs/results.json and outputs/plots/ for the comparison."
