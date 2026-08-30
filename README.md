# Transformers from Scratch + BLT

Encoder-decoder Transformer built from raw PyTorch (no `nn.Transformer`, no `nn.MultiheadAttention`) and a from-scratch BPE tokenizer (no external tokenizer library), trained to map a binary cipher to plaintext across 5 architectural ablations.

## Configs

| Config | Change from C1 |
|---|---|
| C1 | base — sinusoidal PE, multi-head attention, LayerNorm, BPE subwords |
| C2 | RoPE instead of sinusoidal |
| C3 | Grouped-query attention instead of multi-head |
| C4 | RMSNorm instead of LayerNorm |
| C5 | Byte Latent Transformer (token-free) instead of subwords |

## Structure

```
src/
  bpe.py            from-scratch BPE tokenizer
  configs.py        the 5 configs
  dataset.py        data loading + tokenization
  train.py          training loop, WandB logging, HF Hub push
  utils.py          metrics + plots
  models/
    attention.py    scaled dot-product attention, MHA, GQA
    positional.py   sinusoidal PE, RoPE
    norm.py         LayerNorm, RMSNorm
    ffn.py          feed-forward network
    transformer.py  encoder-decoder assembly (drives C1-C4)
    blt.py          patching + local encoder/decoder (drives C5)
outputs/            results.json, plots, sample predictions
requirements.txt
run_all.sh          trains C1-C5 back to back
```

## Setup

```bash
pip install -r requirements.txt
```

Needs `data/brown_cipher.txt` and `data/brown_plain.txt` in a `data/` folder next to this one (not included here — not mine to redistribute).

## Run

```bash
python src/train.py --config C1 --wandb_project <project> --epochs 12
```

or all five with identical hyperparameters:

```bash
./run_all.sh
```

## Results

Test set, 12 epochs, identical hyperparameters across all 5 configs:

| Config | Bit-level acc. | Levenshtein | Params |
|---|---|---|---|
| C1 | 59.9% | 187.0 | 8.1M |
| C2 | 63.7% | 275.3 | 8.1M |
| C3 | 63.0% | 198.8 | 7.0M |
| C4 | 58.5% | 191.5 | 8.1M |
| C5 | 91.7% | 59.1 | 10.0M |

Full metrics in `outputs/results.json`.

- WandB: https://wandb.ai/priyanka42875-international-institute-of-information-tec/anlp-a1
- Checkpoints: https://huggingface.co/priyanka42875/anlp-a1-checkpoints

---

Course assignment (Advanced NLP). Mirrored here for version control only.
