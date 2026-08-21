# ANLP Assignment 1 — Transformers from Scratch, Architectural Variants, and BLT

Rename this whole folder from `rollnumber_assignment1` to `<your-roll-number>_assignment1` before zipping for submission.

## What's here

A from-scratch (no `nn.Transformer` / `nn.MultiheadAttention`) Encoder-Decoder Transformer, plus a simplified Byte Latent Transformer, trained on a binary-cipher-to-plaintext task, run across the 5 ablation configs in `src/configs.py` (C1-C5, matching Table 1 of the assignment PDF).

```
src/
  models/
    attention.py   # scaled dot-product attention, MultiHeadAttention, GroupedQueryAttention
    positional.py  # Sinusoidal absolute PE, RoPE
    norm.py        # LayerNorm, RMSNorm, PreNorm residual wrapper
    ffn.py          # position-wise FFN
    transformer.py # EncoderLayer/DecoderLayer/Seq2SeqTransformer (drives C1-C4)
    blt.py          # Patcher, LocalEncoder, LocalDecoder, BLTSeq2Seq (drives C5)
  configs.py        # ModelConfig dataclass + CONFIGS = {C1..C5}
  dataset.py        # loads data/, byte-chunks cipher, trains BPE tokenizer, DataLoaders
  train.py          # training loop, WandB logging, checkpointing, HF Hub push
  utils.py          # metrics (bit-level acc, seq acc, Levenshtein, BLEU/ROUGE) + plots
outputs/            # results.json, per-config loss curves, comparison plots, sample predictions
run_all.sh          # trains C1-C5 back to back with identical hyperparameters
```

## Dataset

`data/brown_cipher.txt` / `data/brown_plain.txt`: 5,000 line-aligned pairs. Each plaintext line uses only `A-Z a-z` and spaces; its cipher line is an exact 8-bits-per-character binary string (so `len(cipher_bits) == 8 * len(plain_chars)` on every line — verified across the dataset). We don't need to know the underlying cipher algorithm; the model learns the byte-substitution mapping end to end from these pairs. `dataset.py` chunks the cipher bits into bytes (vocab 0-255) as the source sequence, and either BPE-subword-tokenizes (C1-C4) or byte-encodes (C5) the plaintext as the target.

Each example is truncated to the first `--max_chars` plaintext characters (default 256) — and correspondingly the first `8*max_chars` cipher bits — to keep sequence lengths (and O(n²) attention cost) tractable on a free-tier Colab GPU. All 5,000 lines are kept; only the tail of long lines is dropped. Increase `--max_chars` if you have more compute and want to demonstrate handling of longer sequences.

## Setup (Colab)

```bash
!pip install -q -r requirements.txt
!wandb login   # paste your API key
```

Upload/mount this folder (or clone it) so `data/` sits next to `src/` as shown above.

## Training one config

```bash
python src/train.py --config C1 --wandb_project anlp-a1 --epochs 10
```

Repeat for `C2`..`C5` with the **same** `--epochs/--batch_size/--lr/--max_chars/--vocab_size`, or just run:

```bash
WANDB_PROJECT=anlp-a1 EPOCHS=10 ./run_all.sh
```

This writes, per config, under `outputs/`:
- `checkpoints/<CFG>/best.pt`, `final.pt` (+ `tokenizer.json` for C1-C4)
- `plots/<CFG>/loss_curve.png`
- `<CFG>_test_predictions.txt` (50 sample greedy-decoded predictions vs targets)
- `results.json` (test metrics, peak GPU memory, param count — merged across all configs you've run)

After all 5 finish, `run_all.sh` also writes `outputs/plots/compare_*.png` bar charts.

## Pushing checkpoints to Hugging Face

```bash
huggingface-cli login
python src/train.py --config C1 --push_to_hub --hf_repo_id <your-username>/anlp-a1-checkpoints ...
```

## Links to fill in before submitting

- WandB project: `<paste link here>`
- Hugging Face checkpoints repo: `<paste link here>`

## Design notes for the report

- **Source tokenization is fixed across all 5 configs**: cipher bits chunked into bytes (vocab 256), since the input is inherently binary. Only the *target*-side strategy changes for C5 (BLT/byte-free) vs C1-C4 (BPE subwords).
- **RoPE (C2)** is applied only inside self-attention (encoder self-attn, decoder self-attn); cross-attention never receives rotary embeddings, since K/V there come from the encoder's domain, not a shared relative-position axis with the decoder queries.
- **GQA (C3)** uses `n_kv_heads` (see `configs.py`) shared key/value heads with `repeat_interleave` before attention — set `n_heads % n_kv_heads == 0`.
- **BLT (C5)** uses *fixed-size* patches (`patch_size` in `configs.py`), not the entropy-based dynamic patching of the original BLT paper — this is the "simplified" version the assignment asks for. Pay attention to `outputs/results.json`'s `peak_gpu_mem_mb` and epoch wall-clock time to discuss the token-free tradeoffs the report asks about.
- `utils.bleu_rouge` is only computed for C1-C4 (`include_bleu_rouge = cfg.tokenization == "subword"` in `train.py`), matching the spec ("BLEU and ROUGE ... for tokenized models only").
