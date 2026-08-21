"""Main training loop: trains one configuration (C1-C5) end to end, logs to
WandB, evaluates with greedy decoding, saves checkpoints/plots, and
optionally pushes the checkpoint to the Hugging Face Hub.

Usage:
    python train.py --config C1 --wandb_project anlp-a1 --epochs 10
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from configs import CONFIGS, get_config
from dataset import build_dataloaders
from models.blt import BLTSeq2Seq
from models.transformer import Seq2SeqTransformer
from utils import compute_all_metrics, plot_loss_curve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_chars", type=int, default=256, help="truncate each example to this many plaintext chars")
    p.add_argument("--vocab_size", type=int, default=1000, help="subword tokenizer vocab size (C1-C4 only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=20, help="steps between WandB train-loss logs")
    p.add_argument("--eval_every", type=int, default=2, help="epochs between full greedy-decode val evaluations")
    p.add_argument("--wandb_project", default=None, help="omit to disable WandB logging")
    p.add_argument("--run_suffix", default=None)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hf_repo_id", default=None, help="e.g. username/anlp-a1-checkpoints")
    return p.parse_args()


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg, src_vocab_size, tgt_vocab_size, tgt_pad_id, device):
    if cfg.tokenization == "blt":
        model = BLTSeq2Seq(
            patch_size=cfg.patch_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
            local_layers=cfg.local_layers,
            local_heads=cfg.local_heads,
        )
    else:
        model = Seq2SeqTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            tgt_pad_idx=tgt_pad_id,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
            pos_encoding=cfg.pos_encoding,
            attention=cfg.attention,
            norm=cfg.norm,
        )
    return model.to(device)


def compute_loss(model, batch, cfg, device, pad_id):
    src = batch["src"].to(device)
    src_mask = batch["src_mask"].to(device)
    tgt = batch["tgt"].to(device)
    tgt_mask = batch["tgt_mask"].to(device)

    if cfg.tokenization == "blt":
        logits, tgt_patches, _ = model(src, src_mask, tgt, tgt_mask)
        return F.cross_entropy(logits.reshape(-1, 256), tgt_patches.reshape(-1), ignore_index=0)

    decoder_input, decoder_target = tgt[:, :-1], tgt[:, 1:]
    dec_mask_input = tgt_mask[:, :-1]
    logits = model(src, decoder_input, src_mask, dec_mask_input)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1), ignore_index=pad_id)


@torch.no_grad()
def greedy_decode_subword(model, src, src_mask, tokenizer, max_len, device):
    model.eval()
    bos_id, eos_id = tokenizer.token_to_id("[BOS]"), tokenizer.token_to_id("[EOS]")
    b = src.size(0)
    memory = model.encode(src, src_mask)
    ys = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(b, dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        tgt_mask = torch.ones_like(ys, dtype=torch.bool)
        logits = model.decode(ys, memory, tgt_mask, src_mask)
        next_tok = logits[:, -1, :].argmax(-1)
        next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        finished = finished | (next_tok == eos_id)
        if finished.all():
            break
    texts = []
    for row in ys.tolist():
        if eos_id in row[1:]:
            row = row[: row.index(eos_id, 1) + 1]
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


@torch.no_grad()
def greedy_decode_blt(model, src, src_mask, device):
    model.eval()
    # Dataset property: target char count == source byte count exactly, so
    # this bounds generation length without needing an explicit EOS byte.
    lengths = src_mask.sum(dim=1)
    max_patches = int((lengths.max().item() + model.patch_size - 1) // model.patch_size)
    generated = model.generate(src, src_mask, max_patches=max_patches)
    texts = []
    for row, length in zip(generated.tolist(), lengths.tolist()):
        texts.append(bytes(row[:length]).decode("ascii", errors="replace"))
    return texts


@torch.no_grad()
def evaluate(model, loader, cfg, device, tokenizer, max_len, include_bleu_rouge):
    all_preds, all_targets = [], []
    for batch in loader:
        src, src_mask = batch["src"].to(device), batch["src_mask"].to(device)
        if cfg.tokenization == "blt":
            preds = greedy_decode_blt(model, src, src_mask, device)
        else:
            preds = greedy_decode_subword(model, src, src_mask, tokenizer, max_len, device)
        all_preds.extend(preds)
        all_targets.extend(batch["plain"])
    metrics = compute_all_metrics(all_preds, all_targets, include_bleu_rouge=include_bleu_rouge)
    return metrics, all_preds, all_targets


def push_checkpoint_to_hub(ckpt_dir, repo_id, subfolder):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(folder_path=ckpt_dir, repo_id=repo_id, path_in_repo=subfolder)
    print(f"Pushed {ckpt_dir} -> https://huggingface.co/{repo_id}/tree/main/{subfolder}")


def main():
    args = parse_args()
    cfg = get_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    data = build_dataloaders(
        cfg, args.data_dir, batch_size=args.batch_size, max_chars=args.max_chars,
        vocab_size=args.vocab_size, seed=args.seed,
    )

    model = build_model(cfg, data.src_vocab_size, data.tgt_vocab_size, data.pad_id, device)
    print(f"[{cfg.name}] {count_params(model):,} trainable params, device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    total_steps = max(1, args.epochs * len(data.train_loader))
    warmup_steps = max(1, int(0.06 * total_steps))

    def lr_lambda(step):
        step = max(step, 1)
        return min(step / warmup_steps, (warmup_steps / step) ** 0.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    run = None
    if args.wandb_project:
        import wandb

        run_name = f"{cfg.name}-{args.run_suffix}" if args.run_suffix else cfg.name
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **cfg.__dict__,
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "max_chars": args.max_chars, "vocab_size": args.vocab_size, "seed": args.seed,
            },
        )

    ckpt_dir = os.path.join(args.output_dir, "checkpoints", cfg.name)
    plot_dir = os.path.join(args.output_dir, "plots", cfg.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    include_bleu_rouge = cfg.tokenization == "subword"
    max_len_gen = args.max_chars + 20

    global_step = 0
    steps_hist, train_loss_hist = [], []
    val_steps_hist, val_loss_hist = [], []
    best_val_loss = float("inf")
    peak_mem_mb = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in data.train_loader:
            optimizer.zero_grad()
            loss = compute_loss(model, batch, cfg, device, data.pad_id)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            steps_hist.append(global_step)
            train_loss_hist.append(loss.item())
            if run and global_step % args.log_every == 0:
                wandb.log(
                    {"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0], "train/grad_norm": float(grad_norm)},
                    step=global_step,
                )

        avg_train_loss = epoch_loss / max(len(data.train_loader), 1)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in data.val_loader:
                val_loss_total += compute_loss(model, batch, cfg, device, data.pad_id).item()
        avg_val_loss = val_loss_total / max(len(data.val_loader), 1)
        val_steps_hist.append(global_step)
        val_loss_hist.append(avg_val_loss)

        if device.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        log_dict = {"val/loss": avg_val_loss, "val/peak_gpu_mem_mb": peak_mem_mb}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics, _, _ = evaluate(model, data.val_loader, cfg, device, data.tokenizer, max_len_gen, include_bleu_rouge)
            log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
            print(
                f"[{cfg.name}] epoch {epoch}/{args.epochs} train_loss={avg_train_loss:.4f} "
                f"val_loss={avg_val_loss:.4f} val_seq_acc={val_metrics['sequence_accuracy']:.4f} "
                f"peak_mem_mb={peak_mem_mb:.1f}"
            )
        else:
            print(f"[{cfg.name}] epoch {epoch}/{args.epochs} train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

        if run:
            wandb.log(log_dict, step=global_step)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "final.pt"))
    if data.tokenizer is not None:
        data.tokenizer.save(os.path.join(ckpt_dir, "tokenizer.json"))

    plot_loss_curve(
        steps_hist, train_loss_hist, val_steps_hist, val_loss_hist,
        os.path.join(plot_dir, "loss_curve.png"), title=f"{cfg.name} loss",
    )

    print(f"[{cfg.name}] loading best checkpoint for final test-set evaluation...")
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device))
    test_metrics, test_preds, test_targets = evaluate(
        model, data.test_loader, cfg, device, data.tokenizer, max_len_gen, include_bleu_rouge
    )
    print(f"[{cfg.name}] test metrics: {test_metrics}")
    if run:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

    results_path = os.path.join(args.output_dir, "results.json")
    all_results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    all_results[cfg.name] = {"test": test_metrics, "peak_gpu_mem_mb": peak_mem_mb, "params": count_params(model)}
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    with open(os.path.join(args.output_dir, f"{cfg.name}_test_predictions.txt"), "w") as f:
        for p, t in zip(test_preds[:50], test_targets[:50]):
            f.write(f"TARGET: {t}\nPRED:   {p}\n\n")

    if args.push_to_hub:
        if not args.hf_repo_id:
            raise ValueError("--push_to_hub requires --hf_repo_id")
        push_checkpoint_to_hub(ckpt_dir, args.hf_repo_id, cfg.name)

    if run:
        wandb.finish()


if __name__ == "__main__":
    main()
