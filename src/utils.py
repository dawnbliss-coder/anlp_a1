"""Evaluation metrics (bit-level accuracy, sequence accuracy, Levenshtein
distance, BLEU/ROUGE) and plotting helpers for comparing C1-C5."""

from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bit_level_accuracy(pred: str, target: str) -> float:
    pred_bytes = pred.encode("ascii", errors="replace")
    target_bytes = target.encode("ascii", errors="replace")
    n = max(len(pred_bytes), len(target_bytes))
    if n == 0:
        return 1.0
    pred_bytes = pred_bytes.ljust(n, b"\x00")
    target_bytes = target_bytes.ljust(n, b"\x00")
    mismatched_bits = sum(bin(pb ^ tb).count("1") for pb, tb in zip(pred_bytes, target_bytes))
    return 1.0 - mismatched_bits / (n * 8)


def sequence_accuracy(preds: List[str], targets: List[str]) -> float:
    if not preds:
        return 0.0
    return sum(p == t for p, t in zip(preds, targets)) / len(preds)


def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def mean_levenshtein(preds: List[str], targets: List[str]) -> float:
    if not preds:
        return 0.0
    return sum(levenshtein(p, t) for p, t in zip(preds, targets)) / len(preds)


def mean_bit_level_accuracy(preds: List[str], targets: List[str]) -> float:
    if not preds:
        return 0.0
    return sum(bit_level_accuracy(p, t) for p, t in zip(preds, targets)) / len(preds)


def bleu_rouge(preds: List[str], targets: List[str]) -> Dict[str, float]:
    """Standard n-gram overlap metrics; assignment scopes these to tokenized
    (subword) configs only, so callers should skip this for BLT (C5)."""
    import sacrebleu
    from rouge_score import rouge_scorer

    bleu = sacrebleu.corpus_bleu(preds, [targets]).score
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1 = rl = 0.0
    for p, t in zip(preds, targets):
        scores = scorer.score(t, p)
        r1 += scores["rouge1"].fmeasure
        rl += scores["rougeL"].fmeasure
    n = max(len(preds), 1)
    return {"bleu": bleu, "rouge1": r1 / n, "rougeL": rl / n}


def compute_all_metrics(preds: List[str], targets: List[str], include_bleu_rouge: bool) -> Dict[str, float]:
    metrics = {
        "bit_level_accuracy": mean_bit_level_accuracy(preds, targets),
        "sequence_accuracy": sequence_accuracy(preds, targets),
        "levenshtein": mean_levenshtein(preds, targets),
    }
    if include_bleu_rouge:
        metrics.update(bleu_rouge(preds, targets))
    return metrics


def plot_loss_curve(steps, train_loss, val_steps, val_loss, out_path: str, title: str):
    plt.figure(figsize=(6, 4))
    plt.plot(steps, train_loss, label="train")
    if val_steps:
        plt.plot(val_steps, val_loss, label="val", marker="o")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_config_comparison(results: Dict[str, Dict[str, float]], metric: str, out_path: str, ylabel: str = None):
    names = list(results.keys())
    values = [results[n].get(metric, 0.0) for n in names]
    plt.figure(figsize=(6, 4))
    plt.bar(names, values, color="#4C72B0")
    plt.ylabel(ylabel or metric)
    plt.title(f"{metric} across configurations")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
