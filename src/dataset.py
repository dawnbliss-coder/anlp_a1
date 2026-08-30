"""Loads the aligned brown_cipher.txt / brown_plain.txt pair and builds
DataLoaders for both tokenization strategies:

- "subword" (C1-C4): a from-scratch BPE tokenizer (bpe.py) is trained
  separately on the raw cipher bit-string and on the plaintext, so both
  sides use genuinely *learned* subword units rather than a fixed-width
  chunking rule.
- "blt" (C5): target = raw ASCII byte ids, no vocabulary needed; source =
  raw bytes chunked 8 bits at a time. Token-free by design, so fixed-width
  chunking is exactly what this config is meant to demonstrate.
"""

import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from bpe import BPETokenizer


def load_pairs(data_dir: str) -> Tuple[List[str], List[str]]:
    with open(os.path.join(data_dir, "brown_cipher.txt")) as f:
        cipher_lines = [line.rstrip("\n") for line in f]
    with open(os.path.join(data_dir, "brown_plain.txt")) as f:
        plain_lines = [line.rstrip("\n") for line in f]
    assert len(cipher_lines) == len(plain_lines), "cipher/plain line count mismatch"
    return cipher_lines, plain_lines


def bits_to_bytes(bit_string: str) -> List[int]:
    n = len(bit_string) - (len(bit_string) % 8)
    return [int(bit_string[i : i + 8], 2) for i in range(0, n, 8)]


def truncate_pair(bit_string: str, plain: str, max_chars: int) -> Tuple[str, str]:
    # 8 cipher bits per plaintext char on every line, so truncating both to
    # the same char count keeps them aligned.
    plain_t = plain[:max_chars]
    bits_t = bit_string[: len(plain_t) * 8]
    return bits_t, plain_t


def build_examples(data_dir: str, max_chars: int) -> List[Tuple[str, str]]:
    """Returns (bits_str, plain_str) pairs — raw, untokenized. What each
    Dataset class below does with the bits string depends on the config's
    tokenization strategy, so no conversion happens here."""
    cipher_lines, plain_lines = load_pairs(data_dir)
    examples = []
    for bits, plain in zip(cipher_lines, plain_lines):
        if not plain:
            continue
        bits_t, plain_t = truncate_pair(bits, plain, max_chars)
        if not bits_t:
            continue
        examples.append((bits_t, plain_t))
    return examples


def split_examples(examples, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42):
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    n = len(idx)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    pick = lambda ids: [examples[i] for i in ids]
    return pick(train_idx), pick(val_idx), pick(test_idx)


class SubwordCipherDataset(Dataset):
    """C1-C4: both sides BPE-tokenized. Encoding is done once here at
    construction time (not lazily per __getitem__ call) since applying BPE
    merges to a ~2000-symbol raw bit-string is too expensive to repeat on
    every epoch."""

    def __init__(self, examples, src_tokenizer: BPETokenizer, tgt_tokenizer: BPETokenizer):
        self.plain = [plain for _, plain in examples]
        self.src_ids = [src_tokenizer.encode(bits, add_bos_eos=False) for bits, _ in examples]
        self.tgt_ids = [tgt_tokenizer.encode(plain, add_bos_eos=True) for _, plain in examples]

    def __len__(self):
        return len(self.plain)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src_ids[idx], dtype=torch.long),
            torch.tensor(self.tgt_ids[idx], dtype=torch.long),
            self.plain[idx],
        )


class ByteCipherDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        bits, plain = self.examples[idx]
        src_bytes = bits_to_bytes(bits)
        tgt_bytes = list(plain.encode("ascii"))
        return torch.tensor(src_bytes, dtype=torch.long), torch.tensor(tgt_bytes, dtype=torch.long), plain


def pad_batch(seqs, pad_id: int):
    max_len = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
        mask[i, : s.size(0)] = True
    return out, mask


def make_collate_fn(src_pad_id: int, tgt_pad_id: int):
    def collate(batch):
        src_list, tgt_list, plain_list = zip(*batch)
        src, src_mask = pad_batch(src_list, pad_id=src_pad_id)
        tgt, tgt_mask = pad_batch(tgt_list, pad_id=tgt_pad_id)
        return {
            "src": src,
            "src_mask": src_mask,
            "tgt": tgt,
            "tgt_mask": tgt_mask,
            "plain": list(plain_list),
        }

    return collate


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    src_vocab_size: int
    tgt_vocab_size: int
    src_tokenizer: Optional[BPETokenizer]
    tokenizer: Optional[BPETokenizer]
    pad_id: int


def build_dataloaders(
    cfg,
    data_dir: str,
    batch_size: int = 32,
    max_chars: int = 256,
    vocab_size: int = 1000,
    bpe_train_sample: int = 800,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
) -> DataBundle:
    examples = build_examples(data_dir, max_chars)
    train_ex, val_ex, test_ex = split_examples(examples, val_frac, test_frac, seed)

    if cfg.tokenization == "subword":
        src_tokenizer = BPETokenizer(vocab_size=vocab_size)
        src_tokenizer.train([bits for bits, _ in train_ex], sample_size=bpe_train_sample, seed=seed)

        tgt_tokenizer = BPETokenizer(vocab_size=vocab_size)
        tgt_tokenizer.train([plain for _, plain in train_ex], sample_size=None, seed=seed)

        src_pad_id, tgt_pad_id = src_tokenizer.pad_id, tgt_tokenizer.pad_id
        train_ds = SubwordCipherDataset(train_ex, src_tokenizer, tgt_tokenizer)
        val_ds = SubwordCipherDataset(val_ex, src_tokenizer, tgt_tokenizer)
        test_ds = SubwordCipherDataset(test_ex, src_tokenizer, tgt_tokenizer)
        src_vocab_size = src_tokenizer.get_vocab_size()
        tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    else:
        src_tokenizer = None
        tgt_tokenizer = None
        src_pad_id, tgt_pad_id = 0, 0
        train_ds = ByteCipherDataset(train_ex)
        val_ds = ByteCipherDataset(val_ex)
        test_ds = ByteCipherDataset(test_ex)
        src_vocab_size = 256
        tgt_vocab_size = 256

    collate = make_collate_fn(src_pad_id=src_pad_id, tgt_pad_id=tgt_pad_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_tokenizer=src_tokenizer,
        tokenizer=tgt_tokenizer,
        pad_id=tgt_pad_id,
    )
