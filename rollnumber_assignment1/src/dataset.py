"""Loads the aligned brown_cipher.txt / brown_plain.txt pair and builds
DataLoaders for both tokenization strategies:

- "subword" (C1-C4): target = BPE subword token ids (with BOS/EOS), trained
  on the train split only.
- "blt" (C5): target = raw ASCII byte ids, no vocabulary needed.

The source (cipher) side is always the bit string chunked into 8-bit bytes
(vocab 0-255), since the input is inherently binary in every config.

Dataset property exploited for truncation: line k's cipher string is exactly
8 * len(plain_line_k) bits (one byte per plaintext character), so truncating
both to the first `max_chars` characters/bytes keeps them aligned.
"""

import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]

# pure i/o function asserting that both files cipher and plain have same number of lines.
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
    plain_t = plain[:max_chars]
    bits_t = bit_string[: len(plain_t) * 8]
    return bits_t, plain_t


def build_examples(data_dir: str, max_chars: int) -> List[Tuple[List[int], str]]:
    cipher_lines, plain_lines = load_pairs(data_dir)
    examples = []
    for bits, plain in zip(cipher_lines, plain_lines):
        if not plain:
            continue
        bits_t, plain_t = truncate_pair(bits, plain, max_chars)
        src_bytes = bits_to_bytes(bits_t)
        if not src_bytes:
            continue
        examples.append((src_bytes, plain_t))
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


def train_subword_tokenizer(texts: List[str], vocab_size: int = 1000) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    bos_id, eos_id = tokenizer.token_to_id("[BOS]"), tokenizer.token_to_id("[EOS]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
    )
    return tokenizer


class SubwordCipherDataset(Dataset):
    def __init__(self, examples, tokenizer: Tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        src_bytes, plain = self.examples[idx]
        tgt_ids = self.tokenizer.encode(plain).ids
        return torch.tensor(src_bytes, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long), plain


class ByteCipherDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        src_bytes, plain = self.examples[idx]
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


def make_collate_fn(tgt_pad_id: int):
    def collate(batch):
        src_list, tgt_list, plain_list = zip(*batch)
        src, src_mask = pad_batch(src_list, pad_id=0)  # value unused, positions masked out
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
    tokenizer: Optional[Tokenizer]
    pad_id: int


def build_dataloaders(
    cfg,
    data_dir: str,
    batch_size: int = 32,
    max_chars: int = 256,
    vocab_size: int = 1000,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
) -> DataBundle:
    examples = build_examples(data_dir, max_chars)
    train_ex, val_ex, test_ex = split_examples(examples, val_frac, test_frac, seed)

    if cfg.tokenization == "subword":
        tokenizer = train_subword_tokenizer([p for _, p in train_ex], vocab_size=vocab_size)
        pad_id = tokenizer.token_to_id("[PAD]")
        train_ds = SubwordCipherDataset(train_ex, tokenizer)
        val_ds = SubwordCipherDataset(val_ex, tokenizer)
        test_ds = SubwordCipherDataset(test_ex, tokenizer)
        tgt_vocab_size = tokenizer.get_vocab_size()
    else:
        tokenizer = None
        pad_id = 0
        train_ds = ByteCipherDataset(train_ex)
        val_ds = ByteCipherDataset(val_ex)
        test_ds = ByteCipherDataset(test_ex)
        tgt_vocab_size = 256

    collate = make_collate_fn(tgt_pad_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        src_vocab_size=256,
        tgt_vocab_size=tgt_vocab_size,
        tokenizer=tokenizer,
        pad_id=pad_id,
    )
