"""From-scratch byte-pair-encoding tokenizer — no external tokenizer library.

Learns merges the standard way (Sennrich et al. 2016 / GPT-2's tokenizer):
repeatedly combine the most frequent adjacent symbol pair in the training
corpus into a new symbol, until the vocabulary reaches vocab_size. Simpler
than a general-purpose implementation since our alphabets are small and
closed (bits '0'/'1' for the cipher, or letters+space for plaintext)
rather than arbitrary Unicode, so no byte-level escaping is needed.

Training tracks pair counts incrementally (only rescans the sequences that
actually contained the just-merged pair) rather than rescanning the whole
corpus every merge — the cipher side has sequences up to ~2000 symbols
before any merging, and a full-corpus rescan per merge is too slow for that
in pure Python.
"""

import heapq
import json
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

PAD, BOS, EOS, UNK = "[PAD]", "[BOS]", "[EOS]", "[UNK]"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.merge_rank: Dict[Tuple[str, str], int] = {}
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    def get_vocab_size(self) -> int:
        return len(self.id_to_token)

    def train(self, texts: List[str], sample_size: Optional[int] = None, seed: int = 42) -> None:
        """Learns merges from `texts`. If `sample_size` is set and the corpus
        is larger, trains on a random subsample instead of the full corpus —
        merge statistics converge well before seeing every example, and this
        keeps training tractable on the much longer cipher-bit sequences.
        Every example still gets encoded with the learned merges afterward;
        only the merge-learning step is subsampled."""
        if sample_size is not None and len(texts) > sample_size:
            texts = random.Random(seed).sample(texts, sample_size)

        corpus = [list(t) for t in texts]
        alphabet = sorted({ch for seq in corpus for ch in seq})
        vocab = list(SPECIAL_TOKENS) + alphabet

        pair_counts: Counter = Counter()
        pair_locs: Dict[Tuple[str, str], set] = defaultdict(set)
        for si, seq in enumerate(corpus):
            for a, b in zip(seq, seq[1:]):
                pair_counts[(a, b)] += 1
                pair_locs[(a, b)].add(si)

        merges = []
        while len(vocab) < self.vocab_size and pair_counts:
            best_pair, best_count = max(pair_counts.items(), key=lambda kv: kv[1])
            if best_count < 2:
                break
            merged = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab.append(merged)

            affected = pair_locs.pop(best_pair, set())
            del pair_counts[best_pair]

            for si in affected:
                seq = corpus[si]
                for a, b in zip(seq, seq[1:]):
                    key = (a, b)
                    if key in pair_counts:
                        pair_counts[key] -= 1
                        if pair_counts[key] <= 0:
                            del pair_counts[key]
                    pair_locs[key].discard(si)

                new_seq, i = [], 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]:
                        new_seq.append(merged)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                corpus[si] = new_seq

                for a, b in zip(new_seq, new_seq[1:]):
                    pair_counts[(a, b)] += 1
                    pair_locs[(a, b)].add(si)

        self.merges = merges
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}
        self.id_to_token = dict(enumerate(vocab))
        self.token_to_id = {t: i for i, t in self.id_to_token.items()}

    def _apply_merges(self, symbols: List[str]) -> List[str]:
        """O(n log n) via a doubly-linked list + heap of candidate merges,
        instead of rescanning the whole (shrinking) sequence after every
        merge — that O(n^2) approach is too slow for the ~2000-symbol
        cipher sequences before any merging has happened."""
        n = len(symbols)
        if n <= 1:
            return list(symbols)

        tokens = list(symbols)
        prev = [i - 1 for i in range(n)]
        nxt = [i + 1 if i + 1 < n else -1 for i in range(n)]
        alive = [True] * n
        heap = []
        counter = 0

        def push(i):
            nonlocal counter
            j = nxt[i]
            if j == -1:
                return
            pair = (tokens[i], tokens[j])
            rank = self.merge_rank.get(pair)
            if rank is not None:
                heapq.heappush(heap, (rank, counter, i, j, tokens[i], tokens[j]))
                counter += 1

        for i in range(n):
            push(i)

        while heap:
            rank, _, i, j, ti, tj = heapq.heappop(heap)
            if not alive[i] or not alive[j] or tokens[i] != ti or tokens[j] != tj:
                continue  # stale: one side already merged elsewhere since this was queued
            tokens[i] = ti + tj
            alive[j] = False
            nj = nxt[j]
            nxt[i] = nj
            if nj != -1:
                prev[nj] = i
            pi = prev[i]
            if pi != -1 and alive[pi]:
                push(pi)
            push(i)

        result = []
        i = 0
        while i != -1:
            if alive[i]:
                result.append(tokens[i])
            i = nxt[i]
        return result

    def encode(self, text: str, add_bos_eos: bool = True) -> List[int]:
        symbols = self._apply_merges(list(text))
        unk = self.token_to_id[UNK]
        ids = [self.token_to_id.get(s, unk) for s in symbols]
        if add_bos_eos:
            ids = [self.token_to_id[BOS]] + ids + [self.token_to_id[EOS]]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        specials = set(SPECIAL_TOKENS)
        toks = [self.id_to_token.get(i, "") for i in ids]
        if skip_special_tokens:
            toks = [t for t in toks if t not in specials]
        return "".join(toks)

    def to_dict(self) -> dict:
        return {"vocab_size": self.vocab_size, "merges": self.merges, "id_to_token": self.id_to_token}

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        tok = cls(d["vocab_size"])
        tok.merges = [tuple(m) for m in d["merges"]]
        tok.merge_rank = {pair: i for i, pair in enumerate(tok.merges)}
        tok.id_to_token = {int(k): v for k, v in d["id_to_token"].items()}
        tok.token_to_id = {v: k for k, v in tok.id_to_token.items()}
        return tok

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            return cls.from_dict(json.load(f))
