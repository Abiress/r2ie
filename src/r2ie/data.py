"""Data loading and character-level tokenization.

Supports:
  - "tinyshakespeare": downloaded via scripts/download_data.sh (full quickstart).
  - a bundled inline fixture (tests/fixtures/tiny_corpus.txt) used by tests so
    CI never needs network access.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
from torch.utils.data import Dataset


class CharTokenizer:
    """A minimal character-level tokenizer."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.chars = chars
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for i, c in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> List[int]:
        return [self.char2idx.get(c, 0) for c in text]

    def decode(self, ids) -> str:
        return "".join(self.idx2char.get(int(i), "") for i in ids)

    def to_tensor(self, text: str) -> torch.Tensor:
        return torch.tensor(self.encode(text), dtype=torch.long)


class CharDataset(Dataset):
    """Character-level language-modeling dataset.

    Produces (input, target) pairs where target is input shifted by one.
    Sequences are drawn contiguously from the corpus.
    """

    def __init__(self, text: str, seq_len: int, tokenizer: Optional[CharTokenizer] = None):
        self.tokenizer = tokenizer or CharTokenizer(text)
        self.seq_len = seq_len
        self.ids = self.tokenizer.to_tensor(text)
        # Number of non-overlapping-ish windows we can form.
        self.n = max(0, len(self.ids) - seq_len)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.ids[idx : idx + self.seq_len]
        y = self.ids[idx + 1 : idx + self.seq_len + 1]
        return x, y


def load_corpus_text(source: str, data_dir: str = "data") -> str:
    """Load corpus text from a named source.

    Args:
        source: "tinyshakespeare" or a path to a text file, or "fixture" for the
            bundled test corpus.
        data_dir: directory where tinyshakespeare.txt is expected (after
            running scripts/download_data.sh).
    """
    if source == "fixture":
        # This file lives at <repo>/src/r2ie/data.py; the fixture is at
        # <repo>/tests/fixtures/tiny_corpus.txt.
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        fixture = os.path.join(repo_root, "tests", "fixtures", "tiny_corpus.txt")
        with open(fixture, "r", encoding="utf-8") as f:
            return f.read()

    if os.path.isfile(source):
        with open(source, "r", encoding="utf-8") as f:
            return f.read()

    path = os.path.join(data_dir, "tinyshakespeare.txt")
    if source in ("tinyshakespeare", "shakespeare") or source == path:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Could not find {path}. Run `bash scripts/download_data.sh` first, "
                "or pass --data <path-to-txt>."
            )
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    raise ValueError(f"Unknown data source: {source!r}")


def make_dataloader(
    source: str,
    seq_len: int,
    batch_size: int,
    data_dir: str = "data",
    shuffle: bool = True,
):
    """Build a DataLoader for the given corpus source."""
    from torch.utils.data import DataLoader

    text = load_corpus_text(source, data_dir=data_dir)
    tokenizer = CharTokenizer(text)
    dataset = CharDataset(text, seq_len, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return loader, tokenizer
