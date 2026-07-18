"""Standard, real-world benchmark gate for R2IE.

This is the OFFICIAL pre-release benchmark. It evaluates R2IE and the baseline
architectures (Transformer, Mamba, SSA) on the canonical **WikiText-2** language
modeling corpus (the same dataset EleutherAI's lm-evaluation-harness uses for
its perplexity task). The metric is **test-set token perplexity** (lower is
better) - the standard, comparable LM benchmark.

Honesty / "brutal test" rules (project guardrails):
  * Real data only. No synthetic/fixture data for this gate.
  * Identical protocol for every architecture: same seed, tokenizer, vocab
    size cap, sequence length, batch size, learning rate, and step budget.
  * The verdict is derived from the measured numbers. We claim R2IE
    "outperforms" ONLY when it has the lowest test perplexity among all
    architectures under the fixed protocol - and even then the raw numbers are
    reported so anyone can verify.
  * If R2IE does NOT win, the report says so plainly. No fabrication, ever.

Usage:
    python -m r2ie.standard_benchmark --steps 2000 --seq-len 64
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from r2ie.baselines import MambaBaseline, SSABaseline, TransformerBaseline  # noqa: E402
from r2ie.config import ModelConfig  # noqa: E402
from r2ie.engine import R2IEEngine  # noqa: E402

# Default location of the real-world WikiText-2 raw corpus. Override with
# --data-dir. This is the canonical wikitext-2-raw-v1 split used by the
# lm-evaluation-harness wikitext task.
DEFAULT_DATA_DIR = (
    "/home/abir/Desktop/bi/_archive/llm_workspace/datasets/raw/"
    "wikitext2_extracted/wikitext-2-raw"
)

WIKITEXT_FILES = {
    "train": "wiki.train.raw",
    "valid": "wiki.valid.raw",
    "test": "wiki.test.raw",
}


def load_wikitext(split: str, data_dir: str) -> str:
    path = os.path.join(data_dir, WIKITEXT_FILES[split])
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"WikiText-2 {split} not found at {path}. "
            "Pass --data-dir pointing at the wikitext-2-raw-v1 extracted folder."
        )
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


class WordTokenizer:
    """A shared word-level tokenizer (standard for WikiText-2 perplexity).

    Built once from the training split so every architecture is compared on the
    SAME tokenization. Unknown words map to <unk>; we cap the vocab at
    ``vocab_cap`` most-frequent words to keep parameter counts manageable and
    comparable across architectures.
    """

    def __init__(self, text: str, vocab_cap: int = 8000):
        from collections import Counter

        words = text.split()
        freq = Counter(words)
        # Keep the most frequent words; everything else -> <unk>.
        kept = [w for w, _ in freq.most_common(vocab_cap)]
        self.word2idx = {"<unk>": 0, "<pad>": 1}
        for w in kept:
            if w not in self.word2idx:
                self.word2idx[w] = len(self.word2idx)
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode(self, text: str) -> list[int]:
        return [self.word2idx.get(w, 0) for w in text.split()]

    def to_tensor(self, text: str) -> torch.Tensor:
        return torch.tensor(self.encode(text), dtype=torch.long)


class WordDataset(torch.utils.data.Dataset):
    """Word-level LM dataset: (input, target=shifted) windows."""

    def __init__(self, text: str, seq_len: int, tok: WordTokenizer):
        self.tok = tok
        self.seq_len = seq_len
        self.ids = tok.to_tensor(text)
        self.n = max(0, len(self.ids) - seq_len)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.ids[idx : idx + self.seq_len]
        y = self.ids[idx + 1 : idx + self.seq_len + 1]
        return x, y


def _build_r2ie(cfg: ModelConfig, use_dtf: bool, use_hdq: bool, codebook_size: int = 128,
                use_dtf_v2: bool = False, use_dtf_v3: bool = False,
                vq_mode: str = "hard") -> nn.Module:
    cfg = ModelConfig(**{**vars(cfg), "codebook_size": codebook_size,
                         "dtf_v2": use_dtf_v2, "dtf_v3": use_dtf_v3, "vq_mode": vq_mode})
    eng = R2IEEngine(cfg, use_condensation=True, use_hdq=use_hdq, use_dtf=use_dtf,
                     use_dtf_v2=use_dtf_v2, use_dtf_v3=use_dtf_v3)
    return eng.model


def _param_count(model) -> int:
    if hasattr(model, "param_count"):
        return model.param_count()
    return sum(p.numel() for p in model.parameters())


def _train(model, loader, steps, device, lr, clip=1.0):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    it = iter(loader)
    ce_hist = []
    for step in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        ce = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        ce.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        ce_hist.append(ce.item())
    return sum(ce_hist) / len(ce_hist)


@torch.no_grad()
def _eval_perplexity(model, loader, device, max_batches: int = 500):
    """Perplexity over (up to) the first ``max_batches`` batches of the loader.

    We cap the number of batches so evaluation is bounded and reproducible;
    500 word-level batches (~16k tokens) is a stable perplexity estimate and
    matches the spirit of the harness, which also reports on a fixed test set.
    """
    model.to(device).eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        ce = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        )
        total += ce.item()
        n += y.numel()
    return math.exp(total / n)


def run_benchmark(args) -> str:
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    # Deterministic protocol.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_text = load_wikitext("train", args.data_dir)
    valid_text = load_wikitext("valid", args.data_dir)
    test_text = load_wikitext("test", args.data_dir)

    tok = WordTokenizer(train_text, vocab_cap=args.vocab_cap)
    print(f"[std-bench] vocab={tok.vocab_size} device={device}")

    train_ds = WordDataset(train_text, args.seq_len, tok)
    valid_ds = WordDataset(valid_text, args.seq_len, tok)
    test_ds = WordDataset(test_text, args.seq_len, tok)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
    )

    models = {
        "R2IE (base)": lambda: _build_r2ie(cfg, use_dtf=False, use_hdq=False, codebook_size=args.codebook_size, vq_mode=args.vq_mode),
        "R2IE + DTF + HDQ": lambda: _build_r2ie(cfg, use_dtf=True, use_hdq=True, codebook_size=args.codebook_size, vq_mode=args.vq_mode),
        "R2IE + DTFv2 + HDQ": lambda: _build_r2ie(cfg, use_dtf=True, use_hdq=True, codebook_size=args.codebook_size, use_dtf_v2=True, vq_mode=args.vq_mode),
        "R2IE + DTFv3 + HDQ": lambda: _build_r2ie(cfg, use_dtf=True, use_hdq=True, codebook_size=args.codebook_size, use_dtf_v3=True, vq_mode=args.vq_mode),
        "Transformer": lambda: TransformerBaseline(cfg),
        "Mamba": lambda: MambaBaseline(cfg),
        "SSA (linear-attn)": lambda: SSABaseline(cfg),
    }

    rows = []
    for name, factory in models.items():
        print(f"[std-bench] training {name} ...")
        torch.manual_seed(args.seed)
        model = factory()
        t0 = time.time()
        train_ce = _train(model, train_loader, args.steps, device, args.lr)
        val_ppl = _eval_perplexity(model, valid_loader, device, args.eval_batches)
        test_ppl = _eval_perplexity(model, test_loader, device, args.eval_batches)
        dt = time.time() - t0
        rows.append(
            (
                name,
                _param_count(model),
                round(train_ce, 4),
                round(val_ppl, 4),
                round(test_ppl, 4),
                round(dt, 1),
            )
        )
        print(
            f"[std-bench]   {name}: params={rows[-1][1]:,} "
            f"val_ppl={val_ppl:.4f} test_ppl={test_ppl:.4f} time={dt:.1f}s"
        )

    # Verdict: lowest TEST perplexity wins (standard LM metric).
    best = min(rows, key=lambda r: r[4])
    r2ie_best = min(
        r for r in rows if r[0].startswith("R2IE")
    )
    r2ie_wins = r2ie_best[4] == best[4]

    lines = [
        "# Standard Real-World Benchmark (WikiText-2)",
        "",
        "> This is the official pre-release benchmark gate. It uses the canonical "
        "**WikiText-2** corpus (the same data as EleutherAI's lm-evaluation-harness "
        "wikitext perplexity task). All architectures share one tokenizer, vocab "
        f"cap ({args.vocab_cap}), and a fixed protocol: seed={args.seed}, "
        f"steps={args.steps}, seq_len={args.seq_len}, batch={args.batch_size}, "
        f"lr={args.lr}, d_model={args.d_model}, n_layers={args.n_layers}, "
        f"vq_mode={args.vq_mode}, eval_batches={args.eval_batches}, device={device}.",
        "",
        "**Metric: test-set token perplexity (lower is better).**",
        "",
        "| Model | Params | Final train CE | Val ppl | Test ppl | Train time (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, pc, ce, vp, tp, dt in rows:
        lines.append(f"| {name} | {pc:,} | {ce} | {vp} | {tp} | {dt} |")

    lines += [
        "",
        f"**Best test perplexity: {best[0]} ({best[4]}).**",
        "",
    ]
    if r2ie_wins:
        lines.append(
            f"> **VERDICT:** Under this fixed, real-world protocol, the best R2IE "
            f"variant ({r2ie_best[0]}, test ppl {r2ie_best[4]}) achieves the lowest "
            f"test perplexity among all compared architectures. This is a measured "
            f"result on WikiText-2, not a claim extrapolated from other tasks."
        )
    else:
        lines.append(
            f"> **VERDICT:** Under this fixed, real-world protocol, R2IE did NOT "
            f"achieve the lowest test perplexity. Best is {best[0]} ({best[4]}); "
            f"best R2IE variant is {r2ie_best[0]} ({r2ie_best[4]}). No superiority "
            f"claim is made. Report the raw numbers above and investigate before "
            f"any 'outperforms' statement."
        )
    lines.append("")

    md = "\n".join(lines)
    out_path = args.out or os.path.join(
        os.path.dirname(__file__), "..", "..", "BENCHMARKS.md"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[std-bench] wrote {out_path}")
    print(md)
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description="Standard WikiText-2 benchmark gate for R2IE.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="Directory containing wiki.{train,valid,test}.raw")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--vocab-cap", type=int, default=8000)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-batches", type=int, default=500,
                    help="Max batches used for val/test perplexity (bounded, reproducible).")
    ap.add_argument("--codebook-size", type=int, default=128,
                    help="VQ codebook size for R2IE variants (compression capacity).")
    ap.add_argument("--vq-mode", type=str, default="soft",
                    choices=["hard", "soft"],
                    help="VQ compressor mode: 'hard' (true bottleneck) or 'soft' "
                         "(info-preserving softmax-weighted codebook average).")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
