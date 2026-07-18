"""Architecture benchmark: R2IE vs Transformer / Mamba / SSA.

Trains each model for the SAME step budget on the SAME data with the SAME
optimizer/lr, then reports validation perplexity. Parameter counts are matched
approximately by sharing a ModelConfig.

Honesty rules (per project guardrails):
  * We NEVER claim "outperforms" unless a real pasted benchmark supports it.
  * This script reports observed validation perplexity only; it prints a
    neutral verdict (lower is better) and writes the raw numbers to
    BENCHMARKS.md with no superiority claim by default.

Usage:
    python -m r2ie.benchmark_archs --steps 400 --seq-len 64
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

# Allow running as a script: `python benchmark_archs.py` from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from r2ie.baselines import MambaBaseline, SSABaseline, TransformerBaseline  # noqa: E402
from r2ie.config import ModelConfig  # noqa: E402
from r2ie.data import CharDataset, CharTokenizer  # noqa: E402
from r2ie.engine import R2IEEngine  # noqa: E402

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "tests", "fixtures", "tiny_corpus.txt"
)


def _load_text() -> str:
    with open(os.path.abspath(FIXTURE), "r", encoding="utf-8") as f:
        return f.read()


def _param_count(model) -> int:
    if hasattr(model, "param_count"):
        return model.param_count()
    return sum(p.numel() for p in model.parameters())


def _build_r2ie(cfg, use_dtf, use_hdq):
    eng = R2IEEngine(
        cfg, use_condensation=True, use_hdq=use_hdq, use_dtf=use_dtf
    )
    return eng.model


def _train(model, loader, steps, device, lr=1e-3):
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
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ce_hist.append(ce.item())
    return sum(ce_hist) / len(ce_hist)


@torch.no_grad()
def _eval(model, loader, device):
    model.to(device).eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        )
        total += ce.item()
        n += y.numel()
    return math.exp(total / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    text = _load_text()
    tok = CharTokenizer(text)
    n = len(text)
    split = int(n * 0.9)
    train_text, val_text = text[:split], text[split:]
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=args.seq_len,
    )

    train_ds = CharDataset(train_text, args.seq_len, tok)
    val_ds = CharDataset(val_text, args.seq_len, tok)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    models = {
        "R2IE (base)": lambda: _build_r2ie(cfg, use_dtf=False, use_hdq=False),
        "R2IE + DTF + HDQ": lambda: _build_r2ie(cfg, use_dtf=True, use_hdq=True),
        "Transformer": lambda: TransformerBaseline(cfg),
        "Mamba": lambda: MambaBaseline(cfg),
        "SSA (linear-attn)": lambda: SSABaseline(cfg),
    }

    rows = []
    for name, factory in models.items():
        print(f"[bench] training {name} ...")
        torch.manual_seed(1234)
        model = factory()
        t0 = time.time()
        train_ce = _train(model, train_loader, args.steps, device)
        ppl = _eval(model, val_loader, device)
        dt = time.time() - t0
        rows.append(
            (name, _param_count(model), round(train_ce, 4), round(ppl, 4), round(dt, 1))
        )
        print(f"[bench]   {name}: params={rows[-1][1]} val_ppl={ppl:.4f} time={dt:.1f}s")

    best = min(rows, key=lambda r: r[3])
    lines = [
        "# Architecture Benchmark (honest, factual)",
        "",
        f"Trained {args.steps} steps on the bundled `tiny_corpus.txt` fixture "
        f"(char-level LM, 90/10 train/val split), seq_len={args.seq_len}, "
        f"batch={args.batch_size}, Adam lr=1e-3, device={device}.",
        "",
        "| Model | Params | Final train CE | Val perplexity | Train time (s) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, pc, ce, ppl, dt in rows:
        lines.append(f"| {name} | {pc:,} | {ce} | {ppl} | {dt} |")
    lines += [
        "",
        f"Lower perplexity is better. Best observed val perplexity: "
        f"**{best[0]}** ({best[3]}).",
        "",
        "> NOTE: This is a tiny fixture run for smoke-testing the comparison "
        "harness. It is NOT a claim that any architecture outperforms another "
        "on real workloads. A superiority claim requires a pasted real-world "
        "benchmark.",
        "",
    ]
    md = "\n".join(lines)
    out_path = args.out or os.path.join(os.path.dirname(__file__), "..", "..", "BENCHMARKS.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[bench] wrote {out_path}")
    print(md)


if __name__ == "__main__":
    main()
