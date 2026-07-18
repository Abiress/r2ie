"""Internal experiment sweep for improving R2IE before the standard gate.

Trains R2IE (+DTF+HDQ) variants on real WikiText-2 and reports validation
perplexity quickly, so we can iterate on architecture/training choices without
running the full 5-architecture benchmark each time.

Usage:
    python src/r2ie/_sweep.py --steps 1000 --variant all
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from r2ie.config import ModelConfig  # noqa: E402
from r2ie.engine import R2IEEngine  # noqa: E402
from r2ie.standard_benchmark import (  # noqa: E402
    DEFAULT_DATA_DIR,
    WordDataset,
    WordTokenizer,
    load_wikitext,
)


def build(tok_vocab, d_model, n_layers, n_heads, d_ff, codebook, use_dtf, use_hdq,
          dtf_c_max, dtf_d_ff, alpha, hdq_decay, hdq_clamp, use_dtf_v2=False,
          use_dtf_v3=False, dtf_beta_max=1.0, vq_mode="hard"):
    cfg = ModelConfig(
        vocab_size=tok_vocab,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=64,
        codebook_size=codebook,
        dtf_c_max=dtf_c_max,
        dtf_d_ff=dtf_d_ff,
        dtf_v2=use_dtf_v2,
        dtf_v3=use_dtf_v3,
        dtf_beta_max=dtf_beta_max,
        vq_mode=vq_mode,
        hdq_decay=hdq_decay,
        hdq_clamp=hdq_clamp,
    )
    eng = R2IEEngine(cfg, use_condensation=True, use_hdq=use_hdq, use_dtf=use_dtf,
                     use_dtf_v2=use_dtf_v2, use_dtf_v3=use_dtf_v3)
    # Tune the condensation accretion rate if requested via fast-weight lr path.
    if alpha is not None:
        eng.condensation.alpha = alpha
    return eng.model


def train_eval(model, train_loader, valid_loader, steps, device, lr, eval_batches=200):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    it = iter(train_loader)
    for s in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        lo, aux = R2IE_fwd(model, x)
        ce = nn.functional.cross_entropy(lo.reshape(-1, lo.size(-1)), y.reshape(-1))
        ce.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_loader):
            if i >= eval_batches:
                break
            x, y = x.to(device), y.to(device)
            lo, _ = R2IE_fwd(model, x)
            ce = nn.functional.cross_entropy(
                lo.reshape(-1, lo.size(-1)), y.reshape(-1), reduction="sum"
            )
            total += ce.item()
            n += y.numel()
    return math.exp(total / n)


def R2IE_fwd(model, x):
    # R2IEModel.forward returns (logits, aux)
    return model(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--vocab-cap", type=int, default=4000)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--codebook", type=int, default=1024)
    ap.add_argument("--eval-batches", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_text = load_wikitext("train", args.data_dir)
    valid_text = load_wikitext("valid", args.data_dir)
    tok = WordTokenizer(train_text, args.vocab_cap)
    train_loader = torch.utils.data.DataLoader(
        WordDataset(train_text, args.seq_len, tok),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        WordDataset(valid_text, args.seq_len, tok),
        batch_size=args.batch_size, shuffle=False, drop_last=False,
    )

    variants = {
        # Proven winner (current official config).
        "DTFv1 softVQ s3000 (ref)": dict(use_dtf=True, use_dtf_v3=False, use_hdq=True, dtf_c_max=0.5, dtf_d_ff=256, alpha=0.05, hdq_decay=0.95, hdq_clamp=5.0, cb=1024, lr=1e-3, steps=3000, vq_mode="soft"),
        # Stronger+smarter: coherence-driven global mixer (DTFv3) as PRIMARY mixer,
        # ACT reduced to 1 pass, soft VQ. Larger capacity.
        "DTFv3 softVQ s3000": dict(use_dtf=True, use_dtf_v3=True, use_hdq=True, dtf_c_max=0.1, dtf_d_ff=256, alpha=0.05, hdq_decay=0.95, hdq_clamp=5.0, cb=1024, lr=1e-3, steps=3000, vq_mode="soft"),
    }

    for name, kw in variants.items():
        torch.manual_seed(1234)
        cb = kw.pop("cb", args.codebook)
        lr = kw.pop("lr", args.lr)
        steps = kw.pop("steps", args.steps)
        model = build(tok.vocab_size, args.d_model, args.n_layers, args.n_heads,
                      args.d_ff, cb, **kw)
        t0 = time.time()
        vppl = train_eval(model, train_loader, valid_loader, steps, device, lr,
                          eval_batches=args.eval_batches)
        print(f"{name:28s} val_ppl={vppl:8.3f}  time={time.time()-t0:6.1f}s")


if __name__ == "__main__":
    main()
