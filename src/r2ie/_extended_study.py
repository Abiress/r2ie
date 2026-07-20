"""Extended, fair benchmark study for R2IE vs baselines.

Runs each architecture (R2IE+DTFv3+HDQ, Transformer, Mamba, SSA) across a small
LR sweep {5e-4, 1e-3, 2e-3} and an extended training horizon (8000 steps),
recording test perplexity at every 1000-step checkpoint. Results are written
incrementally to a JSON file so the study is resume-safe.

This script is intentionally independent of standard_benchmark.py's verdict
logic; it only collects the trajectory data.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from torch.utils.data import DataLoader  # noqa: E402

from r2ie.baselines import MambaBaseline, SSABaseline, TransformerBaseline  # noqa: E402
from r2ie.standard_benchmark import (  # noqa: E402
    DEFAULT_DATA_DIR,
    ModelConfig,
    WordDataset,
    WordTokenizer,
    _build_r2ie,
    _eval_perplexity,
    load_wikitext,
)


def build_model(name, cfg):
    if name == "R2IE+DTFv3+HDQ":
        return _build_r2ie(cfg, use_dtf=True, use_hdq=True, use_dtf_v3=True,
                           codebook_size=1024, vq_mode="soft")
    if name == "Transformer":
        return TransformerBaseline(cfg)
    if name == "Mamba":
        return MambaBaseline(cfg)
    if name == "SSA":
        return SSABaseline(cfg)
    raise ValueError(name)


def train_with_checkpoints(model, loader, steps, device, lr, weight_decay,
                           test_loader, checkpoint_every, out_key, results):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    it = iter(loader)
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
        if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], dict):
            commit = out[1].get("commitment_loss", None)
            if commit is not None and torch.is_tensor(commit) and commit.requires_grad:
                (ce + 0.25 * commit).backward()
            else:
                ce.backward()
        else:
            ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % checkpoint_every == 0:
            ppl = _eval_perplexity(model, test_loader, device, 300)
            results.setdefault(out_key, {})[step] = round(float(ppl), 4)
            print(f"  [{out_key}] step {step}: test_ppl={ppl:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--lrs", type=str, default="5e-4,1e-3,2e-3")
    ap.add_argument("--vocab-cap", type=int, default=4000)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "..",
                                         "extended_study.json"))
    args = ap.parse_args()

    lrs = [float(x) for x in args.lrs.split(",")]
    models = ["R2IE+DTFv3+HDQ", "Transformer", "Mamba", "SSA"]

    DD = DEFAULT_DATA_DIR
    tr = load_wikitext("train", DD)
    te = load_wikitext("test", DD)
    tok = WordTokenizer(tr, args.vocab_cap)
    tl = DataLoader(WordDataset(tr, 64, tok), 32, shuffle=True, drop_last=True)
    tl2 = DataLoader(WordDataset(te, 64, tok), 32, shuffle=False, drop_last=False)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=args.d_model, n_heads=4,
                      n_layers=args.n_layers, d_ff=256, max_seq_len=64,
                      codebook_size=1024, dtf_c_max=0.5, vq_mode="soft")

    # Resume-safe: load existing results.
    results = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)

    for name in models:
        for lr in lrs:
            key = f"{name}|lr={lr}"
            if key in results and args.steps in results[key]:
                print(f"skip {key} (already complete)", flush=True)
                continue
            print(f"=== training {key} to {args.steps} steps ===", flush=True)
            torch.manual_seed(1234)
            t0 = time.time()
            m = build_model(name, cfg)
            train_with_checkpoints(m, tl, args.steps, "cpu", lr,
                                   args.weight_decay, tl2,
                                   args.checkpoint_every, key, results)
            print(f"  done {key} in {time.time()-t0:.0f}s", flush=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
