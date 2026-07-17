"""Optional benchmark: compare R2IE against a matched-parameter plain Transformer.

This script is OFF by default and is not required for the prototype to work.
If you run it, paste its LITERAL output (loss curves, param counts, wall-clock
time) into BENCHMARKS.md - never fabricate numbers, and never claim R2IE beats
production Transformers unless this script was actually run and its output is
attached verbatim with the run command and hardware noted.

Usage:
    python scripts/run_benchmark.py --steps 1000 --device cpu
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from r2ie.config import ModelConfig
from r2ie.data import CharTokenizer, load_corpus_text
from r2ie.model import R2IEModel


class PlainTransformer(nn.Module):
    """A matched-param-count plain Transformer (no VQ, no ACT, no fast weights)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        x = self.embed(idx) + self.pos[:, :t, :]
        causal = torch.triu(torch.ones(t, t), diagonal=1).bool()
        h = self.enc(x, mask=causal)
        return self.head(h)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def run(model, name, loader, steps, device):
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    it = iter(loader)
    import torch.nn.functional as F

    t0 = time.time()
    for s in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        if isinstance(model, R2IEModel):
            logits, aux = model(x, use_fast_weights=False)
            loss = (
                F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                + aux["commitment_loss"]
                + aux["ponder_cost"]
            )
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        opt.step()
    dt = time.time() - t0
    return count_params(model), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tinyshakespeare")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    text = load_corpus_text(args.data)
    tok = CharTokenizer(text)
    loader, _ = __import__("r2ie.data", fromlist=["make_dataloader"]).make_dataloader(
        args.data, args.seq_len, args.batch_size
    )

    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=max(args.seq_len, 256),
        codebook_size=128,
        max_ponder_steps=4,
    )

    r2ie = R2IEModel(cfg)
    plain = PlainTransformer(cfg)

    print("=== R2IE benchmark (optional, not a claim of superiority) ===")
    p_r2ie, t_r2ie = run(r2ie, "R2IE", loader, args.steps, args.device)
    p_plain, t_plain = run(plain, "PlainTransformer", loader, args.steps, args.device)
    print(f"R2IE        params={p_r2ie}  time={t_r2ie:.1f}s")
    print(f"PlainTrans  params={p_plain}  time={t_plain:.1f}s")
    print("[note] Paste this output verbatim into BENCHMARKS.md if you cite it.")


if __name__ == "__main__":
    main()
