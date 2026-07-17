"""Training loop and CLI entrypoint for R2IE.

Usage:
    python -m r2ie.train --data tinyshakespeare --steps 2000 --device auto
    python -m r2ie.train --data <path-to-txt> --steps 500 --device cpu

Logs cross-entropy loss, VQ commitment loss, average ponder steps and
perplexity to stdout and a CSV under runs/. Saves checkpoints under
checkpoints/.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch

from .config import ModelConfig
from .data import make_dataloader
from .engine import R2IEEngine
from .metrics import average_ponder_steps, perplexity


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the R2IE prototype.")
    p.add_argument("--data", type=str, default="tinyshakespeare",
                   help="Corpus source: 'tinyshakespeare', a .txt path, or 'fixture'.")
    p.add_argument("--steps", type=int, default=2000, help="Number of training steps.")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"], help="Device to train on.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--vocab-size", type=int, default=None,
                   help="Override vocab size (auto-set from corpus if None).")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--use-condensation", action="store_true",
                   help="Enable the Condensation Loop (RALE.docx Loop 1) at "
                        "consolidation time. Off by default.")
    p.add_argument("--use-governor", action="store_true",
                   help="Enable the Velocity Governor (RALE.docx Loop 2) for "
                        "coherence-driven compute acceleration. Off by default.")
    p.add_argument("--use-fast-weights", action="store_true",
                   help="Enable fast-weight (Condensation Loop) head modulation. "
                        "Off by default.")
    return p


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "cuda":
        print("[warn] cuda requested but not available; falling back to cpu.")
        return torch.device("cpu")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"[info] device = {device}")

    # Build data + tokenizer (vocab size derived from corpus).
    loader, tokenizer = make_dataloader(
        args.data, args.seq_len, args.batch_size, shuffle=True
    )
    vocab_size = args.vocab_size or tokenizer.vocab_size
    print(f"[info] vocab_size = {vocab_size} (chars)")

    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=max(args.seq_len, 256),
        codebook_size=128,
        commitment_cost=0.25,
        max_ponder_steps=4,
        act_epsilon=0.01,
        act_ponder_weight=0.01,
        fast_weight_decay=0.9,
        fast_weight_lr=0.1,
        fast_weight_clamp=5.0,
    )
    engine = R2IEEngine(
        config,
        use_condensation=args.use_condensation,
        use_governor=args.use_governor,
        use_fast_weights=args.use_fast_weights,
    ).to(device)
    optimizer = torch.optim.Adam(engine.model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.runs_dir, exist_ok=True)
    csv_path = os.path.join(args.runs_dir, "train_log.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "ce_loss", "commitment_loss", "ponder_cost",
                     "avg_ponder_steps", "perplexity", "governor_bias", "mass_norm"])
    print(f"[info] logging to {csv_path}")

    engine.model.train()
    loader_iter = iter(loader)
    initial_loss = None
    final_loss = None

    for step in range(1, args.steps + 1):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, aux = engine.forward_step(x, consolidate=args.use_condensation)
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        total = ce + aux["commitment_loss"] + aux["ponder_cost"]
        total.backward()
        optimizer.step()

        # Track the cross-entropy (language-modeling) loss for the convergence
        # verdict - it is the primary training objective. The VQ commitment and
        # ACT ponder terms are auxiliary regularizers and are not used to judge
        # whether the model learned.
        if initial_loss is None:
            initial_loss = ce.item()
        final_loss = ce.item()

        if step % args.log_every == 0 or step == 1:
            ppl = perplexity(ce)
            aps = average_ponder_steps(aux["ponder_steps"])
            extra = ""
            if "governor_bias_mean" in aux:
                extra += f" | gov {aux['governor_bias_mean'].item():.4f}"
            if "mass_norm" in aux:
                extra += f" | mass {aux['mass_norm']:.4f}"
            print(
                f"step {step:>6} | ce {ce.item():.4f} | vq {aux['commitment_loss'].item():.4f} "
                f"| ponder {aux['ponder_cost'].item():.4f} | avg_steps {aps:.2f} | ppl {ppl:.2f}{extra}"
            )
            writer.writerow([
                step, f"{ce.item():.4f}", f"{aux['commitment_loss'].item():.4f}",
                f"{aux['ponder_cost'].item():.4f}", f"{aps:.4f}", f"{ppl:.4f}",
                f"{aux.get('governor_bias_mean', torch.tensor(0.0)).item():.4f}",
                f"{aux.get('mass_norm', 0.0):.4f}",
            ])
            csv_file.flush()

    csv_file.close()

    ckpt = os.path.join(args.checkpoint_dir, "r2ie_latest.pt")
    engine.save(ckpt)
    # Persist the tokenizer chars alongside the engine checkpoint.
    torch.save({"tokenizer_chars": tokenizer.chars},
               os.path.join(args.checkpoint_dir, "tokenizer.pt"))
    print(f"[info] saved checkpoint -> {ckpt}")

    if initial_loss is not None and final_loss is not None:
        print(f"[info] initial CE {initial_loss:.4f} -> final CE {final_loss:.4f}")
        if final_loss < 0.9 * initial_loss:
            print("[ok] cross-entropy decreased by >10% (model is learning).")
        else:
            print("[warn] CE did not decrease by >10%; consider more steps / tuning.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
