"""Inference / sampling CLI for a trained R2IE checkpoint.

Usage:
    python -m r2ie.generate --checkpoint checkpoints/r2ie_latest.pt \
        --prompt "THE " --max-tokens 200
"""

from __future__ import annotations

import argparse
import sys

import torch

from .config import ModelConfig
from .data import CharTokenizer
from .model import R2IEModel


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate text from a trained R2IE model.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a saved .pt checkpoint.")
    p.add_argument("--prompt", type=str, default="THE ",
                   help="Initial text to condition generation.")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--use-fast-weights", action="store_true",
                   help="Enable the fast-weight (Condensation Loop) modulation "
                        "at inference time.")
    return p


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "cuda":
        return torch.device("cpu")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ModelConfig(**ckpt["config"])
    chars = ckpt["tokenizer_chars"]
    tokenizer = CharTokenizer("".join(chars))

    model = R2IEModel(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if args.use_fast_weights:
        model.reset_fast_memory()

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [0]
    generated = list(prompt_ids)
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(args.max_tokens):
            logits, _ = model(
                context[:, -config.max_seq_len:],
                use_fast_weights=args.use_fast_weights,
                return_aux=False,
            )
            next_logits = logits[0, -1] / args.temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_id)
            context = torch.tensor([generated], dtype=torch.long, device=device)

    text = tokenizer.decode(generated)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
