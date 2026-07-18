"""Inference / sampling CLI for a trained R2IE checkpoint.

Usage:
    python -m r2ie.generate --checkpoint checkpoints/r2ie_latest.pt \
        --prompt "THE " --max-tokens 200
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from .config import ModelConfig
from .data import CharTokenizer
from .engine import R2IEEngine
from .errors import EmptyPromptError


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
                   help="Enable the fast-weight modulation at inference time.")
    p.add_argument("--use-condensation", action="store_true",
                   help="Enable the Condensation Loop accretion during generation.")
    p.add_argument("--use-governor", action="store_true",
                    help="Enable the Velocity Governor compute acceleration.")
    p.add_argument("--use-hdq", action="store_true",
                    help="Use the C++ HDQ mass buffer for the Condensation Loop.")
    p.add_argument("--use-dtf", action="store_true",
                    help="Enable the DTF Transformation Field during generation.")
    p.add_argument("--vq-mode", type=str, default="hard", choices=["hard", "soft"],
                    help="VQ Mass Compressor mode used for training.")
    return p


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "cuda":
        return torch.device("cpu")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _load_tokenizer(checkpoint_path: str, ckpt: dict, device: str) -> CharTokenizer:
    if "tokenizer_chars" in ckpt:
        chars = ckpt["tokenizer_chars"]
    else:
        tok_path = os.path.join(os.path.dirname(checkpoint_path), "tokenizer.pt")
        tok_ckpt = torch.load(tok_path, map_location=device, weights_only=False)
        chars = tok_ckpt["tokenizer_chars"]
    return CharTokenizer("".join(chars))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ModelConfig(**ckpt["config"])
    tokenizer = _load_tokenizer(args.checkpoint, ckpt, device)

    # Honor loop flags from the checkpoint unless overridden on the CLI.
    use_condensation = ckpt.get("use_condensation", False) or args.use_condensation
    use_governor = ckpt.get("use_governor", False) or args.use_governor
    use_fast_weights = ckpt.get("use_fast_weights", False) or args.use_fast_weights
    use_hdq = ckpt.get("use_hdq", False) or args.use_hdq
    use_dtf = ckpt.get("use_dtf", False) or args.use_dtf
    vq_mode = ckpt.get("vq_mode", args.vq_mode)

    engine = R2IEEngine(
        config,
        use_condensation=use_condensation,
        use_governor=use_governor,
        use_fast_weights=use_fast_weights,
        use_hdq=use_hdq,
        use_dtf=use_dtf,
        vq_mode=vq_mode,
    ).to(device)
    engine.model.load_state_dict(ckpt["model_state"])
    engine.model.eval()
    engine.reset_state()

    prompt_ids = tokenizer.encode(args.prompt)
    try:
        engine.validate_prompt(torch.tensor(prompt_ids, dtype=torch.long))
    except EmptyPromptError:
        print("[warn] prompt encoded to zero tokens; using a single space.")
        prompt_ids = tokenizer.encode(" ")

    generated = list(prompt_ids)
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(args.max_tokens):
            logits, _ = engine.forward_step(
                context[:, -config.max_seq_len:], consolidate=use_condensation
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
