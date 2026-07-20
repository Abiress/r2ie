"""Vendor-agnostic device resolution for R2IE.

R2IE targets a single PyTorch codebase that runs unchanged on:
  * NVIDIA GPUs      -> 'cuda'  (CUDA build of torch)
  * AMD GPUs         -> 'cuda'  (ROCm build of torch exposes the same 'cuda'
                         device string; no code change needed)
  * Apple Silicon    -> 'mps'   (Metal Performance Shaders)
  * CPU              -> 'cpu'

We deliberately do NOT hand-write vendor kernels. PyTorch is the portability
layer; any backend it supports works. The C++ HDQ mass-memory extension is
CPU-only, so on non-CPU devices we force the pure-Python condensation fallback
(see condensation.py) to keep all state on the compute device.
"""
from __future__ import annotations

import torch


def resolve_device(device: str) -> torch.device:
    """Resolve a device string to a torch.device, with multi-vendor support.

    'auto' prefers cuda (covers NVIDIA + AMD/ROCm), then mps (Apple), then cpu.
    Explicit requests fall back to cpu with a warning if unavailable.
    """
    if device in ("cuda", "rocm"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        print(f"[warn] {device} requested but not available; falling back to cpu.")
        return torch.device("cpu")
    if device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[warn] mps requested but not available; falling back to cpu.")
        return torch.device("cpu")
    if device == "cpu":
        return torch.device("cpu")
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # Allow raw torch.device strings like 'cuda:1', 'mps', etc.
    try:
        return torch.device(device)
    except Exception:
        print(f"[warn] could not parse device '{device}'; falling back to cpu.")
        return torch.device("cpu")


def backend_name(device: torch.device) -> str:
    """Human-readable backend label for logging / reproducibility."""
    if device.type == "cuda":
        # ROCm reports as 'cuda' under the hood; surface that distinction.
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return "amd-rocm"
        return "nvidia-cuda"
    if device.type == "mps":
        return "apple-mps"
    return "cpu"


def use_cpp_hdq(device: torch.device) -> bool:
    """The C++ HDQ extension is CPU-only; only enable it on CPU.

    On cuda/mps the pure-Python condensation fallback keeps all state on the
    compute device (no host/device copy per step).
    """
    return device.type == "cpu"
