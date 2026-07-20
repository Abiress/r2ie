"""Distributed-training seam for large/commercial-scale runs.

This module provides a single, backend-agnostic entry point for scaling R2IE
beyond a single device. It is intentionally a *thin seam*: real 1T-param / 10T-
token training requires a data + model parallelism strategy, sharded optimizers,
and a data pipeline, but the integration point is the same everywhere — PyTorch's
``torch.distributed`` + ``FullyShardedDataParallel`` (FSDP), which runs on NVIDIA
CUDA, AMD ROCm, Apple MPS, and CPU through one API.

Nothing here executes unless ``torch.distributed`` is initialized by the caller
(e.g. via ``torchrun``). On a single device the model is returned unchanged, so
this code path is safe and a no-op in the default/local case.

NOTE: this seam is implemented and import-safe, but has NOT been runtime-verified
on a multi-GPU / multi-node machine in this environment (CPU-only, no cluster).
It is the correct integration point for such runs; verify on your hardware.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

try:
    from torch.distributed import DistributedDataParallel
    from torch.distributed.fsdp import CPUOffload, MixedPrecision
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    _HAS_FSDP = True
except Exception:  # pragma: no cover - depends on torch build
    _HAS_FSDP = False


def is_distributed() -> bool:
    """True if a distributed process group is already initialized."""
    return _HAS_FSDP and torch.distributed.is_initialized()


def wrap_for_distributed(
    model: nn.Module,
    *,
    use_fsdp: bool = True,
    cpu_offload: bool = False,
    mixed_precision: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Wrap ``model`` for distributed training if a process group exists.

    Args:
        model: the raw ``nn.Module`` (R2IEModel or a baseline).
        use_fsdp: if True and distributed, wrap in FSDP; else DDP.
        cpu_offload: offload params to host RAM (useful at very large scale).
        mixed_precision: None | 'fp16' | 'bf16' for the FSDP shard compute.
        device: current compute device (for MP dtype placement).

    Returns the (possibly wrapped) model. On a single process the input model is
    returned unchanged.
    """
    if not is_distributed():
        return model

    if not _HAS_FSDP:
        raise RuntimeError(
            "torch.distributed / FSDP not available in this torch build; "
            "cannot wrap for distributed training."
        )

    mp = None
    if mixed_precision in ("fp16", "bf16"):
        dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
        mp = MixedPrecision(
            param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype
        )

    if use_fsdp:
        return FSDP(
            model,
            cpu_offload=CPUOffload(offload=cpu_offload),
            mixed_precision=mp,
            device_id=device.index if (device and device.type == "cuda") else None,
        )
    return DistributedDataParallel(model, device_ids=([device.index] if device and device.type == "cuda" else None))


def maybe_init_process_group(backend: Optional[str] = None) -> None:
    """Initialize a process group if env vars (MASTER_ADDR/PORT/RANK/WORLD_SIZE)
    are present and one isn't already active. Call before ``wrap_for_distributed``.

    The backend is auto-selected: 'nccl' (NVIDIA/AMD GPUs), 'gloo' (CPU/portable),
    or 'mpi'. This keeps the launcher vendor-agnostic.
    """
    if torch.distributed.is_initialized():
        return
    if not all(os.environ.get(k) for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")):
        return
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)
