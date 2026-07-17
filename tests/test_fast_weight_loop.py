"""Tests for the fast-weight Condensation Loop.

Deterministic: seeds set so the 1000-iteration stability check is reproducible.
"""

import torch

from r2ie.fast_weight_loop import FastWeightMemory


def test_update_changes_state():
    torch.manual_seed(0)
    mem = FastWeightMemory(dim=8, decay=0.9, lr_fast=0.1, clamp=5.0)
    before = mem.norm
    q = torch.randn(4, 8)
    v = torch.randn(4, 8)
    mem.apply(q, v)
    after = mem.norm
    assert after != before
    assert torch.isfinite(torch.tensor(after))


def test_norm_stays_bounded_over_many_iterations():
    """After 1000 random updates the fast-weight norm must stay bounded by the
    clamp, with no NaNs/Infs (the spec's stability requirement)."""
    torch.manual_seed(1)
    dim = 16
    mem = FastWeightMemory(dim=dim, decay=0.9, lr_fast=0.1, clamp=5.0)

    for _ in range(1000):
        q = torch.randn(3, dim)
        v = torch.randn(3, dim)
        mem.apply(q, v)
        # Invariant: L2 norm cannot exceed sqrt(dim) * clamp.
        max_norm = (dim**0.5) * mem.clamp
        assert torch.isfinite(mem.fast_w).all(), "fast weights became non-finite"
        assert mem.norm <= max_norm + 1e-5, f"norm {mem.norm} exceeded bound"

    # Also confirm it actually accumulated something non-trivial (not all zeros).
    assert mem.norm > 0.0


def test_reset_clears_state():
    torch.manual_seed(2)
    mem = FastWeightMemory(dim=8)
    mem.apply(torch.randn(2, 8), torch.randn(2, 8))
    assert mem.norm > 0.0
    mem.reset()
    assert mem.norm == 0.0


def test_modulate_shape_and_finite():
    torch.manual_seed(3)
    mem = FastWeightMemory(dim=10)
    mem.apply(torch.randn(5, 10), torch.randn(5, 10))
    x = torch.randn(4, 10)
    out = mem.modulate(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
