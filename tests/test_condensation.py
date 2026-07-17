"""Tests for the Condensation Loop (your original RALE.docx Loop 1).

Deterministic: seeds set so the 1000-iteration stability and swap checks are
reproducible.
"""

import torch

from r2ie.condensation import CondensationLoop


def test_consolidate_changes_mass_and_stays_finite():
    torch.manual_seed(0)
    loop = CondensationLoop(dim=16, alpha=0.05, decay=0.95, clamp=5.0)
    before = loop.mass_norm
    energy = torch.randn(4, 16)
    loop.consolidate(energy, c_squared=2.0)
    after = loop.mass_norm
    assert after != before
    assert torch.isfinite(torch.tensor(after))


def test_norm_stays_bounded_over_many_iterations():
    torch.manual_seed(1)
    dim = 16
    loop = CondensationLoop(dim=dim, alpha=0.1, decay=0.95, clamp=5.0)
    for _ in range(1000):
        energy = torch.randn(3, dim)
        loop.consolidate(energy, c_squared=1.0 + 0.1 * torch.rand(1).item())
        assert torch.isfinite(loop.active.vec).all(), "mass became non-finite"
        # L2 norm cannot exceed sqrt(dim) * clamp.
        max_norm = (dim**0.5) * loop.clamp
        assert loop.mass_norm <= max_norm + 1e-5, f"norm {loop.mass_norm} exceeded bound"
    assert loop.mass_norm > 0.0


def test_swap_moves_state_between_buffers():
    torch.manual_seed(2)
    loop = CondensationLoop(dim=8)
    # Seed active with a recognizable vector.
    with torch.no_grad():
        loop.active.vec.copy_(torch.arange(8, dtype=torch.float32))
    loop.swap()
    # After one swap, background holds the original active values.
    assert torch.allclose(loop.background.vec, torch.arange(8, dtype=torch.float32))
    loop.swap()
    assert torch.allclose(loop.active.vec, torch.arange(8, dtype=torch.float32))


def test_reset_clears_mass():
    torch.manual_seed(3)
    loop = CondensationLoop(dim=8)
    loop.consolidate(torch.randn(2, 8), c_squared=2.0)
    assert loop.mass_norm > 0.0
    loop.reset()
    assert loop.mass_norm == 0.0


def test_modulate_adds_mass_residual():
    torch.manual_seed(4)
    loop = CondensationLoop(dim=10)
    with torch.no_grad():
        loop.active.vec.copy_(torch.full((10,), 0.3))
    x = torch.zeros(3, 10)
    out = loop.modulate(x)
    assert out.shape == x.shape
    assert torch.allclose(out, torch.full((3, 10), 0.3))
    assert torch.isfinite(out).all()
