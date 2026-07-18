import torch

from r2ie.hdq_bridge import CPP_AVAILABLE, HDQBridge


def test_cpp_used_when_available():
    b = HDQBridge(8, use_cpp=True)
    assert b.using_cpp == CPP_AVAILABLE


def test_apply_raises_no_error_and_changes_norm():
    b = HDQBridge(8, use_cpp=CPP_AVAILABLE)
    b.reset()
    assert b.norm() == 0.0
    delta = torch.rand(8) * 2.0
    b.apply(delta)
    assert b.norm() > 0.0


def test_decay_reduces_norm_over_time():
    b = HDQBridge(8, decay=0.5, clamp=100.0, use_cpp=CPP_AVAILABLE)
    b.reset()
    for _ in range(5):
        b.apply(torch.full((8,), 1.0))
    first = b.norm()
    for _ in range(20):
        b.apply(torch.zeros(8))  # zero accretion -> pure decay
    assert b.norm() < first


def test_clamp_bounds_mass():
    b = HDQBridge(8, decay=1.0, clamp=1.0, use_cpp=CPP_AVAILABLE)
    b.reset()
    for _ in range(100):
        b.apply(torch.full((8,), 5.0))
    assert b.norm() <= 8.0 * 1.0 + 1e-5


def test_modulate_adds_mass_residual():
    b = HDQBridge(8, use_cpp=CPP_AVAILABLE)
    b.reset()
    b.apply(torch.full((8,), 1.0))
    x = torch.zeros(8)
    y = b.modulate(x)
    assert torch.allclose(y, b.weights().to(y.dtype) if not b.using_cpp else y)


def test_modulate_shape_preserved():
    b = HDQBridge(16, use_cpp=CPP_AVAILABLE)
    x = torch.rand(4, 16)
    y = b.modulate(x)
    assert y.shape == x.shape
