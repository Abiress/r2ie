"""Tests for the Velocity Governor (your original RALE.docx Loop 2).

Deterministic: seeds set so the stochastic acceleration and event-horizon
checks are reproducible.
"""

import torch

from r2ie.velocity_governor import VelocityGovernor


def test_low_coherence_gets_more_halting_bias():
    torch.manual_seed(0)
    gov = VelocityGovernor(noise_scale=0.2, coherence_floor=0.2, max_noise=0.5)
    # Low-coherence: near-uniform logits (high entropy -> low coherence).
    low = torch.zeros(2, 5, 10)  # uniform probs -> max entropy
    # High-coherence: one logit dominates.
    high = torch.full((2, 5, 10), -10.0)
    high[..., 0] = 10.0

    bias_low = gov.halting_bias(low)
    bias_high = gov.halting_bias(high)
    # Low-coherence tokens get strictly more acceleration bias on average.
    assert bias_low.mean().item() > bias_high.mean().item()


def test_noise_is_bounded():
    torch.manual_seed(1)
    gov = VelocityGovernor(noise_scale=0.3, coherence_floor=0.2, max_noise=0.5)
    low = torch.zeros(4, 6, 10)
    bias = gov.halting_bias(low)
    assert bias.min().item() >= 0.0
    assert bias.max().item() <= gov.max_noise + 1e-6


def test_event_horizon_gate():
    gov = VelocityGovernor()
    energy = torch.tensor([[0.1, 0.9, 0.5]])
    # m=0.3, c^2=2 -> threshold = 0.6
    gate = gov.event_horizon(energy, mass_density=0.3, c_squared=2.0)
    assert gate.reshape(-1).tolist() == [False, True, False]


def test_coherence_high_for_peak_logits():
    gov = VelocityGovernor()
    peak = torch.full((1, 1, 8), -10.0)
    peak[..., 0] = 10.0
    flat = torch.zeros(1, 1, 8)
    assert gov.coherence(peak).item() > gov.coherence(flat).item()


def test_bias_deterministic_under_seed():
    torch.manual_seed(42)
    gov = VelocityGovernor()
    low = torch.zeros(3, 4, 10)
    a = gov.halting_bias(low)
    torch.manual_seed(42)
    b = gov.halting_bias(low)
    assert torch.allclose(a, b)
