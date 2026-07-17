"""Tests for the ACT Transformation Field.

Deterministic: seeds set so ACT step-count variance is reproducible.
"""

import torch

from r2ie.act_field import ACTField, TransformerBlock


def test_single_ponder_equals_vanilla_block():
    """With max_ponder_steps=1 the ACT field must equal one vanilla block pass."""
    torch.manual_seed(0)
    b, t, d, heads, d_ff = 2, 7, 16, 2, 32
    causal = torch.triu(torch.ones(t, t), diagonal=1).bool()

    field = ACTField(d, heads, d_ff, n_layers=1, max_ponder_steps=1, act_epsilon=0.01)
    block = TransformerBlock(d, heads, d_ff)

    # Share weights so the comparison is meaningful.
    block.load_state_dict(field.blocks[0].state_dict())

    x = torch.randn(b, t, d)
    field_out, ponder_cost, steps = field(x, attn_mask=causal)
    block_out = block(x, attn_mask=causal)

    # One ponder step everywhere.
    assert torch.all(steps == 1)
    # Accumulated output == the single block pass (p == 1 on the only step).
    assert torch.allclose(field_out, block_out, atol=1e-5)
    # Ponder cost is zero when exactly one step is taken.
    assert ponder_cost.item() == 0.0


def test_ponder_steps_vary_with_input_entropy():
    """With max_ponder_steps>1, ponder-step counts must vary across a batch
    with artificially varying input entropy (low-entropy vs high-entropy)."""
    torch.manual_seed(1)
    t, d, heads, d_ff = 12, 16, 2, 32
    max_steps = 4

    field = ACTField(
        d, heads, d_ff, n_layers=1, max_ponder_steps=max_steps, act_epsilon=0.01
    )

    # Batch of 4: constant (low entropy) vs random (high entropy) inputs.
    batch = torch.stack(
        [
            torch.zeros(t, d),          # constant -> low entropy
            torch.ones(t, d) * 0.5,     # constant -> low entropy
            torch.randn(t, d),          # random -> higher entropy
            torch.randn(t, d),          # random -> higher entropy
        ]
    )

    _, _, steps = field(batch)
    # Per-sequence ponder steps (mean over time) should differ across batch.
    per_seq = steps.float().mean(dim=1)
    assert per_seq.std().item() > 0.0, "ponder steps did not vary across batch"
    # No token should exceed max_ponder_steps.
    assert steps.max().item() <= max_steps


def test_ponder_cost_is_nonnegative_scalar():
    torch.manual_seed(2)
    field = ACTField(12, 2, 24, n_layers=2, max_ponder_steps=3, act_epsilon=0.01)
    x = torch.randn(2, 6, 12)
    _, ponder_cost, _ = field(x)
    assert ponder_cost.dim() == 0
    assert ponder_cost.item() >= 0.0
