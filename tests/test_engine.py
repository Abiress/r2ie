"""Tests for the R2IEEngine pipeline.

Deterministic: seeds set so variant checks and resets are reproducible.
"""

import torch

from r2ie.config import ModelConfig
from r2ie.engine import R2IEEngine
from r2ie.errors import CheckpointMismatchError, EmptyPromptError


def _cfg(**kw):
    base = dict(
        vocab_size=32, d_model=24, n_heads=2, n_layers=2, d_ff=48,
        max_seq_len=64, codebook_size=16, max_ponder_steps=3,
    )
    base.update(kw)
    return ModelConfig(**base)


def test_engine_matches_model_when_loops_off():
    torch.manual_seed(0)
    eng = R2IEEngine(_cfg(), use_condensation=False, use_governor=False,
                    use_fast_weights=False)
    idx = torch.randint(0, 32, (2, 10))
    logits_eng, aux_eng = eng.forward_step(idx)
    logits_model, aux_model = eng.model(idx, return_aux=True)
    assert torch.allclose(logits_eng, logits_model, atol=1e-6)
    assert "governor_bias_mean" not in aux_eng
    assert "mass_norm" not in aux_eng


def test_engine_reports_governor_and_mass_when_on():
    torch.manual_seed(1)
    eng = R2IEEngine(_cfg(), use_condensation=True, use_governor=True)
    idx = torch.randint(0, 32, (2, 8))
    logits, aux = eng.forward_step(idx, consolidate=True)
    assert "governor_bias_mean" in aux
    assert "mass_norm" in aux
    assert aux["mass_norm"] > 0.0  # consolidate accreted mass


def test_validate_prompt_rejects_empty():
    eng = R2IEEngine(_cfg())
    try:
        eng.validate_prompt(torch.empty(0, dtype=torch.long))
    except EmptyPromptError:
        pass
    else:
        raise AssertionError("EmptyPromptError not raised")


def test_validate_checkpoint_rejects_vocab_mismatch():
    eng = R2IEEngine(_cfg(vocab_size=32))
    try:
        eng.validate_checkpoint(_cfg(vocab_size=40))
    except CheckpointMismatchError:
        pass
    else:
        raise AssertionError("CheckpointMismatchError not raised")


def test_reset_state_clears_buffers():
    torch.manual_seed(2)
    eng = R2IEEngine(_cfg(), use_condensation=True, use_fast_weights=True)
    idx = torch.randint(0, 32, (2, 8))
    eng.forward_step(idx, consolidate=True)
    assert eng.condensation.mass_norm > 0.0
    assert eng.model.fast_memory.norm > 0.0
    eng.reset_state()
    assert eng.condensation.mass_norm == 0.0
    assert eng.model.fast_memory.norm == 0.0


def test_deterministic_under_seed():
    torch.manual_seed(7)
    eng = R2IEEngine(_cfg(), use_condensation=True, use_governor=True)
    idx = torch.randint(0, 32, (2, 8))
    a, _ = eng.forward_step(idx, consolidate=True)
    torch.manual_seed(7)
    eng2 = R2IEEngine(_cfg(), use_condensation=True, use_governor=True)
    b, _ = eng2.forward_step(idx, consolidate=True)
    assert torch.allclose(a, b, atol=1e-6)
