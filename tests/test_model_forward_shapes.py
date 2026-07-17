"""Tests for the full R2IEModel forward pass: shapes, aux losses, gradient flow."""

import torch

from r2ie.config import ModelConfig
from r2ie.model import R2IEModel


def _make_model():
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32,
        d_model=24,
        n_heads=2,
        n_layers=2,
        d_ff=48,
        max_seq_len=64,
        codebook_size=16,
        commitment_cost=0.25,
        max_ponder_steps=3,
        act_epsilon=0.01,
        act_ponder_weight=0.01,
        fast_weight_decay=0.9,
        fast_weight_lr=0.1,
        fast_weight_clamp=5.0,
    )
    return R2IEModel(cfg)


def test_forward_shapes_and_aux():
    model = _make_model()
    b, t = 2, 10
    idx = torch.randint(0, model.config.vocab_size, (b, t))

    logits, aux = model(idx, use_fast_weights=False, return_aux=True)
    assert logits.shape == (b, t, model.config.vocab_size)
    assert "commitment_loss" in aux
    assert "ponder_cost" in aux
    assert "ponder_steps" in aux
    # Aux losses are scalars; ponder steps has the sequence shape.
    assert aux["commitment_loss"].dim() == 0
    assert aux["ponder_cost"].dim() == 0
    assert aux["ponder_steps"].shape == (b, t)


def test_gradient_flow_through_core_params():
    model = _make_model()
    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    logits, aux = model(idx)
    loss = logits.sum() + aux["commitment_loss"] + aux["ponder_cost"]
    loss.backward()

    assert model.embedding.weight.grad is not None
    assert model.vq.encoder.weight.grad is not None
    assert model.act_field.blocks[0].attn.in_proj_weight.grad is not None
    assert model.head.proj.weight.grad is not None
    # Core trained params received non-zero gradient.
    assert model.head.proj.weight.grad.abs().sum() > 0


def test_fast_weight_modulation_does_not_mutate_trained_params():
    model = _make_model()
    idx = torch.randint(0, model.config.vocab_size, (2, 8))

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.reset_fast_memory()
    # Run with fast weights; this should update the fast-memory buffer but not
    # the trained parameters.
    _ = model(idx, use_fast_weights=True)
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), before[n]), f"trained param {n} was mutated"


def test_fast_memory_resets_between_calls():
    model = _make_model()
    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    model.reset_fast_memory()
    assert model.fast_memory.norm == 0.0
    _ = model(idx, use_fast_weights=True)
    assert model.fast_memory.norm > 0.0
    model.reset_fast_memory()
    assert model.fast_memory.norm == 0.0
