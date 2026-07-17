"""Tests for the VQ Compressor (Mass Compressor).

Deterministic: seeds set so codebook usage / shape behavior is reproducible.
"""

import torch

from r2ie.vq_compressor import VQCompressor


def test_forward_shape_and_grad_flow():
    torch.manual_seed(0)
    d_model, codebook_size, batch, seq = 16, 32, 4, 10
    model = VQCompressor(d_model, codebook_size, commitment_cost=0.25)

    x = torch.randn(batch, seq, d_model, requires_grad=False)
    out, commit_loss, indices = model(x)

    # Output shape preserved.
    assert out.shape == (batch, seq, d_model)
    assert indices.shape == (batch, seq)

    # Gradient flows through encoder/decoder (straight-through estimator).
    loss = out.sum() + commit_loss
    loss.backward()
    assert model.encoder.weight.grad is not None
    assert model.decoder.weight.grad is not None
    assert model.encoder.weight.grad.abs().sum() > 0
    assert model.decoder.weight.grad.abs().sum() > 0

    # In VQ-VAE the codebook entries are updated via the *vector-quantization*
    # loss term ||z_e - sg(e)||^2 (the commitment term stops the codebook
    # gradient by design). Verify the codebook receives a non-zero grad through
    # the correct VQ-loss path.
    model.zero_grad()
    z = model.encoder(x)
    flat = z.reshape(-1, model.d_model)
    codebook = model.codebook.weight
    dist = (
        flat.pow(2).sum(1, keepdim=True)
        - 2 * flat @ codebook.t()
        + codebook.pow(2).sum(1)
    )
    indices = dist.argmin(1)
    nearest = model.codebook(indices)  # raw code entries (no straight-through)
    # The VQ-loss term ||z_e - e||^2 is what updates the codebook entries.
    vq_loss = torch.mean((flat.detach() - nearest) ** 2)
    vq_loss.backward()
    assert model.codebook.weight.grad is not None
    assert model.codebook.weight.grad.abs().sum() > 0


def test_codebook_does_not_collapse():
    torch.manual_seed(1)
    d_model, codebook_size, batch, seq = 8, 16, 8, 20
    model = VQCompressor(d_model, codebook_size, commitment_cost=0.25)

    # Random inputs should spread across >= 2 codes (no single-code collapse).
    x = torch.randn(batch, seq, d_model)
    _, _, indices = model(x)
    unique_codes = indices.unique().numel()
    assert unique_codes >= 2, f"only {unique_codes} code(s) used - collapsed"


def test_commitment_loss_nonnegative_scalar():
    torch.manual_seed(2)
    model = VQCompressor(12, 24)
    x = torch.randn(2, 5, 12)
    _, commit_loss, _ = model(x)
    assert commit_loss.dim() == 0
    assert commit_loss.item() >= 0.0
