"""Mass Compressor - a VQ-VAE style vector-quantization bottleneck.

Maps the narrative "Mass Compressor (HDQ)" to a real, trainable technique:
Vector Quantization as in VQ-VAE (van den Oord et al., 2017,
"Neural Discrete Representation Learning").

Pipeline: linear encoder -> nearest-neighbor lookup in a learnable codebook
(nn.Embedding) -> straight-through gradient estimator -> linear decoder.
A commitment loss term (beta * ||z_e - sg(z_q)||^2) is returned for training.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQCompressor(nn.Module):
    """Vector-quantization bottleneck used as the input "Mass Compressor"."""

    def __init__(self, d_model: int, codebook_size: int, commitment_cost: float = 0.25):
        super().__init__()
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost

        self.encoder = nn.Linear(d_model, d_model)
        self.codebook = nn.Embedding(codebook_size, d_model)
        self.decoder = nn.Linear(d_model, d_model)

        # Initialize codebook with small random vectors.
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map encoder outputs to the nearest codebook entries.

        Args:
            z: encoder output of shape (..., d_model).

        Returns:
            z_q: quantized tensor (..., d_model) with straight-through grad.
            commitment_loss: scalar commitment term.
            indices: (...,) long tensor of codebook indices selected.
        """
        # Flatten all but last dim for distance computation.
        flat = z.reshape(-1, self.d_model)  # (N, d)
        codebook = self.codebook.weight  # (K, d)

        # Euclidean distances to each code.
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.pow(2).sum(1)
        )  # (N, K)
        indices = dist.argmin(1)  # (N,)
        z_q = self.codebook(indices).view_as(z)

        # Straight-through estimator: copy gradient from z_q back to z.
        z_q = z + (z_q - z).detach()

        # Commitment loss keeps encoder output near chosen code.
        commitment_loss = F.mse_loss(z.detach(), z_q)
        # Codebook loss: pull codebook entries toward the encoder outputs.
        # This is the second half of the standard VQ-VAE loss and keeps the
        # codebook aligned with the data so the commitment term cannot drift
        # unbounded during training.
        codebook_loss = F.mse_loss(z, z_q.detach())

        return z_q, commitment_loss, codebook_loss, indices.view(z.shape[:-1])

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: input tensor (..., d_model).

        Returns:
            out: reconstructed/decoded tensor (..., d_model).
            commitment_loss: scalar.
            indices: codebook indices (...,).
        """
        z = self.encoder(x)
        z_q, commitment_loss, codebook_loss, indices = self.quantize(z)
        out = self.decoder(z_q)
        vq_loss = self.commitment_cost * commitment_loss + codebook_loss
        return out, vq_loss, indices
