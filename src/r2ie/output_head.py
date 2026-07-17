"""Coherence Projection Layer (CPL) - the output head.

Standard linear projection to vocab size with a temperature-scaled softmax.
Cross-entropy is computed against the target tokens during training. When a
fast-weight matrix is supplied it modulates the hidden state before projection
(the "Condensation Loop" modulating the head, applied only at inference /
consolidation - never during the backprop-trained pass).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputHead(nn.Module):
    """Linear projection head with temperature-scaled softmax."""

    def __init__(self, d_model: int, vocab_size: int, temperature: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        self.temperature = temperature

    def forward(
        self,
        x: torch.Tensor,
        fast_modulation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project hidden states to logits.

        Args:
            x: (B, T, d_model) hidden states.
            fast_modulation: optional (B, T, d_model) fast-weight-modulated
                residual added to x before projection (inference/consolidation
                only). If None, only the trained hidden state is used.

        Returns:
            logits: (B, T, vocab_size), pre-softmax (temperature applied to the
            softmax probabilities, not the logits, for numerical stability).
        """
        if fast_modulation is not None:
            x = x + fast_modulation
        return self.proj(x)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss over the token dimension.

        Args:
            logits: (B, T, vocab_size).
            targets: (B, T) long token ids.
        """
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample token ids from a (..., vocab_size) logits tensor.

        Temperature scaling is applied to the probabilities.
        """
        probs = F.softmax(logits / self.temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
