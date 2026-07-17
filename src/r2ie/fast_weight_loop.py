"""Condensation Loop - a bounded, decayed Hebbian fast-weight update.

Maps the narrative "Condensation Loop" to a real, trainable-adjacent technique:
fast weights / Hebbian associative update (Hinton & Plaut, 1987; Schmidhuber,
1991; Ba, Hinton & Mnih, 2016, "Using Fast Weights to Attend to the Recent
Past"). Applied only at inference time or during an explicit consolidation
phase - it never mutates the frozen, backprop-trained parameters.

Update rule (decayed outer product, clamped):
    fast_W = decay * fast_W + lr_fast * outer(query, value)
    fast_W = fast_W.clamp(-clamp, clamp)

The fast-weight matrix modulates the output head (see output_head.py / model.py)
and is reset between sequences so it cannot diverge or leak across examples.
"""

import torch
import torch.nn as nn


class FastWeightMemory(nn.Module):
    """Bounded, decayed Hebbian fast-weight buffer.

    This module holds NO learned parameters - its state is a runtime buffer
    updated only via `apply` (at inference/consolidation). Backprop never
    touches it.
    """

    def __init__(
        self,
        dim: int,
        decay: float = 0.9,
        lr_fast: float = 0.1,
        clamp: float = 5.0,
    ):
        super().__init__()
        self.dim = dim
        self.decay = decay
        self.lr_fast = lr_fast
        self.clamp = clamp
        # Register as a buffer so it moves with .to(device) and is excluded
        # from trainable parameters.
        self.register_buffer("fast_w", torch.zeros(dim, dim), persistent=False)

    def reset(self) -> None:
        """Clear fast weights (call between sequences / at the start)."""
        self.fast_w.zero_()

    def apply(self, query: torch.Tensor, value: torch.Tensor) -> None:
        """Apply one Hebbian update in-place (no grad tracking).

        Args:
            query: (..., dim) - the 'presynaptic' vector.
            value: (..., dim) - the 'postsynaptic' vector.
        """
        with torch.no_grad():
            q = query.reshape(-1, self.dim)
            v = value.reshape(-1, self.dim)
            outer = q.t() @ v  # (dim, dim)
            self.fast_w.mul_(self.decay)
            self.fast_w.add_(self.lr_fast * outer)
            self.fast_w.clamp_(min=-self.clamp, max=self.clamp)

    def modulate(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate an input through the fast-weight matrix: x @ fast_W^T.

        Args:
            x: (..., dim).

        Returns:
            (..., dim) modulated features.
        """
        return x @ self.fast_w.t()

    @property
    def norm(self) -> float:
        return float(self.fast_w.norm().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.modulate(x)
