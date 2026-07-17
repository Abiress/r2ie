"""Condensation Loop - your original R2IE accretion mechanism (RALE.docx, Loop 1).

Maps the narrative:
    m(t+1) = m(t) + alpha * (E(t) / c(t)^2)
to a real, bounded, differentiable-safe module.

Instead of overwriting the trained model weights in place (which would break
autograd), the "mass" is a SEPARATE persistent buffer - the MassState - held in
a double-buffered pair (active / background). Accretion writes high-confidence
output energy (E) into the mass, divided by the current computational intensity
(c^2 = average ponder steps), scaled by the accretion rate alpha. The buffer is
decayed and hard-clamped so it cannot diverge, and swap() flips the active and
background references (the lock-free pointer swap from the original design,
realized as a safe tensor operation in PyTorch - no threads required).

The loop is only ever updated inside an explicit consolidate() call (inference /
consolidation phase), never silently during the backprop-trained forward pass.
"""

import torch
import torch.nn as nn


class MassState(nn.Module):
    """A single mass buffer (the 'm' in the accretion rule)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.vec = nn.Parameter(torch.zeros(dim))

    def zero_(self) -> None:
        with torch.no_grad():
            self.vec.zero_()

    def norm(self) -> float:
        return float(self.vec.norm().item())


class CondensationLoop(nn.Module):
    """Double-buffered mass accumulator implementing the Condensation Loop.

    Holds an active and a background MassState. consolidate() accretes energy
    into the background buffer, then swaps active <-> background, so the next
    forward read always sees the freshly consolidated mass without blocking.
    """

    def __init__(
        self,
        dim: int,
        alpha: float = 0.05,
        decay: float = 0.95,
        clamp: float = 5.0,
    ):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.decay = decay
        self.clamp = clamp
        self.active = MassState(dim)
        self.background = MassState(dim)

    @torch.no_grad()
    def consolidate(self, energy: torch.Tensor, c_squared: float) -> None:
        """Accrete output energy into the mass buffer.

        Args:
            energy: (..., dim) high-confidence output energy (e.g. residual
                activations of the chosen token).
            c_squared: current computational intensity (e.g. avg ponder steps);
                guards against divide-by-zero.
        """
        if c_squared <= 0.0:
            c_squared = 1.0
        # Aggregate the energy residual to a single (dim,) vector.
        e = energy.reshape(-1, self.dim).mean(dim=0)
        delta = self.alpha * (e / c_squared)
        # Write into background, then flip: the lock-free pointer swap.
        self.background.vec.mul_(self.decay)
        self.background.vec.add_(delta)
        self.background.vec.clamp_(min=-self.clamp, max=self.clamp)
        self.swap()

    def swap(self) -> None:
        """Flip active and background references (no data copy of values)."""
        self.active, self.background = self.background, self.active

    def reset(self) -> None:
        with torch.no_grad():
            self.active.zero_()
            self.background.zero_()

    def modulate(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate hidden state x by the active mass (additive residual)."""
        return x + self.active.vec

    @property
    def mass_norm(self) -> float:
        return self.active.norm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.modulate(x)
