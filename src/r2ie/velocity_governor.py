"""Velocity Governor - your original R2IE Loop 2 (RALE.docx).

Maps the narrative:
    c(t+1)^2 = c(t)^2 + f_stochastic(grad E(t))     "Velocity Update Loop"
    halt when  E > m * c^2                          "Event Horizon"

to a real, bounded mechanism layered on top of the ACT field:

- Coherence of the current distribution is estimated as the negative entropy
  of the softmax over the vocab (high confidence -> high coherence).
- When coherence is LOW (the model is "confused"), bounded stochastic noise is
  injected into the ACT halting logits. This raises the effective halting
  probability pressure, which - combined with the ACT remainder rule - lets the
  field take MORE ponder steps (higher c^2) before halting. This is the
  "stochastic acceleration until escape velocity" behavior.
- The Event Horizon gate exposes a check: given an energy density (mean logit
  magnitude) and a mass density (norm of the condensation mass) and c^2, returns
  whether E > m * c^2 (i.e. the loop may emit).

Everything is deterministic under a fixed seed; the injected noise is bounded
and the ponder steps are always capped by max_ponder_steps.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class VelocityGovernor(nn.Module):
    """Modulates ACT compute intensity via coherence-driven stochastic noise."""

    def __init__(
        self,
        noise_scale: float = 0.1,
        coherence_floor: float = 0.2,
        max_noise: float = 0.5,
    ):
        super().__init__()
        self.noise_scale = noise_scale
        self.coherence_floor = coherence_floor
        self.max_noise = max_noise

    def coherence(self, logits: torch.Tensor) -> torch.Tensor:
        """Per-token coherence = negative entropy of the softmax (B, T)."""
        probs = F.softmax(logits, dim=-1)
        # Entropy per token; negate so high confidence -> high coherence.
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        return -entropy  # (B, T)

    def halting_bias(self, logits: torch.Tensor) -> torch.Tensor:
        """Bounded noise added to ACT halt logits when coherence is low.

        Returns a (B, T) tensor of non-negative bias that pushes the ACT field
        to keep looping (raise c^2) on confused tokens.
        """
        coh = self.coherence(logits)  # (B, T)
        # Deficit: how far below the floor (clamped at >= 0).
        deficit = (self.coherence_floor - coh).clamp(min=0.0)
        # Bounded stochastic acceleration.
        noise = torch.rand_like(deficit) * self.noise_scale * deficit
        return noise.clamp(min=0.0, max=self.max_noise)

    def event_horizon(
        self, energy_density: torch.Tensor, mass_density: float, c_squared: float
    ) -> torch.Tensor:
        """Event Horizon gate: E > m * c^2 ?

        Args:
            energy_density: (B, T) scalar field energy (e.g. mean |logit|).
            mass_density: scalar mass density (e.g. condensation mass norm).
            c_squared: computational intensity (avg ponder steps).

        Returns:
            (B, T) bool tensor, True where emission is permitted.
        """
        if c_squared <= 0.0:
            c_squared = 1.0
        threshold = mass_density * c_squared
        return energy_density > threshold
