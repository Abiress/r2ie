"""Transformation Field - an ACT-wrapped pre-norm Transformer encoder block.

Maps the narrative "Transformation Field (DTF)" to a real, trainable technique:
Adaptive Computation Time (ACT) of Graves (2016, "Adaptive Computation Time for
Recurrent Neural Networks") applied per-token inside a standard pre-norm
Transformer encoder block (Vaswani et al., 2017, "Attention Is All You Need").

ACT loop per token:
  - At each ponder step compute a halting probability h_t = sigmoid(W_h @ s_t).
  - Accumulate weighted hidden states and cumulative halting probability.
  - Stop when cumulative halting >= 1 - epsilon or max_ponder_steps reached.
  - Add the standard ACT ponder cost (sum of ponder steps) to the loss, weighted
    by `ponder_weight`, encouraging fewer steps on easy tokens.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """A single pre-norm Transformer encoder block (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm attention sub-layer (residual).
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        # Pre-norm feed-forward sub-layer (residual).
        x = x + self.ff(self.ff_norm(x))
        return x


class ACTField(nn.Module):
    """Wraps a stack of Transformer blocks in an Adaptive Computation Time loop.

    With `max_ponder_steps == 1` this collapses to a single vanilla block pass
    (the halting probability is forced to 1 on the first step), which the unit
    test relies on for equivalence checking.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int = 1,
        max_ponder_steps: int = 4,
        act_epsilon: float = 0.01,
        act_ponder_weight: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_ponder_steps = max_ponder_steps
        self.epsilon = act_epsilon
        self.ponder_weight = act_ponder_weight

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        # Halting unit: linear projection of hidden state -> halting logit.
        self.halt = nn.Linear(d_model, 1)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the ACT loop over the token sequence.

        Args:
            x: input tensor (B, T, d_model).
            attn_mask: optional causal mask (T, T).

        Returns:
            out: adapted hidden states (B, T, d_model).
            ponder_cost: scalar sum-of-ponder-steps cost (for the loss).
            steps: (B, T) int tensor of ponder steps taken per token.
        """
        b, t, d = x.shape
        # Running (weighted) accumulated state and cumulative halting prob.
        accum = torch.zeros_like(x)
        halting = torch.zeros(b, t, device=x.device, dtype=x.dtype)
        steps = torch.zeros(b, t, device=x.device, dtype=torch.long)

        for step in range(self.max_ponder_steps):
            # Apply all wrapped transformer blocks to the current state.
            s = x
            for block in self.blocks:
                s = block(s, attn_mask=attn_mask)

            # Halting probability for this ponder step.
            h = torch.sigmoid(self.halt(s))  # (B, T, 1)
            h = h.squeeze(-1)

            if self.max_ponder_steps == 1:
                # Single-step configuration collapses to exactly one vanilla
                # block pass with full weight (p == 1).
                p = torch.ones_like(h)
            else:
                # Remaining probability needed to reach the halting threshold.
                remainder = (1.0 - self.epsilon - halting).clamp(min=0.0)
                # On the last allowed step, always halt with the full remainder.
                if step == self.max_ponder_steps - 1:
                    p = remainder
                else:
                    p = torch.min(h, remainder)

            accum = accum + p.unsqueeze(-1) * s
            halting = halting + p
            steps = steps + (p > 0).long()
            x = s  # state advances for the next ponder step

            # Stop early only when every token has halted.
            if bool((halting >= 1.0 - self.epsilon).all()):
                break

        # Standard ACT renormalization: accumulated weights sum to < 1 when the
        # loop stops before full halting, so divide by the total halting mass
        # to keep the output on a comparable scale.
        total = halting.clamp(min=1e-6).unsqueeze(-1)
        accum = accum / total

        # Ponder cost: encourage fewer steps. Sum of (steps - 1) clamped at 0,
        # averaged over the batch/sequence, scaled by ponder_weight.
        excess = (steps - 1).clamp(min=0).float()
        ponder_cost = self.ponder_weight * excess.mean()

        return accum, ponder_cost, steps
