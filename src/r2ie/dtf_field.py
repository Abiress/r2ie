"""DTF Transformation Field.

From RALE.docx: a trainable graph-diffusion operator applied over token
positions. Each token position carries a learned diffusion coefficient c^2
derived from a "coherence" signal; the field evolves as

    h <- h + c^2 * Laplacian(h)

where the Laplacian is the discrete second difference along the sequence
dimension (token positions). This lets nearby positions mix under a learned,
data-dependent diffusion rate instead of fixed attention.

Design:
  * DTFOperator computes per-token c^2 from a small linear map on the hidden
    state (clamped to a non-negative, bounded range) and applies the diffusion.
  * DTFBlock wraps DTFOperator + a feed-forward + residual (pre-norm).
  * The operator is a proper nn.Module: c^2 coefficients are trained
    parameters; hidden state tensors are never mutated in place.

The diffusion is bounded (c^2 in [0, c_max]) so the operator is a stable
averaging step, not a runaway high-pass filter.
"""

from typing import Optional

import torch
import torch.nn as nn


class DTFOperator(nn.Module):
    """Trainable graph-diffusion over token positions."""

    def __init__(self, d_model: int, c_max: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.c_max = c_max
        self.eps = eps
        # coherence -> raw logit for c^2; one scalar per token position per layer.
        self.coherence = nn.Linear(d_model, 1, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply one diffusion step.

        Args:
            h: (B, T, D) hidden states.

        Returns:
            h_out: (B, T, D) after one graph-diffusion step (new tensor).
        """
        b, t, d = h.shape
        # Learned, bounded per-token diffusion coefficient c^2 in [0, c_max].
        c2 = torch.sigmoid(self.coherence(h)) * self.c_max  # (B, T, 1)

        # Discrete Laplacian along the sequence axis. Endpoints use one-sided
        # differences so the operator is well-defined for any T >= 1.
        if t == 1:
            lap = torch.zeros_like(h)
        else:
            prev = torch.cat([h[:, :1, :], h[:, :-1, :]], dim=1)  # (B, T, D)
            nxt = torch.cat([h[:, 1:, :], h[:, -1:, :]], dim=1)  # (B, T, D)
            lap = prev + nxt - 2.0 * h  # (B, T, D)

        c2_exp = c2.expand(b, t, d)
        return h + c2_exp * lap

    def diffusion_strength(self, h: torch.Tensor) -> torch.Tensor:
        """Mean c^2 across the batch/sequence (diagnostic)."""
        with torch.no_grad():
            c2 = torch.sigmoid(self.coherence(h)) * self.c_max
        return c2.mean()


class DTFBlock(nn.Module):
    """Pre-norm DTF block: DTF diffusion + feed-forward with residuals."""

    def __init__(self, d_model: int, d_ff: int, c_max: float = 0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.dtf = DTFOperator(d_model, c_max=c_max)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dtf(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class DTFOperatorV2(nn.Module):
    """Relativistic Transformation Field v2.

    Combines the local graph-diffusion of DTFOperator with a *global*
    coherence-mixing term. The global term is a learned, causal, decay-weighted
    mixing of all positions (the "long-range coherence field" from RALE.docx):
    positions interact with exponentially-decaying strength as a function of
    distance, gated by a learned per-token mixing rate beta derived from the
    coherence signal. This gives R2IE the long-range sequence-mixing that pure
    local diffusion lacks, while keeping the relativistic diffusion identity.

    h_out = h + c^2 * Laplacian(h) + beta * Mix(h)

    where Mix(h)_i = sum_{j<=i} gamma^(i-j) * (score_{i,j} * v_j) and
    score_{i,j} = q_i . k_j / sqrt(d). gamma is a learned per-layer decay in
    (0,1). beta is a learned scalar gate in [0, beta_max].
    """

    def __init__(
        self,
        d_model: int,
        c_max: float = 0.5,
        beta_max: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.c_max = c_max
        self.beta_max = beta_max
        self.eps = eps
        # Local diffusion coefficient (per token).
        self.coherence = nn.Linear(d_model, 1, bias=True)
        # Global mixing projections.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.gamma_log = nn.Parameter(torch.log(0.99 * torch.ones(1)))
        # Global mixing gate beta (per token), bounded in [0, beta_max].
        self.beta_gate = nn.Linear(d_model, 1, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, t, d = h.shape
        # --- local relativistic diffusion ---
        c2 = torch.sigmoid(self.coherence(h)) * self.c_max  # (B, T, 1)
        if t == 1:
            lap = torch.zeros_like(h)
        else:
            prev = torch.cat([h[:, :1, :], h[:, :-1, :]], dim=1)
            nxt = torch.cat([h[:, 1:, :], h[:, -1:, :]], dim=1)
            lap = prev + nxt - 2.0 * h
        h_local = h + (c2.expand(b, t, d)) * lap

        # --- global coherence mixing (causal decay-weighted) ---
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        g = torch.exp(self.gamma_log).clamp(max=0.999)  # scalar decay
        idxs = torch.arange(t, device=h.device)
        dec = (idxs.unsqueeze(1) - idxs.unsqueeze(0)).clamp(min=0)
        mask = g ** dec  # (T, T) lower-triangular
        scores = torch.einsum("bid,bjd,ij->bij", q, k, mask) / (d ** 0.5)
        mixed = torch.einsum("bij,bjd->bid", scores, v)  # (B, T, d)
        mixed = self.out_proj(mixed)
        beta = torch.sigmoid(self.beta_gate(h)) * self.beta_max  # (B, T, 1)
        h_out = h_local + (beta.expand(b, t, d)) * mixed
        return h_out


class DTFBlockV2(nn.Module):
    """Pre-norm DTF v2 block: diffusion + global mixing + feed-forward."""

    def __init__(self, d_model: int, d_ff: int, c_max: float = 0.5, beta_max: float = 1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.dtf = DTFOperatorV2(d_model, c_max=c_max, beta_max=beta_max)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dtf(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class DTFOperatorV3(nn.Module):
    """Relativistic Transformation Field v3 - coherence-driven global mixer.

    This is the primary sequence mixer for R2IE. It uses the proven causal
    decay-weighted linear-attention (retention) structure, but makes BOTH the
    decay and the mixing gate DATA-DEPENDENT via the coherence signal, so each
    token decides how much long-range context to pull in - the "relativistic"
    twist over a fixed scalar-decay SSA.

    h_out = h + beta(h) * Mix(h; gamma(h))

    where Mix is causal decay-weighted attention with a per-token decay
    gamma(i,j) = sigmoid(gamma_net(h_i)) ** (i-j), and beta is a learned
    per-token gate in [0, beta_max]. Local diffusion is included as a small,
    fixed residual (c_max small) for stability.
    """

    def __init__(self, d_model: int, c_max: float = 0.1, beta_max: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.c_max = c_max
        self.beta_max = beta_max
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        # Per-token decay logit (one scalar per token); sigmoid -> (0,1).
        self.gamma_net = nn.Linear(d_model, 1, bias=True)
        # Per-token mixing gate beta in [0, beta_max].
        self.beta_gate = nn.Linear(d_model, 1, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        # Small fixed local diffusion for stability.
        self.coherence = nn.Linear(d_model, 1, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, t, d = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        # Per-token decay gamma in (0,1); build causal decay matrix.
        gamma = torch.sigmoid(self.gamma_net(h))  # (B, T, 1)
        idxs = torch.arange(t, device=h.device)
        dec = (idxs.unsqueeze(1) - idxs.unsqueeze(0)).clamp(min=0)  # (T, T)
        # gamma^(i-j): broadcast per-row decay. (B, T, 1) ** (T, T) -> (B, T, T)
        g = gamma.expand(b, t, t) ** dec.unsqueeze(0)
        scores = torch.einsum("bid,bjd,bij->bij", q, k, g) / (d ** 0.5)
        mixed = torch.einsum("bij,bjd->bid", scores, v)  # (B, T, d)
        mixed = self.out_proj(mixed)
        beta = torch.sigmoid(self.beta_gate(h)) * self.beta_max  # (B, T, 1)
        out = h + (beta.expand(b, t, d)) * mixed
        # Small local diffusion residual.
        if t > 1:
            prev = torch.cat([h[:, :1, :], h[:, :-1, :]], dim=1)
            nxt = torch.cat([h[:, 1:, :], h[:, -1:, :]], dim=1)
            lap = prev + nxt - 2.0 * h
            c2 = torch.sigmoid(self.coherence(h)) * self.c_max
            out = out + (c2.expand(b, t, d)) * lap
        return out


class DTFBlockV3(nn.Module):
    """Pre-norm DTF v3 block: coherence-driven global mixer + feed-forward."""

    def __init__(self, d_model: int, d_ff: int, c_max: float = 0.1, beta_max: float = 1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.dtf = DTFOperatorV3(d_model, c_max=c_max, beta_max=beta_max)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dtf(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


def build_dtf_stacks(
    x: torch.Tensor,
    blocks: nn.ModuleList,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply a stack of DTF blocks to x. attn_mask is accepted for API parity
    with the ACT field but is unused by diffusion (which is local/causal-safe)."""
    out = x
    for blk in blocks:
        out = blk(out)
    return out
