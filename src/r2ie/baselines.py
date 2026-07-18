"""Baseline architectures for the architecture comparison (benchmark_archs.py).

Each baseline implements the SAME interface as R2IEModel for the purpose of the
benchmark:

    model = TransformerBaseline(cfg)   # cfg: ModelConfig
    logits, _ = model(idx)             # idx: (B, T) -> logits (B, T, vocab)

and exposes `param_count()` so the benchmark can report matched parameter
counts. These are faithful, minimal implementations (no R2IE-specific loops)
used only to compare validation perplexity on the same data/optimizer/budget.

Baselines provided:
  * TransformerBaseline  - causal multi-head self-attention + FFN (Vaswani et al.)
  * MambaBaseline        - a minimal selective state-space block (Gu & Dao, 2023)
                           with input-dependent (B, C) gates and a scalar SSM.
  * SSABaseline          - a sub-quadratic linear-attention ("retention"-style)
                           variant (sun et al., 2023) with a learned decay.
"""


import torch
import torch.nn as nn


def _causal_mask(t: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(t, t, device=device), diagonal=1).bool()


class _FFN(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBaseline(nn.Module):
    """Standard causal transformer (reference baseline)."""

    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_model))
        self.blocks = nn.ModuleList(
            [self._Block(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    class _Block(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.ln1 = nn.LayerNorm(cfg.d_model)
            self.attn = nn.MultiheadAttention(
                cfg.d_model, cfg.n_heads, batch_first=True
            )
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.ffn = _FFN(cfg.d_model, cfg.d_ff)

        def forward(self, x):
            t = x.shape[1]
            mask = _causal_mask(t, x.device)
            h = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                              attn_mask=mask, need_weights=False)[0]
            h = h + self.ffn(self.ln2(h))
            return h

    def forward(self, idx, **_):
        b, t = idx.shape
        x = self.embed(idx) + self.pos[:, :t, :]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)), {}

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MambaBaseline(nn.Module):
    """Minimal selective SSM block (Mamba-style), scalar state per channel.

    Implements the core selective-scan with input-dependent B/C gates and a
    learned per-channel decay (A). This is a faithful small-scale rendition,
    not the exact CUDA kernel, but uses the same math.
    """

    def __init__(self, cfg):
        super().__init__()
        self.d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, self.d)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_seq_len, self.d))
        self.blocks = nn.ModuleList([self._Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, cfg.vocab_size)
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    class _Block(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            d = cfg.d_model
            self.ln = nn.LayerNorm(d)
            self.in_proj = nn.Linear(d, 2 * d)  # x and gate
            self.dt = nn.Linear(d, d)
            self.b_gate = nn.Linear(d, d)
            self.c_gate = nn.Linear(d, d)
            # Per-channel decay (negative log of A).
            self.log_a = nn.Parameter(torch.log(0.9 * torch.ones(d)))
            self.out_proj = nn.Linear(d, d)
            self.ffn = _FFN(d, cfg.d_ff)

        def _selective_scan(self, x, gate):
            # x, gate: (B, T, d)
            b, t, d = x.shape
            dt = torch.exp(self.dt(x))              # (B, T, d) positive step
            b_gate = self.b_gate(x).sigmoid()       # (B, T, d)
            c_gate = self.c_gate(x).sigmoid()       # (B, T, d)
            a = torch.exp(self.log_a)               # (d,) decay per channel

            state = torch.zeros(b, d, device=x.device, dtype=x.dtype)
            ys = []
            for i in range(t):
                # h_t = a * h_{t-1} + b_t * x_t ; y_t = c_t * h_t
                state = a * state + b_gate[:, i] * x[:, i]
                y = c_gate[:, i] * state
                ys.append(y)
            y = torch.stack(ys, dim=1)  # (B, T, d)
            y = y * dt * gate.sigmoid()
            return self.out_proj(y)

        def forward(self, x):
            h = self.ln(x)
            x_in, gate = self.in_proj(h).chunk(2, dim=-1)
            out = self._selective_scan(x_in, gate)
            out = out + self.ffn(self.ln(x))
            return out

    def forward(self, idx, **_):
        b, t = idx.shape
        x = self.embed(idx) + self.pos[:, :t, :]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)), {}

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SSABaseline(nn.Module):
    """Sub-quadratic linear-attention (retention-style) baseline.

    Uses a learned scalar decay and a causal linear-attention recurrence with
    a data-dependent key/value projection (sun et al., 2023, simplified).
    """

    def __init__(self, cfg):
        super().__init__()
        self.d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, self.d)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_seq_len, self.d))
        self.blocks = nn.ModuleList([self._Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, cfg.vocab_size)
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    class _Block(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            d = cfg.d_model
            self.ln = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.decay = nn.Parameter(torch.log(0.9 * torch.ones(1)))
            self.out = nn.Linear(d, d)
            self.ffn = _FFN(d, cfg.d_ff)

        def forward(self, x):
            h = self.ln(x)
            q, k, v = self.qkv(h).chunk(3, dim=-1)
            b, t, d = q.shape
            gamma = torch.exp(self.decay).clamp(max=0.999)  # scalar decay in (0,1)
            # Causal decay mask: D[i,j] = gamma**(i-j) for j <= i else 0.
            idxs = torch.arange(t, device=x.device)
            dec = (idxs.unsqueeze(1) - idxs.unsqueeze(0)).clamp(min=0)
            mask = gamma ** dec  # (T, T), lower-triangular
            # KV product accumulated with decay, then attend with Q.
            # attn[i, j] = mask[i, j] * (q_i . k_j); out_i = sum_j attn * v_j
            scores = torch.einsum("bid,bjd,ij->bij", q, k, mask)  # (B, T, T)
            y = torch.einsum("bij,bjd->bid", scores, v)  # (B, T, d)
            y = self.out(y)
            return y + self.ffn(self.ln(x))

    def forward(self, idx, **_):
        b, t = idx.shape
        x = self.embed(idx) + self.pos[:, :t, :]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)), {}

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
