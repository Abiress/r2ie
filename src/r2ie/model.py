"""R2IEModel - assembles the full R2IE architecture.

Data flow (matches the README diagram):
    token -> embedding -> VQ compressor (Mass Compressor)
         -> stack of ACT-wrapped transformer blocks (Transformation Field)
         -> fast-weight-modulated output head (CPL) -> logits
"""

from typing import Tuple

import torch
import torch.nn as nn

from .act_field import ACTField
from .config import ModelConfig
from .dtf_field import DTFBlock, DTFBlockV2, DTFBlockV3
from .fast_weight_loop import FastWeightMemory
from .output_head import OutputHead
from .vq_compressor import VQCompressor


class R2IEModel(nn.Module):
    """The Recursive Relativistic Information Engine (prototype)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.vq = VQCompressor(
            config.d_model, config.codebook_size, config.commitment_cost,
            mode=getattr(config, "vq_mode", "hard"),
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_seq_len, config.d_model)
        )
        self.act_field = ACTField(
            config.d_model,
            config.n_heads,
            config.d_ff,
            n_layers=config.n_layers,
            max_ponder_steps=config.max_ponder_steps,
            act_epsilon=config.act_epsilon,
            act_ponder_weight=config.act_ponder_weight,
        )
        self.head = OutputHead(config.d_model, config.vocab_size, config.temperature)
        self.fast_memory = FastWeightMemory(
            config.d_model,
            decay=config.fast_weight_decay,
            lr_fast=config.fast_weight_lr,
            clamp=config.fast_weight_clamp,
        )

        # DTF Transformation Field (RALE.docx Loop 2): a stack of trainable
        # graph-diffusion blocks applied over token positions. Inserted after
        # the ACT-wrapped transformer blocks so diffusion mixes the attended
        # representations. Disabled by default (config.use_dtf).
        self.use_dtf = config.use_dtf
        if self.use_dtf:
            use_v3 = getattr(config, "dtf_v3", False)
            use_v2 = getattr(config, "dtf_v2", False)
            if use_v3:
                blk = DTFBlockV3
            elif use_v2:
                blk = DTFBlockV2
            else:
                blk = DTFBlock
            self.dtf_blocks = nn.ModuleList(
                [
                    blk(
                        config.d_model,
                        config.dtf_d_ff,
                        c_max=config.dtf_c_max,
                        **({"beta_max": getattr(config, "dtf_beta_max", 1.0)} if (use_v2 or use_v3) else {}),
                    )
                    for _ in range(config.n_layers)
                ]
            )
        else:
            self.dtf_blocks = None

        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        use_fast_weights: bool = False,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass.

        Args:
            idx: (B, T) token ids.
            use_fast_weights: if True, apply (and update) the fast-weight memory
                to modulate the head. Inference/consolidation only.
            return_aux: if True, return a dict of auxiliary losses / diagnostics.

        Returns:
            logits: (B, T, vocab_size).
            aux: dict with "commitment_loss", "ponder_cost", "ponder_steps"
                 (present when return_aux=True).
        """
        b, t = idx.shape
        x = self.embedding(idx) + self.pos_embed[:, :t, :]

        # Mass Compressor: VQ bottleneck.
        x, commitment_loss, _ = self.vq(x)  # commitment_loss already includes VQ codebook term

        # Transformation Field: ACT-wrapped transformer blocks.
        # Causal mask so generation stays autoregressive.
        ponder_cost = torch.zeros((), device=x.device)
        steps = 1
        if getattr(self.config, "act_attention", True):
            causal = torch.triu(torch.ones(t, t, device=idx.device), diagonal=1).bool()
            x, ponder_cost, steps = self.act_field(x, attn_mask=causal)

        # DTF Transformation Field: learned graph-diffusion over positions.
        if self.use_dtf and self.dtf_blocks is not None:
            for blk in self.dtf_blocks:
                x = blk(x)

        fast_modulation = None
        if use_fast_weights:
            # Build a (query, value) pair from the hidden state and update the
            # fast-weight memory, then modulate the head. Never mutates the
            # trained weights.
            self.fast_memory.apply(x, x)
            fast_modulation = self.fast_memory.modulate(x)

        logits = self.head(x, fast_modulation=fast_modulation)

        aux = {}
        if return_aux:
            aux = {
                "commitment_loss": commitment_loss,
                "ponder_cost": ponder_cost,
                "ponder_steps": steps,
                "hidden": x,
            }
        return logits, aux

    def reset_fast_memory(self) -> None:
        self.fast_memory.reset()
