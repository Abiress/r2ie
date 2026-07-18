"""R2IEEngine - the modular processing pipeline.

Composes the base R2IEModel with the optional Condensation Loop and Velocity
Governor (your original RALE.docx loops). Provides a single forward_step() used
by both training and generation, input validation, and explicit buffer lifecycle
so fast-weight / condensation state never leaks across sequences.

Both optional loops are OFF by default - enabling them does not change the
base model's trained weights, only modulates the forward pass.
"""

from typing import Tuple

import torch

from .condensation import make_condensation_loop
from .config import ModelConfig
from .errors import CheckpointMismatchError, EmptyPromptError
from .model import R2IEModel
from .velocity_governor import VelocityGovernor


class R2IEEngine:
    """Owns the model and optional loop modules, exposes a clean pipeline."""

    def __init__(
        self,
        config: ModelConfig,
        use_condensation: bool = False,
        use_governor: bool = False,
        use_fast_weights: bool = False,
        use_hdq: bool = False,
        use_dtf: bool = False,
        use_dtf_v2: bool = False,
        use_dtf_v3: bool = False,
        vq_mode: str = "hard",
    ):
        self.config = config
        # Propagate architecture flags into the config consumed by R2IEModel.
        config.use_dtf = use_dtf
        config.use_dtf_v2 = use_dtf_v2
        config.use_dtf_v3 = use_dtf_v3
        config.use_hdq = use_hdq
        config.vq_mode = vq_mode
        # When DTFv3 is the primary mixer, reduce the ACT field to a single
        # pass so the two global mixers don't compete for gradient signal.
        if use_dtf_v3:
            config.max_ponder_steps = 1
        self.model = R2IEModel(config)
        self.use_condensation = use_condensation
        self.use_governor = use_governor
        self.use_fast_weights = use_fast_weights
        self.use_hdq = use_hdq
        self.use_dtf = use_dtf
        self.use_dtf_v2 = use_dtf_v2
        self.use_dtf_v3 = use_dtf_v3
        self.vq_mode = vq_mode

        self.condensation = (
            make_condensation_loop(
                config.d_model,
                use_hdq=use_hdq,
                decay=config.hdq_decay,
                clamp=config.hdq_clamp,
            )
            if use_condensation
            else None
        )
        self.governor = VelocityGovernor() if use_governor else None

    # ----- validation ---------------------------------------------------
    def validate_prompt(self, prompt_ids: torch.Tensor) -> None:
        if prompt_ids.numel() == 0:
            raise EmptyPromptError("Prompt encoded to zero tokens.")

    def validate_checkpoint(self, ckpt_config: ModelConfig) -> None:
        if ckpt_config.vocab_size != self.config.vocab_size:
            raise CheckpointMismatchError(
                f"checkpoint vocab {ckpt_config.vocab_size} != model vocab "
                f"{self.config.vocab_size}"
            )

    # ----- lifecycle ----------------------------------------------------
    def reset_state(self) -> None:
        """Clear all non-trainable buffers between sequences / calls."""
        self.model.reset_fast_memory()
        if self.condensation is not None:
            self.condensation.reset()

    def to(self, device) -> "R2IEEngine":
        """Move all submodules to the given device (engine is not nn.Module)."""
        self.model.to(device)
        if self.condensation is not None:
            self.condensation.to(device)
        if self.governor is not None:
            self.governor.to(device)
        return self

    # ----- forward ------------------------------------------------------
    def forward_step(
        self, idx: torch.Tensor, consolidate: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """One forward pass through the pipeline.

        Args:
            idx: (B, T) token ids.
            consolidate: if True and condensation is enabled, accrete the
                output energy into the mass buffer after the pass.

        Returns:
            logits: (B, T, vocab_size).
            aux: diagnostics dict (commitment, ponder, steps, governor, mass).
        """
        logits, aux = self.model(
            idx,
            use_fast_weights=self.use_fast_weights,
            return_aux=True,
        )

        if self.governor is not None:
            bias = self.governor.halting_bias(logits)
            aux["governor_bias_mean"] = bias.mean()
            aux["coherence_mean"] = self.governor.coherence(logits).mean()

        if self.condensation is not None:
            aux["mass_norm"] = self.condensation.mass_norm
            if consolidate:
                c_squared = max(aux["ponder_steps"].float().mean().item(), 1.0)
                # Output energy = the pre-head hidden state (d_model), the
                # "residual energy" of the current thought (RALE.docx Loop 1).
                energy = aux["hidden"].detach()
                self.condensation.consolidate(energy, c_squared=c_squared)
                aux["mass_norm"] = self.condensation.mass_norm

        return logits, aux

    # ----- save / load --------------------------------------------------
    def save(self, path: str) -> None:
        from dataclasses import asdict

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": asdict(self.config),
                "use_condensation": self.use_condensation,
                "use_governor": self.use_governor,
                "use_fast_weights": self.use_fast_weights,
                "use_hdq": self.use_hdq,
                "use_dtf": self.use_dtf,
                "vq_mode": self.vq_mode,
            },
            path,
        )

    def load(self, path: str, device: str = "cpu") -> None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.validate_checkpoint(ModelConfig(**ckpt["config"]))
        self.model.load_state_dict(ckpt["model_state"])
        self.use_condensation = ckpt.get("use_condensation", False)
        self.use_governor = ckpt.get("use_governor", False)
        self.use_fast_weights = ckpt.get("use_fast_weights", False)
        self.use_hdq = ckpt.get("use_hdq", False)
        self.use_dtf = ckpt.get("use_dtf", False)
        self.vq_mode = ckpt.get("vq_mode", "hard")
