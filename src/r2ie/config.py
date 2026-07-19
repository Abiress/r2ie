"""Configuration dataclasses for the R2IE model and training."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model hyperparameters for the R2IE architecture."""

    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    max_seq_len: int = 256

    # VQ "Mass Compressor" (van den Oord et al., 2017)
    codebook_size: int = 128
    commitment_cost: float = 0.25
    vq_mode: str = "hard"  # "hard" (true bottleneck) or "soft" (info-preserving)

    # ACT "Transformation Field" (Graves, 2016)
    max_ponder_steps: int = 4
    act_epsilon: float = 0.01
    act_ponder_weight: float = 0.01

    # Fast-weight "Condensation Loop" (Ba et al., 2016)
    # The fast-weight matrix is square (d_model x d_model) and modulates the
    # output head features, so it always matches d_model.
    fast_weight_decay: float = 0.9
    fast_weight_lr: float = 0.1
    fast_weight_clamp: float = 5.0

    # DTF Transformation Field (RALE.docx Loop 2): trainable graph-diffusion
    # applied over token positions inside each transformation-field layer.
    use_dtf: bool = False
    dtf_v2: bool = False
    dtf_v3: bool = False
    dtf_c_max: float = 0.5
    dtf_d_ff: int = 256
    dtf_beta_max: float = 1.0
    # When True the ACT multi-head attention field is skipped and the DTF
    # graph-diffusion blocks act as the sole transformation field. Used to let
    # the global DTFv3 coherence mixer run without a competing attention mixer.
    act_attention: bool = True

    # HDQ Mass Memory (RALE.docx Loop 1): double-buffered C++ mass manager used
    # by the Condensation Loop. Falls back to pure Python if C++ ext missing.
    use_hdq: bool = False
    hdq_decay: float = 0.95
    hdq_clamp: float = 5.0

    # CPL output head
    temperature: float = 1.0


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    batch_size: int = 32
    seq_len: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    steps: int = 2000
    log_every: int = 50
    seed: int = 1234
    device: str = "auto"
    data: str = "tinyshakespeare"
    checkpoint_dir: str = "checkpoints"
    runs_dir: str = "runs"
