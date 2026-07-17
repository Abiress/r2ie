"""R2IE - Recursive Relativistic Information Engine (prototype).

An experimental, small-scale research/education architecture combining a
vector-quantization (VQ) bottleneck, adaptive per-token compute (ACT),
and bounded fast-weight adaptation.
"""

from .config import ModelConfig, TrainConfig

__version__ = "0.1.0"
__all__ = ["R2IEModel", "ModelConfig", "TrainConfig"]


def __getattr__(name: str):
    # Lazy import so the package imports even before all modules are written.
    if name == "R2IEModel":
        from .model import R2IEModel

        return R2IEModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
