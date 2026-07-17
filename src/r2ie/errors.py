"""Typed exceptions for the R2IE prototype.

Used by the engine and CLI entry points so failures are explicit and catchable
rather than surfacing as raw ValueError / KeyError.
"""


class R2IEError(Exception):
    """Base class for all R2IE errors."""


class CorpusNotFoundError(R2IEError):
    """Raised when a requested corpus source cannot be located."""


class CheckpointMismatchError(R2IEError):
    """Raised when a checkpoint's vocab/size does not match the model config."""


class ConfigError(R2IEError):
    """Raised on invalid model or training configuration."""


class EmptyPromptError(R2IEError):
    """Raised when a generation prompt encodes to zero tokens."""
