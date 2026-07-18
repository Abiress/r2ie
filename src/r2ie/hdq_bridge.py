"""Bridge between Python and the C++ HDQ mass-memory manager.

Uses the compiled `r2ie._r2ie_cpp.HDQMemory` (the docx's double-buffered,
lock-free mass buffer) when available; otherwise falls back to a pure-Python
`MassState` so the package still works without a C++ toolchain.

The bridge exposes a small, engine-agnostic interface:
    apply(delta)  - accrete a (dim,) vector, swap buffers (C++ backed)
    norm()        - current active mass norm (for Event-Horizon gate)
    reset()       - zero the mass
    modulate(x)   - add the active mass as a residual to hidden state x
"""

import torch

try:  # C++ backend (preferred)
    from r2ie._r2ie_cpp import HDQMemory as _CppHDQMemory

    CPP_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    _CppHDQMemory = None
    CPP_AVAILABLE = False


class _PurePythonHDQ:
    """Pure-Python fallback matching the C++ manager's bounded behavior."""

    def __init__(self, dim: int, decay: float = 0.95, clamp: float = 5.0):
        self.dim = dim
        self.decay = decay
        self.clamp = clamp
        self.active = torch.zeros(dim)
        self.background = torch.zeros(dim)

    @torch.no_grad()
    def apply(self, delta) -> None:
        self.background.copy_(self.active)
        self.background.mul_(self.decay)
        self.background.add_(delta)
        self.background.clamp_(min=-self.clamp, max=self.clamp)
        self.active, self.background = self.background, self.active

    def weights(self) -> torch.Tensor:
        return self.active.clone()

    def norm(self) -> float:
        return float(self.active.norm().item())

    def reset(self) -> None:
        with torch.no_grad():
            self.active.zero_()
            self.background.zero_()


class HDQBridge:
    """Unified HDQ mass buffer (C++ when built, else pure Python)."""

    def __init__(self, dim: int, decay: float = 0.95, clamp: float = 5.0, use_cpp: bool = True):
        self.dim = dim
        self.using_cpp = CPP_AVAILABLE and use_cpp
        if self.using_cpp:
            self._cpp = _CppHDQMemory(dim, decay, clamp)
        else:
            self._py = _PurePythonHDQ(dim, decay, clamp)

    @torch.no_grad()
    def apply(self, delta: torch.Tensor) -> None:
        d = delta.reshape(-1, self.dim).mean(dim=0).detach().cpu().tolist()
        if self.using_cpp:
            self._cpp.apply_accretion(d)
        else:
            self._py.apply(torch.tensor(d))

    def norm(self) -> float:
        if self.using_cpp:
            return float(self._cpp.active_norm())
        return self._py.norm()

    def reset(self) -> None:
        if self.using_cpp:
            self._cpp.reset()
        else:
            self._py.reset()

    def modulate(self, x: torch.Tensor) -> torch.Tensor:
        if self.using_cpp:
            w = torch.tensor(self._cpp.weights(), dtype=x.dtype, device=x.device)
        else:
            w = self._py.weights().to(x.dtype).to(x.device)
        return x + w
