"""Metrics: perplexity and ACT ponder-cost aggregation."""

import torch


def perplexity(loss: torch.Tensor) -> float:
    """Perplexity from a mean cross-entropy (nats) loss."""
    return float(torch.exp(loss).item())


def average_ponder_steps(steps: torch.Tensor) -> float:
    """Mean ponder steps over the batch/sequence."""
    return float(steps.float().mean().item())


def total_ponder_cost(ponder_costs: list[torch.Tensor]) -> float:
    """Sum of per-layer ACT ponder costs."""
    if not ponder_costs:
        return 0.0
    return float(sum(p.item() for p in ponder_costs))
