"""Evaluation metrics for binary token-presence classification."""

from __future__ import annotations

import torch


@torch.no_grad()
def binary_accuracy_slices(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Compute overall, positive-class, and negative-class accuracy from logits."""

    predictions = (logits >= 0).float()
    correct = predictions.eq(labels)

    positive_mask = labels.eq(1)
    negative_mask = labels.eq(0)

    metrics = {"overall_accuracy": correct.float().mean().item()}
    metrics["positive_accuracy"] = (
        correct[positive_mask].float().mean().item() if positive_mask.any() else float("nan")
    )
    metrics["negative_accuracy"] = (
        correct[negative_mask].float().mean().item() if negative_mask.any() else float("nan")
    )
    return metrics
