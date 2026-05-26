"""Synthetic data generation for token-presence detection."""

from __future__ import annotations

import torch

from config import TaskConfig


def _sample_non_target_tokens(
    shape: tuple[int, ...],
    *,
    vocab_size: int,
    target_token: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample token IDs uniformly from the vocabulary excluding the target token."""

    if not 0 <= target_token < vocab_size:
        raise ValueError(f"target_token must be in [0, {vocab_size}), got {target_token}")

    tokens = torch.randint(0, vocab_size - 1, shape, generator=generator)
    return tokens + (tokens >= target_token).long()


def make_balanced_token_presence_dataset(
    *,
    num_examples: int,
    length: int,
    task: TaskConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a balanced dataset with zero-target negatives and exactly-one positives.

    Positive examples contain exactly one target token. This is intentionally the sparse
    positive case, because it exposes length-dependent signal dilution most clearly.
    """

    if num_examples < 2:
        raise ValueError("num_examples must be at least 2 for a balanced dataset")
    if length < 1:
        raise ValueError("length must be positive")

    num_positive = num_examples // 2
    num_negative = num_examples - num_positive

    negatives = _sample_non_target_tokens(
        (num_negative, length),
        vocab_size=task.vocab_size,
        target_token=task.target_token,
        generator=generator,
    )

    positives = _sample_non_target_tokens(
        (num_positive, length),
        vocab_size=task.vocab_size,
        target_token=task.target_token,
        generator=generator,
    )
    target_positions = torch.randint(0, length, (num_positive,), generator=generator)
    positives[torch.arange(num_positive), target_positions] = task.target_token

    inputs = torch.cat([negatives, positives], dim=0)
    labels = torch.cat(
        [
            torch.zeros(num_negative, dtype=torch.float32),
            torch.ones(num_positive, dtype=torch.float32),
        ],
        dim=0,
    )

    # Shuffle after construction so each mini-batch sees both classes in expectation.
    permutation = torch.randperm(num_examples, generator=generator)
    return inputs[permutation], labels[permutation]
