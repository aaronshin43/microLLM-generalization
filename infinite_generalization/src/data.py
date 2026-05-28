"""Synthetic data generation for token-presence detection."""

from __future__ import annotations

import torch

from config import TaskConfig


def diagnostic_slice_specs(length: int) -> list[tuple[str, str, int, str]]:
    """Return controlled diagnostic slice definitions for a sequence length."""

    return [
        ("negative_zero_target", "negative", 0, "random"),
        ("positive_exactly_one_random", "positive", 1, "random"),
        ("positive_multi_target_k3", "positive", min(3, length), "random"),
        ("positive_multi_target_k10", "positive", min(10, length), "random"),
        (
            "positive_multi_target_density_1pct",
            "positive",
            min(max(2, length // 100), length),
            "random",
        ),
        ("positive_target_begin", "positive", 1, "begin"),
        ("positive_target_middle", "positive", 1, "middle"),
        ("positive_target_end", "positive", 1, "end"),
    ]


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


def make_negative_dataset(
    *,
    num_examples: int,
    length: int,
    task: TaskConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a zero-target negative-only diagnostic dataset."""

    if num_examples < 1:
        raise ValueError("num_examples must be positive")
    if length < 1:
        raise ValueError("length must be positive")

    inputs = _sample_non_target_tokens(
        (num_examples, length),
        vocab_size=task.vocab_size,
        target_token=task.target_token,
        generator=generator,
    )
    labels = torch.zeros(num_examples, dtype=torch.float32)
    return inputs, labels


def make_positive_dataset(
    *,
    num_examples: int,
    length: int,
    task: TaskConfig,
    generator: torch.Generator,
    target_count: int = 1,
    target_region: str = "random",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a positive-only diagnostic dataset with controlled target placement.

    `target_region` may be `random`, `begin`, `middle`, or `end`.
    """

    if num_examples < 1:
        raise ValueError("num_examples must be positive")
    if length < 1:
        raise ValueError("length must be positive")
    if target_count < 1:
        raise ValueError("target_count must be positive")
    if target_count > length:
        raise ValueError("target_count cannot exceed sequence length")

    inputs = _sample_non_target_tokens(
        (num_examples, length),
        vocab_size=task.vocab_size,
        target_token=task.target_token,
        generator=generator,
    )

    positions = _sample_target_positions(
        num_examples=num_examples,
        length=length,
        target_count=target_count,
        region=target_region,
        generator=generator,
    )
    inputs[
        torch.arange(num_examples).unsqueeze(1),
        positions,
    ] = task.target_token

    labels = torch.ones(num_examples, dtype=torch.float32)
    return inputs, labels


def _sample_target_positions(
    *,
    num_examples: int,
    length: int,
    target_count: int,
    region: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample target positions for positive diagnostic datasets."""

    if region == "random":
        candidates = torch.arange(length)
    elif region == "begin":
        candidates = torch.arange(0, max(1, length // 3))
    elif region == "middle":
        start = length // 3
        end = max(start + 1, (2 * length) // 3)
        candidates = torch.arange(start, end)
    elif region == "end":
        candidates = torch.arange((2 * length) // 3, length)
    else:
        raise ValueError(f"unknown target_region: {region}")

    if target_count > len(candidates):
        raise ValueError(
            f"target_count={target_count} exceeds available positions in region={region}"
        )

    rows = []
    for _ in range(num_examples):
        permutation = torch.randperm(len(candidates), generator=generator)
        rows.append(candidates[permutation[:target_count]])
    return torch.stack(rows, dim=0)
