"""Train Stage 3: empirical simplified length-aware attention model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


TARGET_TOKEN_ID = 0
NON_TARGET_TOKEN_ID = 1
TARGET_POSITION = 0


@dataclass(frozen=True)
class Stage3Config:
    """Configuration for the simplified attention validation experiment."""

    seed: int = 42
    device: str = "auto"
    output_dir: str = "runs/stage3_simplified_attention"
    alpha_mode: str = "constant"
    train_length: int = 10
    train_lengths: tuple[int, ...] = ()
    target_position_mode: str = "fixed_start"
    target_token_count: int = 1
    non_target_token_count: int = 1
    non_target_sampling: str = "uniform"
    train_examples: int = 2_000
    val_examples: int = 500
    test_examples: int = 50
    eval_chunk_examples: int = 50
    eval_sampling_mode: str = "random"
    eval_lengths: tuple[int, ...] = (10, 100, 1000, 10000, 100000, 1000000, 5000000, 10000000)
    batch_size: int = 64
    eval_batch_size: int = 16
    epochs: int = 200
    max_train_steps: int | None = None
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    d_head: int = 2
    alpha_log_scale_init: float = -5.0


def project_dir() -> Path:
    """Return the infinite_generalization project directory."""

    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=Stage3Config.seed)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=Stage3Config.device)
    parser.add_argument("--output-dir", type=str, default=Stage3Config.output_dir)
    parser.add_argument(
        "--alpha-mode",
        choices=("constant", "log", "learned_log"),
        default=Stage3Config.alpha_mode,
    )
    parser.add_argument("--train-length", type=int, default=Stage3Config.train_length)
    parser.add_argument(
        "--train-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Training lengths. If omitted, --train-length is used.",
    )
    parser.add_argument(
        "--target-position-mode",
        choices=("fixed_start", "nonfinal_random"),
        default=Stage3Config.target_position_mode,
        help="Positive target placement mode.",
    )
    parser.add_argument(
        "--target-token-count",
        type=int,
        default=Stage3Config.target_token_count,
        help="Number of target token types. Token ids are 0 through this value minus 1.",
    )
    parser.add_argument(
        "--non-target-token-count",
        type=int,
        default=Stage3Config.non_target_token_count,
        help="Number of non-target token types after the target token id range.",
    )
    parser.add_argument(
        "--non-target-sampling",
        choices=("uniform",),
        default=Stage3Config.non_target_sampling,
        help="Sampling distribution for non-target token positions.",
    )
    parser.add_argument("--train-examples", type=int, default=Stage3Config.train_examples)
    parser.add_argument("--val-examples", type=int, default=Stage3Config.val_examples)
    parser.add_argument("--test-examples", type=int, default=Stage3Config.test_examples)
    parser.add_argument(
        "--eval-chunk-examples",
        type=int,
        default=Stage3Config.eval_chunk_examples,
        help="Maximum number of evaluation examples generated at once.",
    )
    parser.add_argument(
        "--eval-sampling-mode",
        choices=("random", "stratified"),
        default=Stage3Config.eval_sampling_mode,
        help="Evaluation sampling mode.",
    )
    parser.add_argument(
        "--eval-lengths",
        type=int,
        nargs="+",
        default=list(Stage3Config.eval_lengths),
        help="Evaluation lengths, e.g. --eval-lengths 10 100 1000.",
    )
    parser.add_argument("--batch-size", type=int, default=Stage3Config.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=Stage3Config.eval_batch_size)
    parser.add_argument("--epochs", type=int, default=Stage3Config.epochs)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=Stage3Config.max_train_steps,
        help="Optional optimizer update budget. Overrides epoch-limited training when set.",
    )
    parser.add_argument("--learning-rate", type=float, default=Stage3Config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Stage3Config.weight_decay)
    parser.add_argument("--d-head", type=int, default=Stage3Config.d_head)
    parser.add_argument(
        "--alpha-log-scale-init",
        type=float,
        default=Stage3Config.alpha_log_scale_init,
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny fast configuration that verifies the pipeline.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Stage3Config:
    """Build the experiment config, with tiny overrides for smoke tests."""

    output_dir = args.output_dir
    if args.smoke_test and output_dir == Stage3Config.output_dir:
        output_dir = "runs/stage3_simplified_attention_smoke"

    train_lengths = tuple(args.train_lengths) if args.train_lengths else (args.train_length,)
    if not train_lengths:
        raise ValueError("At least one training length is required.")
    if any(length < 2 for length in train_lengths):
        raise ValueError("All training lengths must be at least 2.")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        raise ValueError("--max-train-steps must be positive when provided.")
    if args.target_position_mode not in {"fixed_start", "nonfinal_random"}:
        raise ValueError(f"Unsupported target position mode: {args.target_position_mode}")
    if args.target_token_count < 1:
        raise ValueError("--target-token-count must be at least 1.")
    if args.non_target_token_count < 1:
        raise ValueError("--non-target-token-count must be at least 1.")
    if args.non_target_sampling != "uniform":
        raise ValueError(f"Unsupported non-target sampling: {args.non_target_sampling}")
    if args.test_examples < 1:
        raise ValueError("--test-examples must be at least 1.")
    if args.eval_chunk_examples < 1:
        raise ValueError("--eval-chunk-examples must be at least 1.")
    if args.eval_sampling_mode not in {"random", "stratified"}:
        raise ValueError(f"Unsupported eval sampling mode: {args.eval_sampling_mode}")

    config = Stage3Config(
        seed=args.seed,
        device=args.device,
        output_dir=output_dir,
        alpha_mode=args.alpha_mode,
        train_length=args.train_length,
        train_lengths=train_lengths,
        target_position_mode=args.target_position_mode,
        target_token_count=args.target_token_count,
        non_target_token_count=args.non_target_token_count,
        non_target_sampling=args.non_target_sampling,
        train_examples=args.train_examples,
        val_examples=args.val_examples,
        test_examples=args.test_examples,
        eval_chunk_examples=args.eval_chunk_examples,
        eval_sampling_mode=args.eval_sampling_mode,
        eval_lengths=tuple(args.eval_lengths),
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        max_train_steps=args.max_train_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        d_head=args.d_head,
        alpha_log_scale_init=args.alpha_log_scale_init,
    )

    if not args.smoke_test:
        return config

    return Stage3Config(
        seed=config.seed,
        device=config.device,
        output_dir=config.output_dir,
        alpha_mode=config.alpha_mode,
        train_length=10,
        train_lengths=config.train_lengths,
        target_position_mode=config.target_position_mode,
        target_token_count=config.target_token_count,
        non_target_token_count=config.non_target_token_count,
        non_target_sampling=config.non_target_sampling,
        train_examples=64,
        val_examples=32,
        test_examples=config.test_examples,
        eval_chunk_examples=config.eval_chunk_examples,
        eval_sampling_mode=config.eval_sampling_mode,
        eval_lengths=(10, 20),
        batch_size=16,
        eval_batch_size=20,
        epochs=2,
        max_train_steps=config.max_train_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        d_head=config.d_head,
        alpha_log_scale_init=config.alpha_log_scale_init,
    )


def set_reproducibility(seed: int) -> None:
    """Set deterministic seeds for Python and PyTorch."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str) -> torch.device:
    """Resolve a user-facing device string."""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(name)


def first_non_target_token_id(target_token_count: int) -> int:
    """Return the first non-target token id under the contiguous id convention."""

    return target_token_count


def target_token_ids(target_token_count: int) -> list[int]:
    """Return all target token ids."""

    return list(range(target_token_count))


def non_target_token_ids(target_token_count: int, non_target_token_count: int) -> list[int]:
    """Return all non-target token ids."""

    start = first_non_target_token_id(target_token_count)
    return list(range(start, start + non_target_token_count))


def position_bucket_values(length: int) -> list[str]:
    """Return target-position buckets that contain at least one non-final position."""

    buckets: list[str] = []
    for position in range(length - 1):
        bucket = target_position_bucket(position, length)
        if bucket not in buckets:
            buckets.append(bucket)
    return buckets


def sample_position_from_bucket(
    *,
    bucket: str,
    length: int,
    generator: torch.Generator,
) -> int:
    """Sample one non-final target position from a coarse position bucket."""

    positions = [
        position
        for position in range(length - 1)
        if target_position_bucket(position, length) == bucket
    ]
    if not positions:
        raise ValueError(f"Bucket {bucket!r} has no valid positions for length {length}.")
    index = torch.randint(
        low=0,
        high=len(positions),
        size=(),
        generator=generator,
        dtype=torch.long,
    ).item()
    return positions[int(index)]


def repeated_balanced_values(values: list[Any], count: int, *, start_index: int = 0) -> list[Any]:
    """Repeat values in deterministic round-robin order to fill a requested count."""

    if count < 0:
        raise ValueError("count must be non-negative.")
    if count and not values:
        raise ValueError("values must be non-empty when count is positive.")
    return [values[(start_index + index) % len(values)] for index in range(count)]


def make_stage3_dataset_from_counts(
    *,
    length: int,
    positive_count: int,
    negative_count: int,
    seed: int,
    target_position_mode: str,
    target_token_count: int,
    non_target_token_count: int,
    non_target_sampling: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a Stage 3 dataset with explicit positive and negative counts."""

    if length < 2:
        raise ValueError("length must be at least 2 for the exactly-one-target setup.")
    if positive_count < 0 or negative_count < 0:
        raise ValueError("positive_count and negative_count must be non-negative.")
    if positive_count + negative_count < 1:
        raise ValueError("At least one example is required.")
    if target_token_count < 1:
        raise ValueError("target_token_count must be at least 1.")
    if non_target_token_count < 1:
        raise ValueError("non_target_token_count must be at least 1.")
    if non_target_sampling != "uniform":
        raise ValueError(f"Unsupported non_target_sampling: {non_target_sampling}")

    generator = torch.Generator().manual_seed(seed)

    positive_inputs = sample_non_target_tokens(
        shape=(positive_count, length),
        target_token_count=target_token_count,
        non_target_token_count=non_target_token_count,
        generator=generator,
    )
    positive_target_token_ids = sample_target_tokens(
        shape=(positive_count,),
        target_token_count=target_token_count,
        generator=generator,
    )
    if target_position_mode == "fixed_start":
        positive_target_positions = torch.full(
            (positive_count,),
            TARGET_POSITION,
            dtype=torch.long,
        )
    elif target_position_mode == "nonfinal_random":
        positive_target_positions = torch.randint(
            low=0,
            high=length - 1,
            size=(positive_count,),
            generator=generator,
            dtype=torch.long,
        )
    else:
        raise ValueError(f"Unsupported target_position_mode: {target_position_mode}")

    if positive_count:
        positive_inputs[torch.arange(positive_count), positive_target_positions] = (
            positive_target_token_ids
        )
    positive_labels = torch.ones(positive_count, dtype=torch.float32)

    negative_inputs = sample_non_target_tokens(
        shape=(negative_count, length),
        target_token_count=target_token_count,
        non_target_token_count=non_target_token_count,
        generator=generator,
    )
    negative_labels = torch.zeros(negative_count, dtype=torch.float32)
    negative_target_positions = torch.full((negative_count,), -1, dtype=torch.long)
    negative_target_token_ids = torch.full((negative_count,), -1, dtype=torch.long)

    inputs = torch.cat([positive_inputs, negative_inputs], dim=0)
    labels = torch.cat([positive_labels, negative_labels], dim=0)
    target_positions = torch.cat([positive_target_positions, negative_target_positions], dim=0)
    target_ids = torch.cat([positive_target_token_ids, negative_target_token_ids], dim=0)

    total_examples = positive_count + negative_count
    permutation = torch.randperm(total_examples, generator=generator)
    return (
        inputs[permutation],
        labels[permutation],
        target_positions[permutation],
        target_ids[permutation],
    )


def make_two_token_dataset(
    *,
    length: int,
    examples: int,
    seed: int,
    target_position_mode: str,
    target_token_count: int,
    non_target_token_count: int,
    non_target_sampling: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a balanced target-presence dataset.

    Positive examples contain exactly one target. Negative examples are all non-target.
    The token id convention is targets 0..H-1 and non-targets H..H+M-1.
    """

    if examples < 2:
        raise ValueError("examples must be at least 2.")

    positive_count = examples // 2
    negative_count = examples - positive_count
    return make_stage3_dataset_from_counts(
        length=length,
        positive_count=positive_count,
        negative_count=negative_count,
        seed=seed,
        target_position_mode=target_position_mode,
        target_token_count=target_token_count,
        non_target_token_count=non_target_token_count,
        non_target_sampling=non_target_sampling,
    )


def make_stratified_eval_dataset(
    *,
    length: int,
    positive_count: int,
    negative_count: int,
    seed: int,
    target_position_mode: str,
    target_token_count: int,
    non_target_token_count: int,
    non_target_sampling: str,
    positive_stratum_offset: int = 0,
    negative_final_query_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a stratified Stage 3 evaluation dataset.

    Positive strata use the Cartesian product of active diagnostic dimensions:
    target-position bucket, final query non-target token id, and target token id.
    Negative examples are balanced over final query token id when that dimension exists.
    """

    if length < 2:
        raise ValueError("length must be at least 2 for the exactly-one-target setup.")
    if positive_count < 0 or negative_count < 0:
        raise ValueError("positive_count and negative_count must be non-negative.")
    if positive_count + negative_count < 1:
        raise ValueError("At least one example is required.")
    if target_token_count < 1:
        raise ValueError("target_token_count must be at least 1.")
    if non_target_token_count < 1:
        raise ValueError("non_target_token_count must be at least 1.")
    if non_target_sampling != "uniform":
        raise ValueError(f"Unsupported non_target_sampling: {non_target_sampling}")
    if target_position_mode not in {"fixed_start", "nonfinal_random"}:
        raise ValueError(f"Unsupported target_position_mode: {target_position_mode}")

    generator = torch.Generator().manual_seed(seed)
    final_query_values = non_target_token_ids(target_token_count, non_target_token_count)
    target_values = target_token_ids(target_token_count)
    if target_position_mode == "nonfinal_random":
        position_values: list[str | None] = position_bucket_values(length)
    else:
        position_values = [None]

    positive_strata = [
        (position_bucket, final_query_id, target_id)
        for position_bucket in position_values
        for final_query_id in final_query_values
        for target_id in target_values
    ]
    assigned_positive_strata = repeated_balanced_values(
        positive_strata,
        positive_count,
        start_index=positive_stratum_offset,
    )
    assigned_negative_final_ids = repeated_balanced_values(
        final_query_values,
        negative_count,
        start_index=negative_final_query_offset,
    )

    positive_inputs = sample_non_target_tokens(
        shape=(positive_count, length),
        target_token_count=target_token_count,
        non_target_token_count=non_target_token_count,
        generator=generator,
    )
    positive_target_positions = torch.empty(positive_count, dtype=torch.long)
    positive_target_token_ids = torch.empty(positive_count, dtype=torch.long)
    for row_index, (position_bucket, final_query_id, target_id) in enumerate(
        assigned_positive_strata
    ):
        if target_position_mode == "fixed_start":
            target_position = TARGET_POSITION
        else:
            if position_bucket is None:
                raise RuntimeError("Stratified random target positions require a bucket.")
            target_position = sample_position_from_bucket(
                bucket=position_bucket,
                length=length,
                generator=generator,
            )
        positive_inputs[row_index, -1] = final_query_id
        positive_inputs[row_index, target_position] = target_id
        positive_target_positions[row_index] = target_position
        positive_target_token_ids[row_index] = target_id
    positive_labels = torch.ones(positive_count, dtype=torch.float32)

    negative_inputs = sample_non_target_tokens(
        shape=(negative_count, length),
        target_token_count=target_token_count,
        non_target_token_count=non_target_token_count,
        generator=generator,
    )
    for row_index, final_query_id in enumerate(assigned_negative_final_ids):
        negative_inputs[row_index, -1] = final_query_id
    negative_labels = torch.zeros(negative_count, dtype=torch.float32)
    negative_target_positions = torch.full((negative_count,), -1, dtype=torch.long)
    negative_target_token_ids = torch.full((negative_count,), -1, dtype=torch.long)

    inputs = torch.cat([positive_inputs, negative_inputs], dim=0)
    labels = torch.cat([positive_labels, negative_labels], dim=0)
    target_positions = torch.cat([positive_target_positions, negative_target_positions], dim=0)
    target_ids = torch.cat([positive_target_token_ids, negative_target_token_ids], dim=0)

    total_examples = positive_count + negative_count
    permutation = torch.randperm(total_examples, generator=generator)
    return (
        inputs[permutation],
        labels[permutation],
        target_positions[permutation],
        target_ids[permutation],
    )


def chunk_label_counts(total_examples: int, max_chunk_examples: int) -> list[tuple[int, int]]:
    """Return positive and negative counts for each evaluation chunk."""

    if total_examples < 1:
        raise ValueError("total_examples must be at least 1.")
    if max_chunk_examples < 1:
        raise ValueError("max_chunk_examples must be at least 1.")

    positive_remaining = total_examples // 2
    negative_remaining = total_examples - positive_remaining
    label_plan: list[int] = []
    for index in range(total_examples):
        if positive_remaining and (index % 2 == 0 or not negative_remaining):
            label_plan.append(1)
            positive_remaining -= 1
        else:
            label_plan.append(0)
            negative_remaining -= 1

    chunk_counts: list[tuple[int, int]] = []
    for start in range(0, total_examples, max_chunk_examples):
        labels = label_plan[start : start + max_chunk_examples]
        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        chunk_counts.append((positive_count, negative_count))
    return chunk_counts


def make_eval_dataset(
    *,
    length: int,
    positive_count: int,
    negative_count: int,
    seed: int,
    target_position_mode: str,
    target_token_count: int,
    non_target_token_count: int,
    non_target_sampling: str,
    eval_sampling_mode: str,
    positive_stratum_offset: int = 0,
    negative_final_query_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create one evaluation chunk using the selected sampling mode."""

    if eval_sampling_mode == "random":
        return make_stage3_dataset_from_counts(
            length=length,
            positive_count=positive_count,
            negative_count=negative_count,
            seed=seed,
            target_position_mode=target_position_mode,
            target_token_count=target_token_count,
            non_target_token_count=non_target_token_count,
            non_target_sampling=non_target_sampling,
        )
    if eval_sampling_mode == "stratified":
        return make_stratified_eval_dataset(
            length=length,
            positive_count=positive_count,
            negative_count=negative_count,
            seed=seed,
            target_position_mode=target_position_mode,
            target_token_count=target_token_count,
            non_target_token_count=non_target_token_count,
            non_target_sampling=non_target_sampling,
            positive_stratum_offset=positive_stratum_offset,
            negative_final_query_offset=negative_final_query_offset,
        )
    raise ValueError(f"Unsupported eval_sampling_mode: {eval_sampling_mode}")


def sample_target_tokens(
    *,
    shape: tuple[int, ...],
    target_token_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample target token ids from 0..H-1."""

    if target_token_count == 1:
        return torch.full(shape, TARGET_TOKEN_ID, dtype=torch.long)
    return torch.randint(
        low=TARGET_TOKEN_ID,
        high=target_token_count,
        size=shape,
        generator=generator,
        dtype=torch.long,
    )


def sample_non_target_tokens(
    *,
    shape: tuple[int, ...],
    target_token_count: int,
    non_target_token_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample non-target token ids from H..H+M-1."""

    start = first_non_target_token_id(target_token_count)
    if non_target_token_count == 1:
        return torch.full(shape, start, dtype=torch.long)
    return torch.randint(
        low=start,
        high=start + non_target_token_count,
        size=shape,
        generator=generator,
        dtype=torch.long,
    )


def make_loader(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    target_positions: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic DataLoader."""

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(inputs, labels, target_positions, target_ids),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def unpack_batch(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tokens and labels from a Stage 3 batch."""

    return batch[0], batch[1]


def resolved_train_lengths(config: Stage3Config) -> tuple[int, ...]:
    """Return the active training lengths, preserving backward compatibility."""

    return config.train_lengths if config.train_lengths else (config.train_length,)


def make_multilength_loaders(
    *,
    lengths: tuple[int, ...],
    examples_per_length: int,
    target_position_mode: str,
    target_token_count: int,
    non_target_token_count: int,
    non_target_sampling: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[DataLoader]:
    """Create one loader per training length.

    Different sequence lengths cannot be placed in one TensorDataset without padding.
    Padding would change the attended length, so Stage 3B keeps each batch single-length
    and shuffles the order of length-specific batches during training.
    """

    loaders: list[DataLoader] = []
    for offset, length in enumerate(lengths):
        inputs, labels, target_positions, target_ids = make_two_token_dataset(
            length=length,
            examples=examples_per_length,
            seed=seed + offset,
            target_position_mode=target_position_mode,
            target_token_count=target_token_count,
            non_target_token_count=non_target_token_count,
            non_target_sampling=non_target_sampling,
        )
        loaders.append(
            make_loader(
                inputs,
                labels,
                target_positions,
                target_ids,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed + 10_000 + offset,
            )
        )
    return loaders


class SimplifiedLastQueryAttentionClassifier(nn.Module):
    """Minimal trainable model matching the simplified attention setup."""

    def __init__(
        self,
        *,
        d_head: int,
        alpha_mode: str,
        alpha_log_scale_init: float,
        target_token_count: int = 1,
        non_target_token_count: int = 1,
    ) -> None:
        super().__init__()
        if d_head < 1:
            raise ValueError("d_head must be positive.")
        if alpha_mode not in {"constant", "log", "learned_log"}:
            raise ValueError(f"Unsupported alpha_mode: {alpha_mode}")
        if target_token_count < 1:
            raise ValueError("target_token_count must be at least 1.")
        if non_target_token_count < 1:
            raise ValueError("non_target_token_count must be at least 1.")

        self.d_head = d_head
        self.alpha_mode = alpha_mode
        self.target_token_count = target_token_count
        self.non_target_token_count = non_target_token_count
        self.score_vocab_size = target_token_count + non_target_token_count
        self.query_projection = nn.Linear(self.score_vocab_size, d_head, bias=False)
        self.key_projection = nn.Linear(self.score_vocab_size, d_head, bias=False)
        self.classifier = nn.Linear(2, 1)
        self.alpha_log_scale_unconstrained = nn.Parameter(
            torch.tensor(float(alpha_log_scale_init))
        )

    def project_tokens(self, tokens: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        """Apply a no-bias one-hot linear projection using an embedding lookup."""

        return F.embedding(tokens, projection.weight.transpose(0, 1))

    def token_value_output(
        self,
        tokens: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Map all non-target token values to [0, 1] and target values to [1, 0]."""

        target_mass = attention_weights.masked_fill(
            tokens.ge(self.target_token_count),
            0.0,
        ).sum(dim=1)
        return torch.stack([target_mass, 1.0 - target_mass], dim=1)

    def key_vectors_for_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return key vectors for token ids without constructing sequence one-hots."""

        return self.project_tokens(token_ids, self.key_projection)

    def alpha_for_length(self, length: int, device: torch.device) -> torch.Tensor:
        """Return the inverse-temperature scale for a sequence length."""

        if length < 2:
            raise ValueError("length must be at least 2.")

        if self.alpha_mode == "constant":
            return torch.ones((), device=device)
        if self.alpha_mode == "log":
            return torch.tensor(math.log(length), device=device, dtype=torch.float32)

        coefficient = F.softplus(self.alpha_log_scale_unconstrained)
        return 1.0 + coefficient * math.log1p(length)

    def learned_alpha_coefficient(self) -> float:
        """Return the learned log-length coefficient when applicable."""

        if self.alpha_mode != "learned_log":
            return float("nan")
        return F.softplus(self.alpha_log_scale_unconstrained.detach()).item()

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run last-query attention and optional mechanism tracing."""

        length = tokens.shape[1]
        keys = self.project_tokens(tokens, self.key_projection)
        last_query = self.project_tokens(tokens[:, -1], self.query_projection)
        raw_scores = torch.einsum("bd,bld->bl", last_query, keys) / math.sqrt(self.d_head)
        alpha = self.alpha_for_length(length, tokens.device)
        corrected_scores = alpha * raw_scores
        attention_weights = torch.softmax(corrected_scores, dim=-1)
        attention_output = self.token_value_output(tokens, attention_weights)
        logits = self.classifier(attention_output).squeeze(-1)

        if not return_details:
            return logits

        details = {
            "raw_scores": raw_scores,
            "corrected_scores": corrected_scores,
            "attention_weights": attention_weights,
            "attention_output": attention_output,
            "last_query": last_query,
            "alpha": alpha.expand(tokens.shape[0]),
        }
        return logits, details


def target_attention_theory(
    *,
    length: int,
    alpha: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Compute theoretical target attention mass from length, alpha, and margin."""

    log_non_target_count = math.log(length - 1)
    return torch.sigmoid(alpha * delta - log_non_target_count)


def generalized_target_attention_theory(
    *,
    alpha: torch.Tensor,
    margins: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Compute target attention with multiple non-target score margins."""

    denominator_terms = counts.to(dtype=torch.float32) * torch.exp(-alpha.unsqueeze(1) * margins)
    return 1.0 / (1.0 + denominator_terms.sum(dim=1))


def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Compute overall, positive, and negative accuracy."""

    predictions = (logits >= 0).float()
    correct = predictions.eq(labels)
    positive_mask = labels.eq(1)
    negative_mask = labels.eq(0)
    return {
        "accuracy": correct.float().mean().item(),
        "positive_accuracy": (
            correct[positive_mask].float().mean().item() if positive_mask.any() else float("nan")
        ),
        "negative_accuracy": (
            correct[negative_mask].float().mean().item() if negative_mask.any() else float("nan")
        ),
    }


def mean_or_nan(values: torch.Tensor) -> float:
    """Return a tensor mean or NaN for empty tensors."""

    return values.mean().item() if values.numel() else float("nan")


def run_epoch(
    model: SimplifiedLastQueryAttentionClassifier,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    """Run one training or evaluation epoch."""

    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        tokens, labels = unpack_batch(batch)
        tokens = tokens.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(tokens)
            loss = criterion(logits, labels)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits_cpu = torch.cat(all_logits)
    labels_cpu = torch.cat(all_labels)
    return total_loss / total_examples, binary_accuracy(logits_cpu, labels_cpu)


def run_loaders_once(
    model: SimplifiedLastQueryAttentionClassifier,
    loaders: list[DataLoader],
    *,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_steps: int | None = None,
    batch_shuffle_seed: int | None = None,
) -> tuple[float, dict[str, float], int]:
    """Run one pass over length-specific loaders.

    Returns the loss, accuracy metrics, and number of optimizer updates performed.
    """

    is_training = optimizer is not None
    model.train(is_training)

    batches = [batch for loader in loaders for batch in loader]
    if batch_shuffle_seed is not None:
        random.Random(batch_shuffle_seed).shuffle(batches)

    total_loss = 0.0
    total_examples = 0
    update_count = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in batches:
        if is_training and max_steps is not None and update_count >= max_steps:
            break

        tokens, labels = unpack_batch(batch)
        tokens = tokens.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(tokens)
            loss = criterion(logits, labels)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update_count += 1

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if total_examples == 0:
        return float("nan"), {"accuracy": float("nan"), "positive_accuracy": float("nan"), "negative_accuracy": float("nan")}, update_count

    logits_cpu = torch.cat(all_logits)
    labels_cpu = torch.cat(all_labels)
    return total_loss / total_examples, binary_accuracy(logits_cpu, labels_cpu), update_count


def target_position_bucket(position: int, length: int) -> str:
    """Map a non-final target position to a coarse position bucket."""

    nonfinal_count = length - 1
    if position < 0 or position >= nonfinal_count:
        raise ValueError(f"Target position must be in [0, {nonfinal_count - 1}].")

    if position < nonfinal_count / 3:
        return "beginning"
    if position < 2 * nonfinal_count / 3:
        return "middle"
    return "end_nonfinal"


def summarize_position_buckets(
    *,
    length: int,
    target_position_mode: str,
    target_positions: torch.Tensor,
    positive_correct: torch.Tensor,
    target_attentions: torch.Tensor,
    deltas: torch.Tensor,
    non_target_stds: torch.Tensor,
) -> list[dict[str, float | int | str]]:
    """Summarize positive examples by target-position bucket."""

    rows: list[dict[str, float | int | str]] = []
    buckets = [target_position_bucket(int(position), length) for position in target_positions]
    for bucket in ("beginning", "middle", "end_nonfinal"):
        mask = torch.tensor([value == bucket for value in buckets], dtype=torch.bool)
        if not mask.any():
            continue
        positions = target_positions[mask]
        rows.append(
            {
                "length": length,
                "target_position_mode": target_position_mode,
                "bucket": bucket,
                "positive_examples": int(mask.sum().item()),
                "positive_accuracy": positive_correct[mask].float().mean().item(),
                "mean_empirical_target_attention": target_attentions[mask].mean().item(),
                "mean_delta": deltas[mask].mean().item(),
                "mean_non_target_score_std": non_target_stds[mask].mean().item(),
                "min_target_position": int(positions.min().item()),
                "max_target_position": int(positions.max().item()),
                "final_target_count": int(positions.eq(length - 1).sum().item()),
            }
        )
    return rows


def iter_eval_batches(
    *,
    length: int,
    examples: int,
    eval_chunk_examples: int,
    batch_size: int,
    seed: int,
    target_position_mode: str,
    target_token_count: int,
    non_target_token_count: int,
    non_target_sampling: str,
    eval_sampling_mode: str,
):
    """Yield evaluation batches while generating only one chunk at a time."""

    positive_seen = 0
    negative_seen = 0
    for chunk_index, (positive_count, negative_count) in enumerate(
        chunk_label_counts(examples, eval_chunk_examples)
    ):
        inputs, labels, target_positions, target_ids = make_eval_dataset(
            length=length,
            positive_count=positive_count,
            negative_count=negative_count,
            seed=seed + chunk_index,
            target_position_mode=target_position_mode,
            target_token_count=target_token_count,
            non_target_token_count=non_target_token_count,
            non_target_sampling=non_target_sampling,
            eval_sampling_mode=eval_sampling_mode,
            positive_stratum_offset=positive_seen,
            negative_final_query_offset=negative_seen,
        )
        loader = make_loader(
            inputs,
            labels,
            target_positions,
            target_ids,
            batch_size=batch_size,
            shuffle=False,
            seed=seed + 50_000 + chunk_index,
        )
        yield from loader
        positive_seen += positive_count
        negative_seen += negative_count


@torch.no_grad()
def evaluate_length(
    model: SimplifiedLastQueryAttentionClassifier,
    *,
    length: int,
    examples: int,
    batch_size: int,
    seed: int,
    target_position_mode: str,
    target_token_count: int,
    non_target_token_count: int,
    non_target_sampling: str,
    device: torch.device,
    eval_chunk_examples: int = Stage3Config.eval_chunk_examples,
    eval_sampling_mode: str = Stage3Config.eval_sampling_mode,
) -> tuple[
    dict[str, float | int | str],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    """Evaluate one length and collect mechanism metrics."""

    if examples < 1:
        raise ValueError("examples must be at least 1.")
    if eval_chunk_examples < 1:
        raise ValueError("eval_chunk_examples must be at least 1.")
    if eval_sampling_mode not in {"random", "stratified"}:
        raise ValueError(f"Unsupported eval_sampling_mode: {eval_sampling_mode}")

    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    positive_logits: list[torch.Tensor] = []
    positive_probabilities: list[torch.Tensor] = []
    negative_logits: list[torch.Tensor] = []
    negative_probabilities: list[torch.Tensor] = []
    target_scores: list[torch.Tensor] = []
    non_target_means: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    non_target_stds: list[torch.Tensor] = []
    min_margins: list[torch.Tensor] = []
    max_non_target_scores: list[torch.Tensor] = []
    non_target_type_score_stds: list[torch.Tensor] = []
    empirical_attentions: list[torch.Tensor] = []
    empirical_theory_attentions: list[torch.Tensor] = []
    generalized_theory_attentions: list[torch.Tensor] = []
    alpha_values: list[torch.Tensor] = []
    positive_target_positions: list[torch.Tensor] = []
    positive_target_ids: list[torch.Tensor] = []
    positive_final_query_ids: list[torch.Tensor] = []
    positive_correct_values: list[torch.Tensor] = []
    per_type_values: dict[int, dict[str, list[torch.Tensor]]] = {
        token_id: {
            "counts": [],
            "scores": [],
            "margins": [],
            "denominator_contributions": [],
            "denominator_fractions": [],
        }
        for token_id in non_target_token_ids(target_token_count, non_target_token_count)
    }
    per_target_type_values: dict[tuple[int, int], dict[str, list[torch.Tensor]]] = {
        (final_query_id, target_id): {
            "correct": [],
            "target_scores": [],
            "target_attentions": [],
            "min_margins": [],
        }
        for final_query_id in non_target_token_ids(target_token_count, non_target_token_count)
        for target_id in target_token_ids(target_token_count)
    }

    for tokens, batch_labels, batch_target_positions, batch_target_ids in iter_eval_batches(
        length=length,
        examples=examples,
        eval_chunk_examples=eval_chunk_examples,
        batch_size=batch_size,
        seed=seed,
        target_position_mode=target_position_mode,
        target_token_count=target_token_count,
        non_target_token_count=non_target_token_count,
        non_target_sampling=non_target_sampling,
        eval_sampling_mode=eval_sampling_mode,
    ):
        tokens = tokens.to(device)
        batch_labels = batch_labels.to(device)
        batch_target_positions = batch_target_positions.to(device)
        batch_target_ids = batch_target_ids.to(device)
        logits, details = model(tokens, return_details=True)
        probabilities = torch.sigmoid(logits)

        all_logits.append(logits.cpu())
        all_labels.append(batch_labels.cpu())

        positive_mask = batch_labels.eq(1)
        negative_mask = batch_labels.eq(0)

        positive_logits.append(logits[positive_mask].cpu())
        positive_probabilities.append(probabilities[positive_mask].cpu())
        negative_logits.append(logits[negative_mask].cpu())
        negative_probabilities.append(probabilities[negative_mask].cpu())

        if positive_mask.any():
            raw_scores = details["raw_scores"][positive_mask]
            attention_weights = details["attention_weights"][positive_mask]
            positive_tokens = tokens[positive_mask]
            alpha = details["alpha"][positive_mask]
            last_query = details["last_query"][positive_mask]
            target_position = batch_target_positions[positive_mask]
            target_id = batch_target_ids[positive_mask]
            final_query_id = positive_tokens[:, -1]
            if target_position.eq(length - 1).any():
                raise RuntimeError("Positive examples must not place the target at final position.")
            if target_id.lt(0).any() or target_id.ge(target_token_count).any():
                raise RuntimeError("Positive examples must have valid target token ids.")
            if final_query_id.lt(target_token_count).any():
                raise RuntimeError("Final query token must be non-target in Stage 3 evaluation.")
            target_score = raw_scores.gather(1, target_position.unsqueeze(1)).squeeze(1)
            non_target_mask = torch.ones_like(raw_scores, dtype=torch.bool)
            non_target_mask.scatter_(1, target_position.unsqueeze(1), False)
            non_target_scores = raw_scores[non_target_mask].view(
                raw_scores.shape[0],
                length - 1,
            )
            non_target_mean = non_target_scores.mean(dim=1)
            delta = target_score - non_target_mean
            non_target_token_id_tensor = torch.tensor(
                non_target_token_ids(target_token_count, non_target_token_count),
                device=device,
                dtype=torch.long,
            )
            non_target_type_keys = model.key_vectors_for_token_ids(non_target_token_id_tensor)
            non_target_type_scores = (
                torch.einsum("bd,kd->bk", last_query, non_target_type_keys)
                / math.sqrt(model.d_head)
            )
            non_target_type_counts = torch.stack(
                [
                    positive_tokens.eq(token_id).sum(dim=1)
                    for token_id in non_target_token_id_tensor.tolist()
                ],
                dim=1,
            )
            type_margins = target_score.unsqueeze(1) - non_target_type_scores
            denominator_contributions = non_target_type_counts.to(dtype=torch.float32) * torch.exp(
                -alpha.unsqueeze(1) * type_margins
            )
            denominator_sum = denominator_contributions.sum(dim=1)
            denominator_fractions = denominator_contributions / denominator_sum.clamp_min(
                1e-30
            ).unsqueeze(1)
            empirical_attention = attention_weights.gather(
                1,
                target_position.unsqueeze(1),
            ).squeeze(1)
            theory_attention = target_attention_theory(
                length=length,
                alpha=alpha,
                delta=delta,
            )
            generalized_theory_attention = generalized_target_attention_theory(
                alpha=alpha,
                margins=type_margins,
                counts=non_target_type_counts,
            )

            target_scores.append(target_score.cpu())
            non_target_means.append(non_target_mean.cpu())
            deltas.append(delta.cpu())
            non_target_stds.append(non_target_scores.std(dim=1, unbiased=False).cpu())
            min_margins.append(type_margins.min(dim=1).values.cpu())
            max_non_target_scores.append(non_target_type_scores.max(dim=1).values.cpu())
            non_target_type_score_stds.append(
                non_target_type_scores.std(dim=1, unbiased=False).cpu()
            )
            empirical_attentions.append(empirical_attention.cpu())
            empirical_theory_attentions.append(theory_attention.cpu())
            generalized_theory_attentions.append(generalized_theory_attention.cpu())
            alpha_values.append(alpha.cpu())
            positive_target_positions.append(target_position.cpu())
            positive_target_ids.append(target_id.cpu())
            positive_final_query_ids.append(final_query_id.cpu())
            positive_correct_values.append(logits[positive_mask].ge(0).cpu())
            positive_correct = logits[positive_mask].ge(0)
            for column_index, token_id in enumerate(non_target_token_id_tensor.tolist()):
                per_type_values[token_id]["counts"].append(
                    non_target_type_counts[:, column_index].cpu()
                )
                per_type_values[token_id]["scores"].append(
                    non_target_type_scores[:, column_index].cpu()
                )
                per_type_values[token_id]["margins"].append(type_margins[:, column_index].cpu())
                per_type_values[token_id]["denominator_contributions"].append(
                    denominator_contributions[:, column_index].cpu()
                )
                per_type_values[token_id]["denominator_fractions"].append(
                    denominator_fractions[:, column_index].cpu()
                )
            for final_query_value in non_target_token_id_tensor.tolist():
                final_query_mask = final_query_id.eq(final_query_value)
                if not final_query_mask.any():
                    continue
                for target_value in range(target_token_count):
                    group_mask = final_query_mask & target_id.eq(target_value)
                    if not group_mask.any():
                        continue
                    key = (final_query_value, target_value)
                    per_target_type_values[key]["correct"].append(
                        positive_correct[group_mask].cpu()
                    )
                    per_target_type_values[key]["target_scores"].append(
                        target_score[group_mask].cpu()
                    )
                    per_target_type_values[key]["target_attentions"].append(
                        empirical_attention[group_mask].cpu()
                    )
                    per_target_type_values[key]["min_margins"].append(
                        type_margins.min(dim=1).values[group_mask].cpu()
                    )

    logits_cpu = torch.cat(all_logits)
    labels_cpu = torch.cat(all_labels)
    metrics = binary_accuracy(logits_cpu, labels_cpu)

    positive_logits_cpu = torch.cat(positive_logits)
    positive_probabilities_cpu = torch.cat(positive_probabilities)
    negative_logits_cpu = torch.cat(negative_logits)
    negative_probabilities_cpu = torch.cat(negative_probabilities)
    target_scores_cpu = torch.cat(target_scores)
    non_target_means_cpu = torch.cat(non_target_means)
    deltas_cpu = torch.cat(deltas)
    non_target_stds_cpu = torch.cat(non_target_stds)
    min_margins_cpu = torch.cat(min_margins)
    max_non_target_scores_cpu = torch.cat(max_non_target_scores)
    non_target_type_score_stds_cpu = torch.cat(non_target_type_score_stds)
    empirical_attentions_cpu = torch.cat(empirical_attentions)
    empirical_theory_attentions_cpu = torch.cat(empirical_theory_attentions)
    generalized_theory_attentions_cpu = torch.cat(generalized_theory_attentions)
    alpha_values_cpu = torch.cat(alpha_values)
    positive_target_positions_cpu = torch.cat(positive_target_positions)
    positive_target_ids_cpu = torch.cat(positive_target_ids)
    positive_final_query_ids_cpu = torch.cat(positive_final_query_ids)
    positive_correct_cpu = torch.cat(positive_correct_values)

    attention_abs_error = (empirical_attentions_cpu - empirical_theory_attentions_cpu).abs()
    generalized_attention_abs_error = (
        empirical_attentions_cpu - generalized_theory_attentions_cpu
    ).abs()
    classifier_weight = model.classifier.weight.detach().cpu().squeeze(0)
    learned_alpha_coefficient = model.learned_alpha_coefficient()

    row = {
        "length": length,
        "split": "test",
        "alpha_mode": model.alpha_mode,
        "target_position_mode": target_position_mode,
        "target_token_count": target_token_count,
        "non_target_token_count": non_target_token_count,
        "non_target_sampling": non_target_sampling,
        "test_examples": examples,
        "eval_chunk_examples": eval_chunk_examples,
        "eval_sampling_mode": eval_sampling_mode,
        "positive_examples": int(labels_cpu.eq(1).sum().item()),
        "negative_examples": int(labels_cpu.eq(0).sum().item()),
        "alpha_value": alpha_values_cpu.mean().item(),
        "accuracy": metrics["accuracy"],
        "positive_accuracy": metrics["positive_accuracy"],
        "negative_accuracy": metrics["negative_accuracy"],
        "mean_logit_positive": mean_or_nan(positive_logits_cpu),
        "mean_logit_negative": mean_or_nan(negative_logits_cpu),
        "mean_probability_positive": mean_or_nan(positive_probabilities_cpu),
        "mean_probability_negative": mean_or_nan(negative_probabilities_cpu),
        "mean_target_score_a": mean_or_nan(target_scores_cpu),
        "mean_non_target_score_b": mean_or_nan(non_target_means_cpu),
        "mean_delta": mean_or_nan(deltas_cpu),
        "std_non_target_scores": mean_or_nan(non_target_stds_cpu),
        "mean_min_margin_delta": mean_or_nan(min_margins_cpu),
        "min_margin_delta": min_margins_cpu.min().item(),
        "mean_max_non_target_score": mean_or_nan(max_non_target_scores_cpu),
        "mean_non_target_type_score_std": mean_or_nan(non_target_type_score_stds_cpu),
        "mean_empirical_target_attention": mean_or_nan(empirical_attentions_cpu),
        "mean_theory_target_attention_using_empirical_delta": mean_or_nan(
            empirical_theory_attentions_cpu
        ),
        "mean_generalized_theory_target_attention": mean_or_nan(
            generalized_theory_attentions_cpu
        ),
        "mean_theory_target_attention_using_train_delta": float("nan"),
        "mean_attention_absolute_error_empirical_vs_theory": mean_or_nan(attention_abs_error),
        "mean_attention_absolute_error_empirical_vs_generalized_theory": mean_or_nan(
            generalized_attention_abs_error
        ),
        "learned_alpha_coefficient": learned_alpha_coefficient,
        "learned_log_c_delta_min_mean": (
            learned_alpha_coefficient * mean_or_nan(min_margins_cpu)
            if model.alpha_mode == "learned_log"
            else float("nan")
        ),
        "learned_log_c_delta_min_worst": (
            learned_alpha_coefficient * min_margins_cpu.min().item()
            if model.alpha_mode == "learned_log"
            else float("nan")
        ),
        "classifier_weight_target_coord": classifier_weight[0].item(),
        "classifier_weight_non_target_coord": classifier_weight[1].item(),
        "classifier_bias": model.classifier.bias.detach().cpu().item(),
    }
    position_rows = summarize_position_buckets(
        length=length,
        target_position_mode=target_position_mode,
        target_positions=positive_target_positions_cpu,
        positive_correct=positive_correct_cpu,
        target_attentions=empirical_attentions_cpu,
        deltas=deltas_cpu,
        non_target_stds=non_target_stds_cpu,
    )
    for position_row in position_rows:
        position_row.update(
            {
                "split": "test",
                "alpha_mode": model.alpha_mode,
                "target_token_count": target_token_count,
                "non_target_token_count": non_target_token_count,
                "non_target_sampling": non_target_sampling,
                "test_examples": examples,
                "eval_chunk_examples": eval_chunk_examples,
                "eval_sampling_mode": eval_sampling_mode,
            }
        )
    non_target_type_rows = []
    for token_id, values in per_type_values.items():
        counts = torch.cat(values["counts"])
        scores = torch.cat(values["scores"])
        margins = torch.cat(values["margins"])
        denominator_contributions = torch.cat(values["denominator_contributions"])
        denominator_fractions = torch.cat(values["denominator_fractions"])
        non_target_type_rows.append(
            {
                "length": length,
                "split": "test",
                "alpha_mode": model.alpha_mode,
                "target_position_mode": target_position_mode,
                "target_token_count": target_token_count,
                "non_target_token_count": non_target_token_count,
                "non_target_sampling": non_target_sampling,
                "test_examples": examples,
                "eval_chunk_examples": eval_chunk_examples,
                "eval_sampling_mode": eval_sampling_mode,
                "non_target_token_id": token_id,
                "mean_non_target_count_in_sequence": counts.to(dtype=torch.float32).mean().item(),
                "mean_non_target_score": mean_or_nan(scores),
                "mean_margin_from_target": mean_or_nan(margins),
                "mean_denominator_contribution": mean_or_nan(denominator_contributions),
                "mean_denominator_fraction": mean_or_nan(denominator_fractions),
            }
        )
    target_type_rows = []
    for (final_query_id_value, target_id_value), values in per_target_type_values.items():
        if not values["correct"]:
            continue
        correct = torch.cat(values["correct"])
        target_scores_for_type = torch.cat(values["target_scores"])
        target_attentions_for_type = torch.cat(values["target_attentions"])
        min_margins_for_type = torch.cat(values["min_margins"])
        target_type_rows.append(
            {
                "length": length,
                "split": "test",
                "alpha_mode": model.alpha_mode,
                "target_token_count": target_token_count,
                "non_target_token_count": non_target_token_count,
                "target_position_mode": target_position_mode,
                "test_examples": examples,
                "eval_chunk_examples": eval_chunk_examples,
                "eval_sampling_mode": eval_sampling_mode,
                "final_query_token_id": final_query_id_value,
                "target_token_id": target_id_value,
                "positive_examples": int(correct.numel()),
                "positive_accuracy": correct.to(dtype=torch.float32).mean().item(),
                "mean_target_score": mean_or_nan(target_scores_for_type),
                "mean_target_attention": mean_or_nan(target_attentions_for_type),
                "mean_min_margin": mean_or_nan(min_margins_for_type),
                "worst_observed_min_margin": min_margins_for_type.min().item(),
                "mean_c_delta_min": (
                    learned_alpha_coefficient * mean_or_nan(min_margins_for_type)
                    if model.alpha_mode == "learned_log"
                    else float("nan")
                ),
                "worst_observed_c_delta_min": (
                    learned_alpha_coefficient * min_margins_for_type.min().item()
                    if model.alpha_mode == "learned_log"
                    else float("nan")
                ),
            }
        )
    return row, position_rows, non_target_type_rows, target_type_rows


def add_train_delta_theory(
    rows: list[dict[str, float | int | str]],
    *,
    train_length: int,
) -> None:
    """Add theory predictions that reuse the measured training-length margin."""

    train_rows = [row for row in rows if int(row["length"]) == train_length]
    train_delta = float(train_rows[0]["mean_delta"] if train_rows else rows[0]["mean_delta"])

    for row in rows:
        length = int(row["length"])
        alpha = torch.tensor(float(row["alpha_value"]))
        delta = torch.tensor(train_delta)
        row["mean_theory_target_attention_using_train_delta"] = target_attention_theory(
            length=length,
            alpha=alpha,
            delta=delta,
        ).item()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2)


def save_model_checkpoint(
    path: Path,
    *,
    model: SimplifiedLastQueryAttentionClassifier,
    config: Stage3Config,
    optimizer_updates: int,
) -> None:
    """Save a trained Stage 3 model checkpoint for mechanism analysis."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "d_head": config.d_head,
            "alpha_mode": config.alpha_mode,
            "alpha_log_scale_init": config.alpha_log_scale_init,
            "train_length": config.train_length,
            "train_lengths": list(resolved_train_lengths(config)),
            "optimizer_updates": optimizer_updates,
            "target_token_id": TARGET_TOKEN_ID,
            "target_token_count": config.target_token_count,
            "target_token_ids": target_token_ids(config.target_token_count),
            "non_target_token_id": first_non_target_token_id(config.target_token_count),
            "non_target_token_count": config.non_target_token_count,
            "non_target_token_ids": non_target_token_ids(
                config.target_token_count,
                config.non_target_token_count,
            ),
            "non_target_sampling": config.non_target_sampling,
            "target_position_mode": config.target_position_mode,
            "final_target_allowed": False,
        },
        path,
    )


def write_figures(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write diagnostic figures when matplotlib is available."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped Stage 3 figures.")
        return

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    lengths = [int(row["length"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(
        lengths,
        [float(row["mean_empirical_target_attention"]) for row in rows],
        marker="o",
        label="empirical target attention",
    )
    ax.plot(
        lengths,
        [float(row["mean_theory_target_attention_using_empirical_delta"]) for row in rows],
        marker="s",
        label="theory using empirical delta",
    )
    if "mean_generalized_theory_target_attention" in rows[0]:
        ax.plot(
            lengths,
            [float(row["mean_generalized_theory_target_attention"]) for row in rows],
            marker="D",
            label="generalized theory",
        )
    ax.plot(
        lengths,
        [float(row["mean_theory_target_attention_using_train_delta"]) for row in rows],
        marker="^",
        label="theory using train delta",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Target attention mass")
    ax.set_title("Stage 3 theory vs empirical attention")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(figure_dir / "stage3_theory_vs_empirical_attention.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(lengths, [float(row["mean_delta"]) for row in rows], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Measured margin delta")
    ax.set_title("Stage 3 measured margin by length")
    ax.grid(True, alpha=0.25)
    fig.savefig(figure_dir / "stage3_delta_by_length.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(lengths, [float(row["std_non_target_scores"]) for row in rows], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Mean non-target score std")
    ax.set_title("Stage 3 non-target score variation")
    ax.grid(True, alpha=0.25)
    fig.savefig(figure_dir / "stage3_non_target_score_std.png", bbox_inches="tight")
    plt.close(fig)

    fig, (ax_acc, ax_logit) = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    ax_acc.plot(lengths, [float(row["accuracy"]) for row in rows], marker="o", label="overall")
    ax_acc.plot(
        lengths,
        [float(row["positive_accuracy"]) for row in rows],
        marker="s",
        label="positive",
    )
    ax_acc.plot(
        lengths,
        [float(row["negative_accuracy"]) for row in rows],
        marker="^",
        label="negative",
    )
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(-0.03, 1.03)
    ax_acc.grid(True, alpha=0.25)
    ax_acc.legend()

    ax_logit.plot(
        lengths,
        [float(row["mean_logit_positive"]) for row in rows],
        marker="o",
        label="positive logit",
    )
    ax_logit.plot(
        lengths,
        [float(row["mean_logit_negative"]) for row in rows],
        marker="s",
        label="negative logit",
    )
    ax_logit.axhline(0.0, color="black", linewidth=1)
    ax_logit.set_xscale("log")
    ax_logit.set_xlabel("Sequence length")
    ax_logit.set_ylabel("Mean logit")
    ax_logit.grid(True, alpha=0.25)
    ax_logit.legend()
    fig.savefig(figure_dir / "stage3_accuracy_and_logits.png", bbox_inches="tight")
    plt.close(fig)


def train_model(
    config: Stage3Config,
    *,
    device: torch.device,
    output_dir: Path,
) -> tuple[SimplifiedLastQueryAttentionClassifier, int]:
    """Train the simplified attention model."""

    train_lengths = resolved_train_lengths(config)
    train_loaders = make_multilength_loaders(
        lengths=train_lengths,
        examples_per_length=config.train_examples,
        target_position_mode=config.target_position_mode,
        target_token_count=config.target_token_count,
        non_target_token_count=config.non_target_token_count,
        non_target_sampling=config.non_target_sampling,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed + 1,
    )
    val_loaders = make_multilength_loaders(
        lengths=train_lengths,
        examples_per_length=config.val_examples,
        target_position_mode=config.target_position_mode,
        target_token_count=config.target_token_count,
        non_target_token_count=config.non_target_token_count,
        non_target_sampling=config.non_target_sampling,
        batch_size=config.eval_batch_size,
        shuffle=False,
        seed=config.seed + 2,
    )

    model = SimplifiedLastQueryAttentionClassifier(
        d_head=config.d_head,
        alpha_mode=config.alpha_mode,
        alpha_log_scale_init=config.alpha_log_scale_init,
        target_token_count=config.target_token_count,
        non_target_token_count=config.non_target_token_count,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history_rows: list[dict[str, float | int | str]] = []
    optimizer_updates = 0
    epoch = 0
    while True:
        if config.max_train_steps is None and epoch >= config.epochs:
            break
        if config.max_train_steps is not None and optimizer_updates >= config.max_train_steps:
            break

        epoch += 1
        remaining_steps = (
            None
            if config.max_train_steps is None
            else config.max_train_steps - optimizer_updates
        )
        train_loss, train_metrics, updates_this_epoch = run_loaders_once(
            model,
            train_loaders,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_steps=remaining_steps,
            batch_shuffle_seed=config.seed + 20_000 + epoch,
        )
        optimizer_updates += updates_this_epoch
        val_loss, val_metrics, _ = run_loaders_once(
            model,
            val_loaders,
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "optimizer_updates": optimizer_updates,
                "updates_this_epoch": updates_this_epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_positive_accuracy": train_metrics["positive_accuracy"],
                "train_negative_accuracy": train_metrics["negative_accuracy"],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_positive_accuracy": val_metrics["positive_accuracy"],
                "val_negative_accuracy": val_metrics["negative_accuracy"],
            }
        )

    write_csv(output_dir / "train_history.csv", history_rows)
    return model, optimizer_updates


def main() -> None:
    """Run the Stage 3 simplified attention experiment."""

    args = parse_args()
    config = build_config(args)
    set_reproducibility(config.seed)
    device = resolve_device(config.device)
    output_dir = project_dir() / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_lengths = resolved_train_lengths(config)

    model, optimizer_updates = train_model(config, device=device, output_dir=output_dir)
    save_model_checkpoint(
        output_dir / "model.pt",
        model=model,
        config=config,
        optimizer_updates=optimizer_updates,
    )
    write_json(
        output_dir / "config.json",
        {
            **asdict(config),
            "eval_lengths": list(config.eval_lengths),
            "train_lengths": list(train_lengths),
            "train_length_count": len(train_lengths),
            "examples_per_train_length": config.train_examples,
            "optimizer_updates": optimizer_updates,
            "resolved_device": str(device),
            "target_token_id": TARGET_TOKEN_ID,
            "target_token_count": config.target_token_count,
            "target_token_ids": target_token_ids(config.target_token_count),
            "non_target_token_id": first_non_target_token_id(config.target_token_count),
            "non_target_token_count": config.non_target_token_count,
            "non_target_token_ids": non_target_token_ids(
                config.target_token_count,
                config.non_target_token_count,
            ),
            "non_target_sampling": config.non_target_sampling,
            "target_position_mode": config.target_position_mode,
            "final_target_allowed": False,
        },
    )

    evaluation_results = [
        evaluate_length(
            model,
            length=length,
            examples=config.test_examples,
            eval_chunk_examples=config.eval_chunk_examples,
            eval_sampling_mode=config.eval_sampling_mode,
            batch_size=config.eval_batch_size,
            seed=config.seed + 10_000 + length,
            target_position_mode=config.target_position_mode,
            target_token_count=config.target_token_count,
            non_target_token_count=config.non_target_token_count,
            non_target_sampling=config.non_target_sampling,
            device=device,
        )
        for length in config.eval_lengths
    ]
    rows = [row for row, _, _, _ in evaluation_results]
    target_position_rows = [
        position_row
        for _, position_rows, _, _ in evaluation_results
        for position_row in position_rows
    ]
    non_target_type_rows = [
        type_row
        for _, _, type_rows, _ in evaluation_results
        for type_row in type_rows
    ]
    target_type_rows = [
        type_row
        for _, _, _, type_rows in evaluation_results
        for type_row in type_rows
    ]
    add_train_delta_theory(rows, train_length=train_lengths[0])
    for row in rows:
        row["train_lengths"] = " ".join(str(length) for length in train_lengths)
        row["train_length_count"] = len(train_lengths)
        row["optimizer_updates"] = optimizer_updates
        row["examples_per_train_length"] = config.train_examples
        row["final_target_allowed"] = False
    for row in target_position_rows:
        row["train_lengths"] = " ".join(str(length) for length in train_lengths)
        row["train_length_count"] = len(train_lengths)
        row["optimizer_updates"] = optimizer_updates
        row["examples_per_train_length"] = config.train_examples
        row["final_target_allowed"] = False
    for row in non_target_type_rows:
        row["train_lengths"] = " ".join(str(length) for length in train_lengths)
        row["train_length_count"] = len(train_lengths)
        row["optimizer_updates"] = optimizer_updates
        row["examples_per_train_length"] = config.train_examples
        row["final_target_allowed"] = False
    for row in target_type_rows:
        row["train_lengths"] = " ".join(str(length) for length in train_lengths)
        row["train_length_count"] = len(train_lengths)
        row["optimizer_updates"] = optimizer_updates
        row["examples_per_train_length"] = config.train_examples
        row["final_target_allowed"] = False
    write_csv(output_dir / "metrics_by_length.csv", rows)
    if target_position_rows:
        write_csv(output_dir / "target_position_metrics.csv", target_position_rows)
    if non_target_type_rows:
        write_csv(output_dir / "non_target_type_metrics.csv", non_target_type_rows)
    if target_type_rows:
        write_csv(output_dir / "target_type_metrics.csv", target_type_rows)
    write_figures(output_dir, rows)

    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
