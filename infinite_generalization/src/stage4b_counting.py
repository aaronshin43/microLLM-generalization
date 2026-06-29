"""Train Stage 4B: target occurrence counting on the reduced attention model.

Stage 4B keeps the Stage 3/4A reduced last-query attention architecture but changes the
label to the total number of target-token occurrences. The value pathway remains the
normalized per-target-type attention mass plus total non-target mass, while the head predicts
one of ``K + 1`` count classes.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stage3_simplified_attention import (
    SimplifiedLastQueryAttentionClassifier,
    non_target_token_ids,
    project_dir,
    resolve_device,
    set_reproducibility,
    target_token_ids,
    write_csv,
    write_json,
)


@dataclass(frozen=True)
class Stage4BConfig:
    """Configuration for the Stage 4B counting experiment."""

    seed: int = 42
    device: str = "auto"
    output_dir: str = "runs/stage4b_counting"
    alpha_mode: str = "constant"
    readout_mode: str = "softmax_mass"
    train_length: int = 10
    train_lengths: tuple[int, ...] = ()
    target_position_mode: str = "fixed_start"
    target_token_count: int = 1
    non_target_token_count: int = 1
    non_target_sampling: str = "uniform"
    max_target_count: int = 3
    train_examples: int = 2_000
    val_examples: int = 500
    test_examples: int = 64
    eval_chunk_examples: int = 64
    eval_sampling_mode: str = "stratified"
    eval_lengths: tuple[int, ...] = (10, 100, 1000, 10000, 100000, 1000000, 5000000, 10000000)
    batch_size: int = 64
    eval_batch_size: int = 16
    epochs: int = 200
    max_train_steps: int | None = None
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    d_head: int = 2
    alpha_log_scale_init: float = -5.0


def resolved_train_lengths(config: Stage4BConfig) -> tuple[int, ...]:
    """Return the active training lengths."""

    return config.train_lengths if config.train_lengths else (config.train_length,)


def resolve_output_path(path_value: str) -> Path:
    """Resolve an output path using the project directory for relative paths."""

    path = Path(path_value)
    return path if path.is_absolute() else project_dir() / path


def count_class_labels(
    *,
    examples: int,
    max_target_count: int,
    start_index: int = 0,
) -> torch.Tensor:
    """Return balanced count labels in deterministic round-robin order."""

    if examples < 1:
        raise ValueError("examples must be at least 1.")
    if max_target_count < 1:
        raise ValueError("max_target_count must be at least 1.")
    class_count = max_target_count + 1
    labels = [(start_index + index) % class_count for index in range(examples)]
    return torch.tensor(labels, dtype=torch.long)


def random_count_class_labels(
    *,
    examples: int,
    max_target_count: int,
    seed: int,
    start_index: int = 0,
) -> torch.Tensor:
    """Return deterministic per-example random count labels."""

    if examples < 1:
        raise ValueError("examples must be at least 1.")
    if max_target_count < 1:
        raise ValueError("max_target_count must be at least 1.")

    labels: list[int] = []
    for index in range(examples):
        generator = torch.Generator().manual_seed(seed + start_index + index)
        label = torch.randint(
            low=0,
            high=max_target_count + 1,
            size=(),
            generator=generator,
            dtype=torch.long,
        ).item()
        labels.append(int(label))
    return torch.tensor(labels, dtype=torch.long)


def _sample_non_target_tokens(
    *,
    shape: tuple[int, ...],
    target_token_count: int,
    non_target_token_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample non-target token ids under the contiguous id convention."""

    start = target_token_count
    if non_target_token_count == 1:
        return torch.full(shape, start, dtype=torch.long)
    return torch.randint(
        low=start,
        high=start + non_target_token_count,
        size=shape,
        generator=generator,
        dtype=torch.long,
    )


def _sample_target_token_ids(
    *,
    count: int,
    target_token_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample target token ids for target occurrences."""

    if target_token_count == 1:
        return torch.zeros(count, dtype=torch.long)
    return torch.randint(
        low=0,
        high=target_token_count,
        size=(count,),
        generator=generator,
        dtype=torch.long,
    )


def _target_positions_for_count(
    *,
    length: int,
    count: int,
    target_position_mode: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return distinct non-final target positions for one example."""

    if count == 0:
        return torch.empty(0, dtype=torch.long)
    if count > length - 1:
        raise ValueError(
            f"count={count} cannot fit into the {length - 1} non-final positions."
        )
    if target_position_mode == "fixed_start":
        return torch.arange(count, dtype=torch.long)
    if target_position_mode == "nonfinal_random":
        return torch.randperm(length - 1, generator=generator, dtype=torch.long)[:count]
    raise ValueError(f"Unsupported target_position_mode: {target_position_mode}")


def make_count_dataset_from_labels(
    *,
    length: int,
    count_labels: torch.Tensor,
    seed: int,
    target_position_mode: str,
    target_token_count: int,
    non_target_token_count: int,
    non_target_sampling: str,
    max_target_count: int,
    global_start_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a Stage 4B counting dataset from explicit count labels.

    ``target_positions`` and ``target_ids`` are padded with -1 to shape
    ``(examples, max_target_count)``.
    """

    if length < 2:
        raise ValueError("length must be at least 2.")
    if max_target_count < 1:
        raise ValueError("max_target_count must be at least 1.")
    if max_target_count > length - 1:
        raise ValueError("max_target_count must fit into non-final positions.")
    if target_token_count < 1:
        raise ValueError("target_token_count must be at least 1.")
    if non_target_token_count < 1:
        raise ValueError("non_target_token_count must be at least 1.")
    if non_target_sampling != "uniform":
        raise ValueError(f"Unsupported non_target_sampling: {non_target_sampling}")
    if target_position_mode not in {"fixed_start", "nonfinal_random"}:
        raise ValueError(f"Unsupported target_position_mode: {target_position_mode}")
    if count_labels.numel() < 1:
        raise ValueError("count_labels must be non-empty.")
    if count_labels.lt(0).any() or count_labels.gt(max_target_count).any():
        raise ValueError("count_labels must be in 0..max_target_count.")

    examples = int(count_labels.numel())
    inputs = torch.empty((examples, length), dtype=torch.long)
    target_positions = torch.full((examples, max_target_count), -1, dtype=torch.long)
    target_ids = torch.full((examples, max_target_count), -1, dtype=torch.long)

    for row_index, label in enumerate(count_labels.tolist()):
        example_seed = seed + global_start_index + row_index
        generator = torch.Generator().manual_seed(example_seed)
        row = _sample_non_target_tokens(
            shape=(length,),
            target_token_count=target_token_count,
            non_target_token_count=non_target_token_count,
            generator=generator,
        )
        positions = _target_positions_for_count(
            length=length,
            count=int(label),
            target_position_mode=target_position_mode,
            generator=generator,
        )
        sampled_target_ids = _sample_target_token_ids(
            count=int(label),
            target_token_count=target_token_count,
            generator=generator,
        )
        if int(label):
            row[positions] = sampled_target_ids
            target_positions[row_index, : int(label)] = positions
            target_ids[row_index, : int(label)] = sampled_target_ids
        inputs[row_index] = row

    return inputs, count_labels.clone(), target_positions, target_ids


def make_count_dataset(
    *,
    length: int,
    examples: int,
    seed: int,
    target_position_mode: str,
    target_token_count: int,
    non_target_token_count: int,
    non_target_sampling: str,
    max_target_count: int,
    label_start_index: int = 0,
    global_start_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a balanced target-counting dataset."""

    labels = count_class_labels(
        examples=examples,
        max_target_count=max_target_count,
        start_index=label_start_index,
    )
    return make_count_dataset_from_labels(
        length=length,
        count_labels=labels,
        seed=seed,
        target_position_mode=target_position_mode,
        target_token_count=target_token_count,
        non_target_token_count=non_target_token_count,
        non_target_sampling=non_target_sampling,
        max_target_count=max_target_count,
        global_start_index=global_start_index,
    )


def make_count_loader(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    target_positions: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic DataLoader for Stage 4B tensors."""

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(inputs, labels, target_positions, target_ids),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def make_count_multilength_loaders(
    *,
    lengths: tuple[int, ...],
    examples_per_length: int,
    target_position_mode: str,
    target_token_count: int,
    non_target_token_count: int,
    non_target_sampling: str,
    max_target_count: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[DataLoader]:
    """Create one count dataset loader per sequence length."""

    loaders: list[DataLoader] = []
    for offset, length in enumerate(lengths):
        inputs, labels, target_positions, target_ids = make_count_dataset(
            length=length,
            examples=examples_per_length,
            seed=seed + offset * 100_000,
            target_position_mode=target_position_mode,
            target_token_count=target_token_count,
            non_target_token_count=non_target_token_count,
            non_target_sampling=non_target_sampling,
            max_target_count=max_target_count,
        )
        loaders.append(
            make_count_loader(
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


def iter_count_eval_batches(
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
    max_target_count: int,
    eval_sampling_mode: str,
):
    """Yield deterministic evaluation batches while generating one chunk at a time."""

    if eval_sampling_mode not in {"random", "stratified"}:
        raise ValueError(f"Unsupported eval_sampling_mode: {eval_sampling_mode}")
    if examples < 1:
        raise ValueError("examples must be at least 1.")
    if eval_chunk_examples < 1:
        raise ValueError("eval_chunk_examples must be at least 1.")

    generated = 0
    while generated < examples:
        chunk_examples = min(eval_chunk_examples, examples - generated)
        if eval_sampling_mode == "stratified":
            inputs, labels, target_positions, target_ids = make_count_dataset(
                length=length,
                examples=chunk_examples,
                seed=seed,
                target_position_mode=target_position_mode,
                target_token_count=target_token_count,
                non_target_token_count=non_target_token_count,
                non_target_sampling=non_target_sampling,
                max_target_count=max_target_count,
                label_start_index=generated,
                global_start_index=generated,
            )
        else:
            labels = random_count_class_labels(
                examples=chunk_examples,
                max_target_count=max_target_count,
                seed=seed + 25_000,
                start_index=generated,
            )
            inputs, labels, target_positions, target_ids = make_count_dataset_from_labels(
                length=length,
                count_labels=labels,
                seed=seed,
                target_position_mode=target_position_mode,
                target_token_count=target_token_count,
                non_target_token_count=non_target_token_count,
                non_target_sampling=non_target_sampling,
                max_target_count=max_target_count,
                global_start_index=generated,
            )
        loader = make_count_loader(
            inputs,
            labels,
            target_positions,
            target_ids,
            batch_size=batch_size,
            shuffle=False,
            seed=seed + 50_000 + generated,
        )
        yield from loader
        generated += chunk_examples


class SimplifiedLastQueryAttentionCounter(SimplifiedLastQueryAttentionClassifier):
    """Reduced last-query attention model with a count-class readout."""

    READOUT_MODES = {"softmax_mass", "unnormalized_sum"}

    def __init__(
        self,
        *,
        d_head: int,
        alpha_mode: str,
        alpha_log_scale_init: float,
        target_token_count: int = 1,
        non_target_token_count: int = 1,
        max_target_count: int = 3,
        readout_mode: str = "softmax_mass",
    ) -> None:
        super().__init__(
            d_head=d_head,
            alpha_mode=alpha_mode,
            alpha_log_scale_init=alpha_log_scale_init,
            target_token_count=target_token_count,
            non_target_token_count=non_target_token_count,
        )
        if max_target_count < 1:
            raise ValueError("max_target_count must be at least 1.")
        if readout_mode not in self.READOUT_MODES:
            raise ValueError(f"Unsupported readout_mode: {readout_mode}")
        self.max_target_count = max_target_count
        self.readout_mode = readout_mode
        self.value_dim = target_token_count + 1
        self.num_count_classes = max_target_count + 1
        self.classifier = nn.Linear(self.value_dim, self.num_count_classes)

    def token_value_output(
        self,
        tokens: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-target-type attention mass followed by non-target mass."""

        masses = [
            attention_weights.masked_fill(tokens.ne(token_id), 0.0).sum(dim=1)
            for token_id in range(self.target_token_count)
        ]
        non_target_mass = attention_weights.masked_fill(
            tokens.lt(self.target_token_count),
            0.0,
        ).sum(dim=1)
        return torch.stack(masses + [non_target_mass], dim=1)

    def unnormalized_value_output(
        self,
        tokens: torch.Tensor,
        corrected_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-type unnormalized numerator sums from exponentiated scores."""

        numerator_values = torch.exp(corrected_scores)
        sums = [
            numerator_values.masked_fill(tokens.ne(token_id), 0.0).sum(dim=1)
            for token_id in range(self.target_token_count)
        ]
        non_target_sum = numerator_values.masked_fill(
            tokens.lt(self.target_token_count),
            0.0,
        ).sum(dim=1)
        return torch.stack(sums + [non_target_sum], dim=1)

    def readout_value_output(
        self,
        tokens: torch.Tensor,
        attention_weights: torch.Tensor,
        corrected_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Return the configured classifier readout."""

        if self.readout_mode == "softmax_mass":
            return self.token_value_output(tokens, attention_weights)
        return self.unnormalized_value_output(tokens, corrected_scores)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run last-query attention with the configured Stage 4B value readout."""

        length = tokens.shape[1]
        keys = self.project_tokens(tokens, self.key_projection)
        last_query = self.project_tokens(tokens[:, -1], self.query_projection)
        raw_scores = torch.einsum("bd,bld->bl", last_query, keys) / math.sqrt(self.d_head)
        alpha = self.alpha_for_length(length, tokens.device)
        corrected_scores = alpha * raw_scores
        attention_weights = torch.softmax(corrected_scores, dim=-1)
        normalized_attention_output = self.token_value_output(tokens, attention_weights)
        attention_output = self.readout_value_output(
            tokens,
            attention_weights,
            corrected_scores,
        )
        logits = self.classifier(attention_output)

        if not return_details:
            return logits

        details = {
            "raw_scores": raw_scores,
            "corrected_scores": corrected_scores,
            "attention_weights": attention_weights,
            "attention_output": attention_output,
            "normalized_attention_output": normalized_attention_output,
            "last_query": last_query,
            "alpha": alpha.expand(tokens.shape[0]),
        }
        return logits, details


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=Stage4BConfig.seed)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=Stage4BConfig.device)
    parser.add_argument("--output-dir", type=str, default=Stage4BConfig.output_dir)
    parser.add_argument(
        "--alpha-mode",
        choices=("constant", "log", "learned_log"),
        default=Stage4BConfig.alpha_mode,
    )
    parser.add_argument(
        "--readout-mode",
        choices=("softmax_mass", "unnormalized_sum"),
        default=Stage4BConfig.readout_mode,
    )
    parser.add_argument("--train-length", type=int, default=Stage4BConfig.train_length)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=None)
    parser.add_argument(
        "--target-position-mode",
        choices=("fixed_start", "nonfinal_random"),
        default=Stage4BConfig.target_position_mode,
    )
    parser.add_argument("--target-token-count", type=int, default=Stage4BConfig.target_token_count)
    parser.add_argument(
        "--non-target-token-count",
        type=int,
        default=Stage4BConfig.non_target_token_count,
    )
    parser.add_argument(
        "--non-target-sampling",
        choices=("uniform",),
        default=Stage4BConfig.non_target_sampling,
    )
    parser.add_argument("--max-target-count", type=int, default=Stage4BConfig.max_target_count)
    parser.add_argument("--train-examples", type=int, default=Stage4BConfig.train_examples)
    parser.add_argument("--val-examples", type=int, default=Stage4BConfig.val_examples)
    parser.add_argument("--test-examples", type=int, default=Stage4BConfig.test_examples)
    parser.add_argument(
        "--eval-chunk-examples",
        type=int,
        default=Stage4BConfig.eval_chunk_examples,
    )
    parser.add_argument(
        "--eval-sampling-mode",
        choices=("random", "stratified"),
        default=Stage4BConfig.eval_sampling_mode,
    )
    parser.add_argument("--eval-lengths", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=Stage4BConfig.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=Stage4BConfig.eval_batch_size)
    parser.add_argument("--epochs", type=int, default=Stage4BConfig.epochs)
    parser.add_argument("--max-train-steps", type=int, default=Stage4BConfig.max_train_steps)
    parser.add_argument("--learning-rate", type=float, default=Stage4BConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Stage4BConfig.weight_decay)
    parser.add_argument("--d-head", type=int, default=Stage4BConfig.d_head)
    parser.add_argument(
        "--alpha-log-scale-init",
        type=float,
        default=Stage4BConfig.alpha_log_scale_init,
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Stage4BConfig:
    """Build a Stage 4B config, with tiny overrides for smoke tests."""

    output_dir = args.output_dir
    if args.smoke_test and output_dir == Stage4BConfig.output_dir:
        output_dir = "runs/stage4b_counting_smoke"

    train_lengths = tuple(args.train_lengths) if args.train_lengths else (args.train_length,)
    if args.max_target_count < 1:
        raise ValueError("--max-target-count must be at least 1.")
    if any(length <= args.max_target_count for length in train_lengths):
        raise ValueError("All training lengths must be greater than max_target_count.")
    eval_lengths = (
        tuple(args.eval_lengths)
        if args.eval_lengths is not None
        else Stage4BConfig.eval_lengths
    )
    if any(length <= args.max_target_count for length in eval_lengths):
        raise ValueError("All evaluation lengths must be greater than max_target_count.")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        raise ValueError("--max-train-steps must be positive when provided.")
    if args.readout_mode not in SimplifiedLastQueryAttentionCounter.READOUT_MODES:
        raise ValueError(f"Unsupported readout mode: {args.readout_mode}")
    if args.target_token_count < 1:
        raise ValueError("--target-token-count must be at least 1.")
    if args.non_target_token_count < 1:
        raise ValueError("--non-target-token-count must be at least 1.")
    if args.test_examples < 1:
        raise ValueError("--test-examples must be at least 1.")
    if args.eval_chunk_examples < 1:
        raise ValueError("--eval-chunk-examples must be at least 1.")

    common = dict(
        seed=args.seed,
        device=args.device,
        output_dir=output_dir,
        alpha_mode=args.alpha_mode,
        readout_mode=args.readout_mode,
        train_length=args.train_length,
        train_lengths=train_lengths,
        target_position_mode=args.target_position_mode,
        target_token_count=args.target_token_count,
        non_target_token_count=args.non_target_token_count,
        non_target_sampling=args.non_target_sampling,
        max_target_count=args.max_target_count,
        eval_sampling_mode=args.eval_sampling_mode,
        eval_chunk_examples=args.eval_chunk_examples,
        test_examples=args.test_examples,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        d_head=args.d_head,
        alpha_log_scale_init=args.alpha_log_scale_init,
    )

    if not args.smoke_test:
        return Stage4BConfig(
            train_examples=args.train_examples,
            val_examples=args.val_examples,
            eval_lengths=eval_lengths,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            epochs=args.epochs,
            max_train_steps=args.max_train_steps,
            **common,
        )

    return Stage4BConfig(
        train_examples=64,
        val_examples=32,
        eval_lengths=(10, 20),
        batch_size=16,
        eval_batch_size=8,
        epochs=2,
        max_train_steps=args.max_train_steps if args.max_train_steps is not None else 8,
        **common,
    )


def run_loaders_once(
    model: SimplifiedLastQueryAttentionCounter,
    loaders: list[DataLoader],
    *,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_steps: int | None = None,
    batch_shuffle_seed: int | None = None,
) -> tuple[float, float, int]:
    """Run one pass over count loaders. Returns loss, accuracy, update count."""

    is_training = optimizer is not None
    model.train(is_training)
    batches = [batch for loader in loaders for batch in loader]
    if batch_shuffle_seed is not None:
        random.Random(batch_shuffle_seed).shuffle(batches)

    total_loss = 0.0
    total_examples = 0
    update_count = 0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for tokens, labels, _, _ in batches:
        if is_training and max_steps is not None and update_count >= max_steps:
            break
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
        all_preds.append(logits.detach().argmax(dim=1).cpu())
        all_labels.append(labels.detach().cpu())

    if total_examples == 0:
        return float("nan"), float("nan"), update_count
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return total_loss / total_examples, preds.eq(labels).float().mean().item(), update_count


def _mean_or_nan(values: torch.Tensor) -> float:
    """Return a tensor mean or NaN for empty tensors."""

    return values.mean().item() if values.numel() else float("nan")


@torch.no_grad()
def evaluate_length(
    model: SimplifiedLastQueryAttentionCounter,
    *,
    length: int,
    examples: int,
    batch_size: int,
    seed: int,
    target_position_mode: str,
    target_token_count: int,
    non_target_token_count: int,
    non_target_sampling: str,
    max_target_count: int,
    device: torch.device,
    eval_chunk_examples: int = Stage4BConfig.eval_chunk_examples,
    eval_sampling_mode: str = Stage4BConfig.eval_sampling_mode,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate one length and collect count metrics and diagnostics."""

    model.eval()
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    target_readouts: list[torch.Tensor] = []
    non_target_readouts: list[torch.Tensor] = []
    target_attention_masses: list[torch.Tensor] = []
    non_target_attention_masses: list[torch.Tensor] = []
    pos_min_margins: list[torch.Tensor] = []
    alpha_values: list[torch.Tensor] = []
    max_corrected_scores: list[torch.Tensor] = []
    max_readout_values: list[torch.Tensor] = []
    readout_finite_values: list[torch.Tensor] = []
    target_type_readout_values: dict[int, list[torch.Tensor]] = {
        token_id: [] for token_id in range(target_token_count)
    }
    target_type_attention_mass_values: dict[int, list[torch.Tensor]] = {
        token_id: [] for token_id in range(target_token_count)
    }
    target_type_present_values: dict[int, list[torch.Tensor]] = {
        token_id: [] for token_id in range(target_token_count)
    }

    non_target_id_tensor = torch.tensor(
        non_target_token_ids(target_token_count, non_target_token_count),
        device=device,
        dtype=torch.long,
    )

    for tokens, labels, _, target_ids in iter_count_eval_batches(
        length=length,
        examples=examples,
        eval_chunk_examples=eval_chunk_examples,
        batch_size=batch_size,
        seed=seed,
        target_position_mode=target_position_mode,
        target_token_count=target_token_count,
        non_target_token_count=non_target_token_count,
        non_target_sampling=non_target_sampling,
        max_target_count=max_target_count,
        eval_sampling_mode=eval_sampling_mode,
    ):
        tokens = tokens.to(device)
        labels = labels.to(device)
        target_ids = target_ids.to(device)
        logits, details = model(tokens, return_details=True)
        preds = logits.argmax(dim=1)
        readout_output = details["attention_output"]
        normalized_attention_output = details["normalized_attention_output"]
        target_readout = readout_output[:, :target_token_count].sum(dim=1)
        non_target_readout = readout_output[:, target_token_count]
        target_attention_mass = normalized_attention_output[:, :target_token_count].sum(dim=1)
        non_target_attention_mass = normalized_attention_output[:, target_token_count]

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        target_readouts.append(target_readout.cpu())
        non_target_readouts.append(non_target_readout.cpu())
        target_attention_masses.append(target_attention_mass.cpu())
        non_target_attention_masses.append(non_target_attention_mass.cpu())
        alpha_values.append(details["alpha"].cpu())
        max_corrected_scores.append(details["corrected_scores"].amax(dim=1).cpu())
        max_readout_values.append(readout_output.detach().amax(dim=1).cpu())
        readout_finite_values.append(
            torch.isfinite(readout_output.detach()).all(dim=1).to(dtype=torch.float32).cpu()
        )

        for token_id in range(target_token_count):
            target_type_readout_values[token_id].append(readout_output[:, token_id].cpu())
            target_type_attention_mass_values[token_id].append(
                normalized_attention_output[:, token_id].cpu()
            )
            present = target_ids.eq(token_id).any(dim=1)
            target_type_present_values[token_id].append(present.cpu())

        pos_mask = labels.gt(0)
        if pos_mask.any():
            last_query = details["last_query"][pos_mask]
            positive_target_ids = target_ids[pos_mask]
            target_type_keys = model.key_vectors_for_token_ids(
                torch.arange(target_token_count, device=device, dtype=torch.long)
            )
            non_target_keys = model.key_vectors_for_token_ids(non_target_id_tensor)
            target_type_scores = (
                torch.einsum("bd,kd->bk", last_query, target_type_keys)
                / math.sqrt(model.d_head)
            )
            non_target_scores = (
                torch.einsum("bd,kd->bk", last_query, non_target_keys)
                / math.sqrt(model.d_head)
            )
            valid_target_mask = positive_target_ids.ge(0)
            safe_target_ids = positive_target_ids.clamp_min(0)
            occurrence_scores = target_type_scores.gather(1, safe_target_ids)
            occurrence_margins = (
                occurrence_scores.unsqueeze(2) - non_target_scores.unsqueeze(1)
            )
            occurrence_margins = occurrence_margins.masked_fill(
                ~valid_target_mask.unsqueeze(2),
                float("inf"),
            )
            pos_min_margins.append(
                occurrence_margins.view(occurrence_margins.shape[0], -1).min(dim=1).values.cpu()
            )

    preds_cpu = torch.cat(all_preds)
    labels_cpu = torch.cat(all_labels)
    target_readout_cpu = torch.cat(target_readouts)
    non_target_readout_cpu = torch.cat(non_target_readouts)
    target_attention_mass_cpu = torch.cat(target_attention_masses)
    non_target_attention_mass_cpu = torch.cat(non_target_attention_masses)
    alpha_cpu = torch.cat(alpha_values)
    max_corrected_score_cpu = torch.cat(max_corrected_scores)
    max_readout_value_cpu = torch.cat(max_readout_values)
    readout_finite_cpu = torch.cat(readout_finite_values)
    errors = (preds_cpu - labels_cpu).abs()
    learned_coefficient = model.learned_alpha_coefficient()
    pos_min_margins_cpu = (
        torch.cat(pos_min_margins) if pos_min_margins else torch.empty(0)
    )
    mean_margin = _mean_or_nan(pos_min_margins_cpu)
    worst_margin = pos_min_margins_cpu.min().item() if pos_min_margins_cpu.numel() else float("nan")
    is_learned_log = model.alpha_mode == "learned_log"

    metrics_row: dict[str, Any] = {
        "length": length,
        "split": "test",
        "alpha_mode": model.alpha_mode,
        "readout_mode": model.readout_mode,
        "target_position_mode": target_position_mode,
        "target_token_count": target_token_count,
        "non_target_token_count": non_target_token_count,
        "non_target_sampling": non_target_sampling,
        "max_target_count": max_target_count,
        "num_classes": max_target_count + 1,
        "test_examples": examples,
        "eval_chunk_examples": eval_chunk_examples,
        "eval_sampling_mode": eval_sampling_mode,
        "alpha_value": _mean_or_nan(alpha_cpu),
        "accuracy": preds_cpu.eq(labels_cpu).float().mean().item(),
        "mean_predicted_count": preds_cpu.to(dtype=torch.float32).mean().item(),
        "mean_true_count": labels_cpu.to(dtype=torch.float32).mean().item(),
        "mean_absolute_count_error": errors.to(dtype=torch.float32).mean().item(),
        "mean_target_readout": _mean_or_nan(target_readout_cpu),
        "mean_non_target_readout": _mean_or_nan(non_target_readout_cpu),
        "mean_target_attention_mass": _mean_or_nan(target_attention_mass_cpu),
        "mean_non_target_attention_mass": _mean_or_nan(non_target_attention_mass_cpu),
        "mean_max_corrected_score": _mean_or_nan(max_corrected_score_cpu),
        "max_corrected_score": max_corrected_score_cpu.max().item(),
        "mean_max_readout_value": _mean_or_nan(max_readout_value_cpu),
        "max_readout_value": max_readout_value_cpu.max().item(),
        "readout_finite_fraction": _mean_or_nan(readout_finite_cpu),
        "mean_min_margin_delta": mean_margin,
        "worst_min_margin_delta": worst_margin,
        "learned_alpha_coefficient": learned_coefficient,
        "learned_log_c_delta_min_mean": (
            learned_coefficient * mean_margin if is_learned_log else float("nan")
        ),
        "learned_log_c_delta_min_worst": (
            learned_coefficient * worst_margin if is_learned_log else float("nan")
        ),
    }

    count_rows: list[dict[str, Any]] = []
    for count_value in range(max_target_count + 1):
        mask = labels_cpu.eq(count_value)
        if not mask.any():
            continue
        count_rows.append(
            {
                "length": length,
                "split": "test",
                "alpha_mode": model.alpha_mode,
                "readout_mode": model.readout_mode,
                "target_position_mode": target_position_mode,
                "target_token_count": target_token_count,
                "non_target_token_count": non_target_token_count,
                "max_target_count": max_target_count,
                "test_examples": examples,
                "eval_chunk_examples": eval_chunk_examples,
                "eval_sampling_mode": eval_sampling_mode,
                "true_count": count_value,
                "examples": int(mask.sum().item()),
                "recall": preds_cpu[mask].eq(count_value).float().mean().item(),
                "mean_predicted_count": preds_cpu[mask].to(dtype=torch.float32).mean().item(),
                "mean_absolute_count_error": errors[mask].to(dtype=torch.float32).mean().item(),
                "mean_target_readout": _mean_or_nan(target_readout_cpu[mask]),
                "mean_non_target_readout": _mean_or_nan(non_target_readout_cpu[mask]),
                "mean_target_attention_mass": _mean_or_nan(
                    target_attention_mass_cpu[mask]
                ),
                "mean_non_target_attention_mass": _mean_or_nan(
                    non_target_attention_mass_cpu[mask]
                ),
            }
        )

    confusion_rows: list[dict[str, Any]] = []
    for true_count in range(max_target_count + 1):
        true_mask = labels_cpu.eq(true_count)
        true_total = int(true_mask.sum().item())
        if true_total == 0:
            continue
        for predicted_count in range(max_target_count + 1):
            cell = int((true_mask & preds_cpu.eq(predicted_count)).sum().item())
            confusion_rows.append(
                {
                    "length": length,
                    "split": "test",
                    "alpha_mode": model.alpha_mode,
                    "readout_mode": model.readout_mode,
                    "true_count": true_count,
                    "predicted_count": predicted_count,
                    "examples": cell,
                    "fraction_of_true_count": cell / true_total,
                }
            )

    target_type_rows: list[dict[str, Any]] = []
    for token_id in range(target_token_count):
        readout = torch.cat(target_type_readout_values[token_id])
        mass = torch.cat(target_type_attention_mass_values[token_id])
        present = torch.cat(target_type_present_values[token_id])
        target_type_rows.append(
            {
                "length": length,
                "split": "test",
                "alpha_mode": model.alpha_mode,
                "readout_mode": model.readout_mode,
                "target_token_id": token_id,
                "examples": int(readout.numel()),
                "present_examples": int(present.sum().item()),
                "mean_readout": _mean_or_nan(readout),
                "mean_readout_when_present": _mean_or_nan(readout[present]),
                "mean_attention_mass": _mean_or_nan(mass),
                "mean_attention_mass_when_present": _mean_or_nan(mass[present]),
            }
        )

    return [metrics_row], count_rows, confusion_rows, target_type_rows


def train_model(
    config: Stage4BConfig,
    *,
    device: torch.device,
    output_dir: Path,
) -> tuple[SimplifiedLastQueryAttentionCounter, int]:
    """Train the Stage 4B counting model."""

    train_lengths = resolved_train_lengths(config)
    loader_kwargs = dict(
        lengths=train_lengths,
        target_position_mode=config.target_position_mode,
        target_token_count=config.target_token_count,
        non_target_token_count=config.non_target_token_count,
        non_target_sampling=config.non_target_sampling,
        max_target_count=config.max_target_count,
    )
    train_loaders = make_count_multilength_loaders(
        examples_per_length=config.train_examples,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed + 1,
        **loader_kwargs,
    )
    val_loaders = make_count_multilength_loaders(
        examples_per_length=config.val_examples,
        batch_size=config.eval_batch_size,
        shuffle=False,
        seed=config.seed + 2,
        **loader_kwargs,
    )

    model = SimplifiedLastQueryAttentionCounter(
        d_head=config.d_head,
        alpha_mode=config.alpha_mode,
        alpha_log_scale_init=config.alpha_log_scale_init,
        target_token_count=config.target_token_count,
        non_target_token_count=config.non_target_token_count,
        max_target_count=config.max_target_count,
        readout_mode=config.readout_mode,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history_rows: list[dict[str, Any]] = []
    optimizer_updates = 0
    epoch = 0
    while True:
        if config.max_train_steps is None and epoch >= config.epochs:
            break
        if config.max_train_steps is not None and optimizer_updates >= config.max_train_steps:
            break
        epoch += 1
        remaining = (
            None if config.max_train_steps is None else config.max_train_steps - optimizer_updates
        )
        train_loss, train_acc, updates = run_loaders_once(
            model,
            train_loaders,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_steps=remaining,
            batch_shuffle_seed=config.seed + 20_000 + epoch,
        )
        optimizer_updates += updates
        val_loss, val_acc, _ = run_loaders_once(
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
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

    write_csv(output_dir / "train_history.csv", history_rows)
    return model, optimizer_updates


def run_evaluation(
    model: SimplifiedLastQueryAttentionCounter,
    config: Stage4BConfig,
    *,
    device: torch.device,
    output_dir: Path,
    optimizer_updates: int,
) -> None:
    """Run configured Stage 4B evaluation lengths and write metric CSVs."""

    metric_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    target_type_rows: list[dict[str, Any]] = []
    for length in config.eval_lengths:
        rows, per_count, confusion, target_type = evaluate_length(
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
            max_target_count=config.max_target_count,
            device=device,
        )
        for row in rows:
            row["optimizer_updates"] = optimizer_updates
        metric_rows.extend(rows)
        count_rows.extend(per_count)
        confusion_rows.extend(confusion)
        target_type_rows.extend(target_type)

    write_csv(output_dir / "metrics_by_length.csv", metric_rows)
    write_csv(output_dir / "count_metrics.csv", count_rows)
    write_csv(output_dir / "count_confusion_matrix.csv", confusion_rows)
    if target_type_rows:
        write_csv(output_dir / "target_type_metrics.csv", target_type_rows)


def main() -> None:
    """Run the Stage 4B counting experiment."""

    args = parse_args()
    config = build_config(args)
    set_reproducibility(config.seed)
    device = resolve_device(config.device)
    output_dir = resolve_output_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_lengths = resolved_train_lengths(config)

    model, optimizer_updates = train_model(config, device=device, output_dir=output_dir)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "d_head": config.d_head,
            "alpha_mode": config.alpha_mode,
            "readout_mode": config.readout_mode,
            "alpha_log_scale_init": config.alpha_log_scale_init,
            "target_token_count": config.target_token_count,
            "non_target_token_count": config.non_target_token_count,
            "max_target_count": config.max_target_count,
            "num_count_classes": model.num_count_classes,
            "optimizer_updates": optimizer_updates,
        },
        output_dir / "model.pt",
    )
    write_json(
        output_dir / "config.json",
        {
            **asdict(config),
            "eval_lengths": list(config.eval_lengths),
            "train_lengths": list(train_lengths),
            "optimizer_updates": optimizer_updates,
            "resolved_device": str(device),
            "target_token_ids": target_token_ids(config.target_token_count),
            "non_target_token_ids": non_target_token_ids(
                config.target_token_count,
                config.non_target_token_count,
            ),
            "count_class_ids": list(range(config.max_target_count + 1)),
        },
    )

    run_evaluation(
        model,
        config,
        device=device,
        output_dir=output_dir,
        optimizer_updates=optimizer_updates,
    )
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
