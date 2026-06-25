"""Train Stage 4A: non-binary target classification on the reduced model.

Stage 4A keeps the Stage 3 reduced last-query attention model but changes the output.
Instead of deciding target present vs absent, the model reports which target token type
is present, or a dedicated ``n`` class if no target is present. The value pathway carries
per-target-type attention mass, and the head is a multi-class classifier over ``H + 1``
classes (``H`` target types plus ``n``).

The dataset, chunked evaluation, stratified evaluation, and length-aware ``alpha`` modes
are reused from ``stage3_simplified_attention`` so this module only adds the multi-class
value pathway, head, loss, and metrics.
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
from torch.utils.data import DataLoader

from stage3_simplified_attention import (
    SimplifiedLastQueryAttentionClassifier,
    iter_eval_batches,
    make_multilength_loaders,
    non_target_token_ids,
    project_dir,
    resolve_device,
    set_reproducibility,
    target_token_ids,
    write_csv,
    write_json,
)


@dataclass(frozen=True)
class Stage4AConfig:
    """Configuration for the Stage 4A non-binary classification experiment."""

    seed: int = 42
    device: str = "auto"
    output_dir: str = "runs/stage4a_nonbinary"
    alpha_mode: str = "constant"
    train_length: int = 10
    train_lengths: tuple[int, ...] = ()
    target_position_mode: str = "fixed_start"
    target_token_count: int = 3
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


def resolved_train_lengths(config: Stage4AConfig) -> tuple[int, ...]:
    """Return the active training lengths, preserving single-length compatibility."""

    return config.train_lengths if config.train_lengths else (config.train_length,)


class SimplifiedLastQueryAttentionMultiClass(SimplifiedLastQueryAttentionClassifier):
    """Reduced last-query attention model with a multi-class identity readout.

    The query/key projections and the length-aware ``alpha`` modes are inherited unchanged.
    Only the value pathway and the head differ: the value output is the per-target-type
    attention mass plus the total non-target mass, and the head is a multi-class classifier.
    """

    def __init__(
        self,
        *,
        d_head: int,
        alpha_mode: str,
        alpha_log_scale_init: float,
        target_token_count: int = 3,
        non_target_token_count: int = 1,
    ) -> None:
        super().__init__(
            d_head=d_head,
            alpha_mode=alpha_mode,
            alpha_log_scale_init=alpha_log_scale_init,
            target_token_count=target_token_count,
            non_target_token_count=non_target_token_count,
        )
        # Classes are the H target token types plus a dedicated "no target" class.
        self.num_classes = target_token_count + 1
        self.classifier = nn.Linear(self.num_classes, self.num_classes)

    def token_value_output(
        self,
        tokens: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-target-type attention mass followed by total non-target mass.

        Output shape is (batch, target_token_count + 1). The columns are
        [mass_0, ..., mass_{H-1}, nontarget_mass]. With H = 1 this reduces to
        [target_mass, 1 - target_mass], matching the Stage 3 binary value output.
        """

        masses = [
            attention_weights.masked_fill(tokens.ne(token_id), 0.0).sum(dim=1)
            for token_id in range(self.target_token_count)
        ]
        non_target_mass = attention_weights.masked_fill(
            tokens.lt(self.target_token_count),
            0.0,
        ).sum(dim=1)
        return torch.stack(masses + [non_target_mass], dim=1)


def none_class_index(target_token_count: int) -> int:
    """Return the class index used for the no-target (n) class."""

    return target_token_count


def class_labels_from_presence(
    presence: torch.Tensor,
    target_ids: torch.Tensor,
    target_token_count: int,
) -> torch.Tensor:
    """Map (presence, target id) into a multi-class label.

    Positive examples take their target token id as the label. Negative examples take
    the dedicated no-target class index.
    """

    none_class = torch.full_like(target_ids, none_class_index(target_token_count))
    return torch.where(presence.eq(1), target_ids, none_class)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=Stage4AConfig.seed)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=Stage4AConfig.device)
    parser.add_argument("--output-dir", type=str, default=Stage4AConfig.output_dir)
    parser.add_argument(
        "--alpha-mode",
        choices=("constant", "log", "learned_log"),
        default=Stage4AConfig.alpha_mode,
    )
    parser.add_argument("--train-length", type=int, default=Stage4AConfig.train_length)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=None)
    parser.add_argument(
        "--target-position-mode",
        choices=("fixed_start", "nonfinal_random"),
        default=Stage4AConfig.target_position_mode,
    )
    parser.add_argument("--target-token-count", type=int, default=Stage4AConfig.target_token_count)
    parser.add_argument(
        "--non-target-token-count",
        type=int,
        default=Stage4AConfig.non_target_token_count,
    )
    parser.add_argument(
        "--non-target-sampling",
        choices=("uniform",),
        default=Stage4AConfig.non_target_sampling,
    )
    parser.add_argument("--train-examples", type=int, default=Stage4AConfig.train_examples)
    parser.add_argument("--val-examples", type=int, default=Stage4AConfig.val_examples)
    parser.add_argument("--test-examples", type=int, default=Stage4AConfig.test_examples)
    parser.add_argument(
        "--eval-chunk-examples",
        type=int,
        default=Stage4AConfig.eval_chunk_examples,
    )
    parser.add_argument(
        "--eval-sampling-mode",
        choices=("random", "stratified"),
        default=Stage4AConfig.eval_sampling_mode,
    )
    parser.add_argument(
        "--eval-lengths",
        type=int,
        nargs="+",
        default=list(Stage4AConfig.eval_lengths),
    )
    parser.add_argument("--batch-size", type=int, default=Stage4AConfig.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=Stage4AConfig.eval_batch_size)
    parser.add_argument("--epochs", type=int, default=Stage4AConfig.epochs)
    parser.add_argument("--max-train-steps", type=int, default=Stage4AConfig.max_train_steps)
    parser.add_argument("--learning-rate", type=float, default=Stage4AConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Stage4AConfig.weight_decay)
    parser.add_argument("--d-head", type=int, default=Stage4AConfig.d_head)
    parser.add_argument(
        "--alpha-log-scale-init",
        type=float,
        default=Stage4AConfig.alpha_log_scale_init,
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Stage4AConfig:
    """Build the experiment config, with tiny overrides for smoke tests."""

    output_dir = args.output_dir
    if args.smoke_test and output_dir == Stage4AConfig.output_dir:
        output_dir = "runs/stage4a_nonbinary_smoke"

    train_lengths = tuple(args.train_lengths) if args.train_lengths else (args.train_length,)
    if any(length < 2 for length in train_lengths):
        raise ValueError("All training lengths must be at least 2.")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        raise ValueError("--max-train-steps must be positive when provided.")
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
        train_length=args.train_length,
        train_lengths=train_lengths,
        target_position_mode=args.target_position_mode,
        target_token_count=args.target_token_count,
        non_target_token_count=args.non_target_token_count,
        non_target_sampling=args.non_target_sampling,
        eval_sampling_mode=args.eval_sampling_mode,
        eval_chunk_examples=args.eval_chunk_examples,
        test_examples=args.test_examples,
        max_train_steps=args.max_train_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        d_head=args.d_head,
        alpha_log_scale_init=args.alpha_log_scale_init,
    )

    if not args.smoke_test:
        return Stage4AConfig(
            train_examples=args.train_examples,
            val_examples=args.val_examples,
            eval_lengths=tuple(args.eval_lengths),
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            epochs=args.epochs,
            **common,
        )

    return Stage4AConfig(
        train_examples=64,
        val_examples=32,
        eval_lengths=(10, 20),
        batch_size=16,
        eval_batch_size=20,
        epochs=2,
        **common,
    )


def run_loaders_once(
    model: SimplifiedLastQueryAttentionMultiClass,
    loaders: list[DataLoader],
    *,
    criterion: nn.Module,
    device: torch.device,
    target_token_count: int,
    optimizer: torch.optim.Optimizer | None = None,
    max_steps: int | None = None,
    batch_shuffle_seed: int | None = None,
) -> tuple[float, float, int]:
    """Run one pass over length-specific loaders. Returns loss, accuracy, update count."""

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

    for tokens, presence, _, target_ids in batches:
        if is_training and max_steps is not None and update_count >= max_steps:
            break
        tokens = tokens.to(device)
        labels = class_labels_from_presence(
            presence.to(device),
            target_ids.to(device),
            target_token_count,
        )
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
    accuracy = preds.eq(labels).float().mean().item()
    return total_loss / total_examples, accuracy, update_count


def _mean_or_nan(values: torch.Tensor) -> float:
    """Return a tensor mean or NaN for empty tensors."""

    return values.mean().item() if values.numel() else float("nan")


@torch.no_grad()
def evaluate_length(
    model: SimplifiedLastQueryAttentionMultiClass,
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
    eval_chunk_examples: int = Stage4AConfig.eval_chunk_examples,
    eval_sampling_mode: str = Stage4AConfig.eval_sampling_mode,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one length and collect multi-class and mechanism metrics."""

    if examples < 1:
        raise ValueError("examples must be at least 1.")
    model.eval()
    none_class = none_class_index(target_token_count)

    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    pos_preds: list[torch.Tensor] = []
    pos_target_ids: list[torch.Tensor] = []
    pos_target_attn: list[torch.Tensor] = []
    pos_min_margins: list[torch.Tensor] = []
    alpha_values: list[torch.Tensor] = []

    non_target_id_tensor = torch.tensor(
        non_target_token_ids(target_token_count, non_target_token_count),
        device=device,
        dtype=torch.long,
    )

    for tokens, presence, target_positions, target_ids in iter_eval_batches(
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
        presence = presence.to(device)
        target_positions = target_positions.to(device)
        target_ids = target_ids.to(device)
        labels = class_labels_from_presence(presence, target_ids, target_token_count)
        logits, details = model(tokens, return_details=True)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        alpha_values.append(details["alpha"].cpu())

        pos_mask = presence.eq(1)
        if pos_mask.any():
            attn = details["attention_weights"][pos_mask]
            tpos = target_positions[pos_mask]
            pos_target_attn.append(attn.gather(1, tpos.unsqueeze(1)).squeeze(1).cpu())
            pos_preds.append(preds[pos_mask].cpu())
            pos_target_ids.append(target_ids[pos_mask].cpu())

            raw_scores = details["raw_scores"][pos_mask]
            last_query = details["last_query"][pos_mask]
            target_score = raw_scores.gather(1, tpos.unsqueeze(1)).squeeze(1)
            non_target_keys = model.key_vectors_for_token_ids(non_target_id_tensor)
            non_target_scores = (
                torch.einsum("bd,kd->bk", last_query, non_target_keys) / math.sqrt(model.d_head)
            )
            type_margins = target_score.unsqueeze(1) - non_target_scores
            pos_min_margins.append(type_margins.min(dim=1).values.cpu())

    preds_cpu = torch.cat(all_preds)
    labels_cpu = torch.cat(all_labels)
    accuracy = preds_cpu.eq(labels_cpu).float().mean().item()
    negative_mask = labels_cpu.eq(none_class)
    none_accuracy = (
        preds_cpu[negative_mask].eq(none_class).float().mean().item()
        if negative_mask.any()
        else float("nan")
    )

    pos_preds_cpu = torch.cat(pos_preds) if pos_preds else torch.empty(0, dtype=torch.long)
    pos_target_ids_cpu = (
        torch.cat(pos_target_ids) if pos_target_ids else torch.empty(0, dtype=torch.long)
    )
    pos_target_attn_cpu = (
        torch.cat(pos_target_attn) if pos_target_attn else torch.empty(0)
    )
    pos_min_margins_cpu = (
        torch.cat(pos_min_margins) if pos_min_margins else torch.empty(0)
    )

    positive_count = pos_preds_cpu.numel()
    if positive_count:
        correct_pos = pos_preds_cpu.eq(pos_target_ids_cpu)
        frac_correct = correct_pos.float().mean().item()
        frac_pred_none = pos_preds_cpu.eq(none_class).float().mean().item()
        frac_pred_other = (
            (~correct_pos & pos_preds_cpu.ne(none_class)).float().mean().item()
        )
    else:
        frac_correct = frac_pred_none = frac_pred_other = float("nan")

    learned_coefficient = model.learned_alpha_coefficient()
    worst_margin = pos_min_margins_cpu.min().item() if positive_count else float("nan")
    mean_margin = _mean_or_nan(pos_min_margins_cpu)
    is_learned_log = model.alpha_mode == "learned_log"

    row: dict[str, Any] = {
        "length": length,
        "split": "test",
        "alpha_mode": model.alpha_mode,
        "target_position_mode": target_position_mode,
        "target_token_count": target_token_count,
        "non_target_token_count": non_target_token_count,
        "non_target_sampling": non_target_sampling,
        "num_classes": none_class + 1,
        "test_examples": examples,
        "eval_chunk_examples": eval_chunk_examples,
        "eval_sampling_mode": eval_sampling_mode,
        "positive_examples": positive_count,
        "negative_examples": int(negative_mask.sum().item()),
        "alpha_value": _mean_or_nan(torch.cat(alpha_values)),
        "accuracy": accuracy,
        "none_class_accuracy": none_accuracy,
        "positive_correct_fraction": frac_correct,
        "positive_predicted_none_fraction": frac_pred_none,
        "positive_predicted_other_target_fraction": frac_pred_other,
        "mean_target_attention": _mean_or_nan(pos_target_attn_cpu),
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

    target_type_rows: list[dict[str, Any]] = []
    for token_id in range(target_token_count):
        type_mask = pos_target_ids_cpu.eq(token_id)
        if not type_mask.any():
            continue
        type_margins = pos_min_margins_cpu[type_mask]
        type_worst = type_margins.min().item()
        type_mean = _mean_or_nan(type_margins)
        target_type_rows.append(
            {
                "length": length,
                "split": "test",
                "alpha_mode": model.alpha_mode,
                "target_position_mode": target_position_mode,
                "target_token_count": target_token_count,
                "non_target_token_count": non_target_token_count,
                "test_examples": examples,
                "eval_chunk_examples": eval_chunk_examples,
                "eval_sampling_mode": eval_sampling_mode,
                "target_token_id": token_id,
                "positive_examples": int(type_mask.sum().item()),
                "recall": pos_preds_cpu[type_mask].eq(token_id).float().mean().item(),
                "mean_target_attention": _mean_or_nan(pos_target_attn_cpu[type_mask]),
                "mean_min_margin": type_mean,
                "worst_min_margin": type_worst,
                "mean_c_delta_min": (
                    learned_coefficient * type_mean if is_learned_log else float("nan")
                ),
                "worst_c_delta_min": (
                    learned_coefficient * type_worst if is_learned_log else float("nan")
                ),
            }
        )
    return row, target_type_rows


def train_model(
    config: Stage4AConfig,
    *,
    device: torch.device,
    output_dir: Path,
) -> tuple[SimplifiedLastQueryAttentionMultiClass, int]:
    """Train the multi-class reduced model."""

    train_lengths = resolved_train_lengths(config)
    loader_kwargs = dict(
        lengths=train_lengths,
        target_position_mode=config.target_position_mode,
        target_token_count=config.target_token_count,
        non_target_token_count=config.non_target_token_count,
        non_target_sampling=config.non_target_sampling,
    )
    train_loaders = make_multilength_loaders(
        examples_per_length=config.train_examples,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed + 1,
        **loader_kwargs,
    )
    val_loaders = make_multilength_loaders(
        examples_per_length=config.val_examples,
        batch_size=config.eval_batch_size,
        shuffle=False,
        seed=config.seed + 2,
        **loader_kwargs,
    )

    model = SimplifiedLastQueryAttentionMultiClass(
        d_head=config.d_head,
        alpha_mode=config.alpha_mode,
        alpha_log_scale_init=config.alpha_log_scale_init,
        target_token_count=config.target_token_count,
        non_target_token_count=config.non_target_token_count,
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
            target_token_count=config.target_token_count,
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
            target_token_count=config.target_token_count,
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


def main() -> None:
    """Run the Stage 4A non-binary classification experiment."""

    args = parse_args()
    config = build_config(args)
    set_reproducibility(config.seed)
    device = resolve_device(config.device)
    output_dir = project_dir() / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_lengths = resolved_train_lengths(config)

    model, optimizer_updates = train_model(config, device=device, output_dir=output_dir)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "d_head": config.d_head,
            "alpha_mode": config.alpha_mode,
            "alpha_log_scale_init": config.alpha_log_scale_init,
            "target_token_count": config.target_token_count,
            "non_target_token_count": config.non_target_token_count,
            "num_classes": model.num_classes,
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
            "none_class_index": none_class_index(config.target_token_count),
        },
    )

    rows: list[dict[str, Any]] = []
    target_type_rows: list[dict[str, Any]] = []
    for length in config.eval_lengths:
        row, type_rows = evaluate_length(
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
        row["optimizer_updates"] = optimizer_updates
        rows.append(row)
        target_type_rows.extend(type_rows)

    write_csv(output_dir / "metrics_by_length.csv", rows)
    if target_type_rows:
        write_csv(output_dir / "target_type_metrics.csv", target_type_rows)

    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
