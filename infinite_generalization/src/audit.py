"""Audit example export utilities for sequence-classification experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch import nn

from config import Stage0Config, Stage1Config, Stage2AConfig, TaskConfig
from data import make_balanced_token_presence_dataset


def preview_sequence(tokens: torch.Tensor, *, max_edge_tokens: int) -> str:
    """Return a compact sequence preview that remains readable for long inputs."""

    values = [int(token) for token in tokens.tolist()]
    if len(values) <= max_edge_tokens * 2:
        return " ".join(str(value) for value in values)

    left = " ".join(str(value) for value in values[:max_edge_tokens])
    right = " ".join(str(value) for value in values[-max_edge_tokens:])
    return f"{left} ... {right}"


def target_positions(tokens: torch.Tensor, *, target_token: int) -> str:
    """Return target-token positions as a compact JSON-style list string."""

    positions = torch.nonzero(tokens.eq(target_token), as_tuple=False).flatten().tolist()
    return json.dumps([int(position) for position in positions])


def select_audit_indices(
    labels: torch.Tensor,
    *,
    examples_per_class: int,
) -> list[int]:
    """Select a stable set of negative and positive row indices for auditing."""

    selected: list[int] = []
    for label_value in (0.0, 1.0):
        matches = torch.nonzero(labels.eq(label_value), as_tuple=False).flatten()
        selected.extend(int(index) for index in matches[:examples_per_class].tolist())
    return selected


@torch.no_grad()
def build_audit_rows(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    split: str,
    length: int,
    task: TaskConfig,
    device: torch.device,
    examples_per_class: int,
    preview_tokens: int,
) -> list[dict[str, object]]:
    """Create audit rows with raw examples and model outputs."""

    model.eval()
    indices = select_audit_indices(labels, examples_per_class=examples_per_class)
    if not indices:
        return []

    batch_inputs = inputs[indices].to(device)
    logits = model(batch_inputs).cpu()
    probabilities = torch.sigmoid(logits)
    predictions = (logits >= 0).long()

    rows: list[dict[str, object]] = []
    for row_id, source_index in enumerate(indices):
        tokens = inputs[source_index]
        label = int(labels[source_index].item())
        prediction = int(predictions[row_id].item())
        rows.append(
            {
                "split": split,
                "length": length,
                "dataset_index": source_index,
                "label": label,
                "prediction": prediction,
                "correct": prediction == label,
                "logit": float(logits[row_id].item()),
                "probability": float(probabilities[row_id].item()),
                "target_count": int(tokens.eq(task.target_token).sum().item()),
                "target_positions": target_positions(tokens, target_token=task.target_token),
                "sequence_preview": preview_sequence(tokens, max_edge_tokens=preview_tokens),
            }
        )
    return rows


def write_audit_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write audit examples and model outputs to CSV."""

    fieldnames = [
        "split",
        "length",
        "dataset_index",
        "label",
        "prediction",
        "correct",
        "logit",
        "probability",
        "target_count",
        "target_positions",
        "sequence_preview",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_audit_examples(
    model: nn.Module,
    *,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    val_inputs: torch.Tensor,
    val_labels: torch.Tensor,
    task: TaskConfig,
    config: Stage0Config | Stage1Config,
    device: torch.device,
    output_dir: Path,
    examples_per_class: int,
    preview_tokens: int,
) -> None:
    """Save train, validation, and length-sweep test audit examples."""

    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    train_rows = build_audit_rows(
        model,
        train_inputs,
        train_labels,
        split="train",
        length=task.train_length,
        task=task,
        device=device,
        examples_per_class=examples_per_class,
        preview_tokens=preview_tokens,
    )
    val_rows = build_audit_rows(
        model,
        val_inputs,
        val_labels,
        split="val",
        length=task.train_length,
        task=task,
        device=device,
        examples_per_class=examples_per_class,
        preview_tokens=preview_tokens,
    )

    test_rows: list[dict[str, object]] = []
    for index, length in enumerate(task.eval_lengths):
        generator = torch.Generator().manual_seed(config.seed + 10_000 + index)
        test_inputs, test_labels = make_balanced_token_presence_dataset(
            num_examples=config.test_examples,
            length=length,
            task=task,
            generator=generator,
        )
        test_rows.extend(
            build_audit_rows(
                model,
                test_inputs,
                test_labels,
                split="test",
                length=length,
                task=task,
                device=device,
                examples_per_class=examples_per_class,
                preview_tokens=preview_tokens,
            )
        )

    write_audit_csv(examples_dir / "train_examples.csv", train_rows)
    write_audit_csv(examples_dir / "val_examples.csv", val_rows)
    write_audit_csv(examples_dir / "test_examples_by_length.csv", test_rows)


def save_multilength_audit_examples(
    model: nn.Module,
    *,
    task: TaskConfig,
    config: Stage2AConfig,
    device: torch.device,
    output_dir: Path,
    examples_per_class: int,
    preview_tokens: int,
) -> None:
    """Save audit examples for Stage 2A multi-length train, validation, and test splits."""

    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    for length_index, length in enumerate(config.train_lengths):
        train_generator = torch.Generator().manual_seed(config.seed + length_index)
        train_inputs, train_labels = make_balanced_token_presence_dataset(
            num_examples=config.train_examples_per_length,
            length=length,
            task=task,
            generator=train_generator,
        )
        train_rows.extend(
            build_audit_rows(
                model,
                train_inputs,
                train_labels,
                split="train",
                length=length,
                task=task,
                device=device,
                examples_per_class=examples_per_class,
                preview_tokens=preview_tokens,
            )
        )

        val_generator = torch.Generator().manual_seed(config.seed + 20_000 + length_index)
        val_inputs, val_labels = make_balanced_token_presence_dataset(
            num_examples=config.val_examples_per_length,
            length=length,
            task=task,
            generator=val_generator,
        )
        val_rows.extend(
            build_audit_rows(
                model,
                val_inputs,
                val_labels,
                split="val",
                length=length,
                task=task,
                device=device,
                examples_per_class=examples_per_class,
                preview_tokens=preview_tokens,
            )
        )

    test_rows: list[dict[str, object]] = []
    for index, length in enumerate(task.eval_lengths):
        generator = torch.Generator().manual_seed(config.seed + 10_000 + index)
        test_inputs, test_labels = make_balanced_token_presence_dataset(
            num_examples=config.test_examples,
            length=length,
            task=task,
            generator=generator,
        )
        test_rows.extend(
            build_audit_rows(
                model,
                test_inputs,
                test_labels,
                split="test",
                length=length,
                task=task,
                device=device,
                examples_per_class=examples_per_class,
                preview_tokens=preview_tokens,
            )
        )

    write_audit_csv(examples_dir / "train_examples.csv", train_rows)
    write_audit_csv(examples_dir / "val_examples.csv", val_rows)
    write_audit_csv(examples_dir / "test_examples_by_length.csv", test_rows)
