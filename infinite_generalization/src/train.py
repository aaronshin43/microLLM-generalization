"""Shared training, evaluation, and metric-output utilities."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import Stage0Config, TaskConfig
from data import make_balanced_token_presence_dataset
from metrics import binary_accuracy_slices


def set_reproducibility(seed: int) -> None:
    """Set deterministic seeds for Python and PyTorch."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Create a DataLoader, using an explicit generator only when shuffling."""

    return DataLoader(
        TensorDataset(inputs, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    """Run one train or evaluation epoch."""

    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(inputs)
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
    return total_loss / total_examples, binary_accuracy_slices(logits_cpu, labels_cpu)


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    task: TaskConfig,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Evaluate a dataset and compute target-count-aware accuracy slices."""

    model.eval()

    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_target_counts: list[torch.Tensor] = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)
        loss = criterion(logits, labels)

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_target_counts.append(inputs.eq(task.target_token).sum(dim=1).cpu())

    logits_cpu = torch.cat(all_logits)
    labels_cpu = torch.cat(all_labels)
    target_counts_cpu = torch.cat(all_target_counts)

    metrics = binary_accuracy_slices(logits_cpu, labels_cpu)
    predictions = (logits_cpu >= 0).float()
    correct = predictions.eq(labels_cpu)
    exactly_one_positive_mask = labels_cpu.eq(1) & target_counts_cpu.eq(1)
    metrics["exactly_one_positive_accuracy"] = (
        correct[exactly_one_positive_mask].float().mean().item()
        if exactly_one_positive_mask.any()
        else float("nan")
    )

    return total_loss / total_examples, metrics


@torch.no_grad()
def evaluate_by_length(
    model: nn.Module,
    *,
    task: TaskConfig,
    config: Stage0Config,
    device: torch.device,
) -> list[dict[str, float | int]]:
    """Evaluate the trained model on every required sequence length."""

    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    rows: list[dict[str, float | int]] = []

    for index, length in enumerate(task.eval_lengths):
        generator = torch.Generator().manual_seed(config.seed + 10_000 + index)
        inputs, labels = make_balanced_token_presence_dataset(
            num_examples=config.test_examples,
            length=length,
            task=task,
            generator=generator,
        )
        loader = make_loader(
            inputs,
            labels,
            batch_size=config.batch_size,
            shuffle=False,
        )
        loss, metrics = evaluate_dataset(
            model,
            loader,
            criterion=criterion,
            task=task,
            device=device,
        )
        rows.append(
            {
                "length": length,
                "loss": loss,
                "overall_accuracy": metrics["overall_accuracy"],
                "positive_accuracy": metrics["positive_accuracy"],
                "negative_accuracy": metrics["negative_accuracy"],
                "exactly_one_positive_accuracy": metrics["exactly_one_positive_accuracy"],
            }
        )

    return rows


def write_json(path: Path, payload: object) -> None:
    """Write a JSON file with stable formatting."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_metrics_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Write length-sweep metrics to CSV."""

    fieldnames = [
        "length",
        "loss",
        "overall_accuracy",
        "positive_accuracy",
        "negative_accuracy",
        "exactly_one_positive_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

