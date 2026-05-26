"""Train and evaluate the Stage 0 max-pooling baseline."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import Stage0Config, TaskConfig
from data import make_balanced_token_presence_dataset
from metrics import binary_accuracy_slices
from models import MaxPoolTokenPresenceClassifier, count_parameters


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Stage 0 experiment."""

    defaults = Stage0Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--train-examples", type=int, default=defaults.train_examples)
    parser.add_argument("--val-examples", type=int, default=defaults.val_examples)
    parser.add_argument("--test-examples", type=int, default=defaults.test_examples)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--embedding-dim", type=int, default=defaults.embedding_dim)
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--output-dir", type=str, default=defaults.output_dir)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a tiny configuration that verifies the pipeline quickly.",
    )
    parser.add_argument(
        "--save-examples",
        action="store_true",
        help="Save audit CSV files with sample sequences and model outputs.",
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=4,
        help="Number of positive and negative examples to save per split or length.",
    )
    parser.add_argument(
        "--preview-tokens",
        type=int,
        default=12,
        help="Number of tokens to keep at each edge for long sequence previews.",
    )
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    """Set deterministic seeds for Python and PyTorch."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_config(args: argparse.Namespace) -> Stage0Config:
    """Build the training config, with small overrides for smoke tests."""

    config = Stage0Config(
        seed=args.seed,
        train_examples=args.train_examples,
        val_examples=args.val_examples,
        test_examples=args.test_examples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dir=args.output_dir,
    )

    if args.smoke_test:
        config = replace(
            config,
            train_examples=2_048,
            val_examples=512,
            test_examples=512,
            batch_size=256,
            epochs=3,
            output_dir="runs/stage0_smoke_test",
        )
    return config


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
    config: Stage0Config,
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
        generator = torch.Generator().manual_seed(config.seed + 20_000 + index)
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


def main() -> None:
    """Run the full Stage 0 training and evaluation pipeline."""

    args = parse_args()
    config = make_config(args)
    task = TaskConfig()
    set_reproducibility(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_generator = torch.Generator().manual_seed(config.seed)
    loader_generator = torch.Generator().manual_seed(config.seed + 1)

    train_inputs, train_labels = make_balanced_token_presence_dataset(
        num_examples=config.train_examples,
        length=task.train_length,
        task=task,
        generator=data_generator,
    )
    val_inputs, val_labels = make_balanced_token_presence_dataset(
        num_examples=config.val_examples,
        length=task.train_length,
        task=task,
        generator=data_generator,
    )

    train_loader = make_loader(
        train_inputs,
        train_labels,
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = make_loader(
        val_inputs,
        val_labels,
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = MaxPoolTokenPresenceClassifier(
        vocab_size=task.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    run_metadata = {
        "task": task.to_dict(),
        "stage0": config.to_dict(),
        "device": str(device),
        "parameter_count": count_parameters(model),
    }
    write_json(output_dir / "config.json", run_metadata)

    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion=criterion,
            device=device,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_overall_accuracy": train_metrics["overall_accuracy"],
            "train_positive_accuracy": train_metrics["positive_accuracy"],
            "train_negative_accuracy": train_metrics["negative_accuracy"],
            "val_loss": val_loss,
            "val_overall_accuracy": val_metrics["overall_accuracy"],
            "val_positive_accuracy": val_metrics["positive_accuracy"],
            "val_negative_accuracy": val_metrics["negative_accuracy"],
        }
        history.append(row)
        print(
            "epoch={epoch:02d} train_loss={train_loss:.4f} "
            "train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}".format(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_metrics["overall_accuracy"],
                val_loss=val_loss,
                val_acc=val_metrics["overall_accuracy"],
            )
        )

    metrics_by_length = evaluate_by_length(model, task=task, config=config, device=device)
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "metrics_by_length.json", metrics_by_length)
    write_metrics_csv(output_dir / "metrics_by_length.csv", metrics_by_length)
    torch.save(model.state_dict(), output_dir / "model.pt")

    if args.save_examples:
        save_audit_examples(
            model,
            train_inputs=train_inputs,
            train_labels=train_labels,
            val_inputs=val_inputs,
            val_labels=val_labels,
            task=task,
            config=config,
            device=device,
            output_dir=output_dir,
            examples_per_class=args.examples_per_class,
            preview_tokens=args.preview_tokens,
        )

    print("\nLength sweep:")
    for row in metrics_by_length:
        print(
            "length={length:4d} overall={overall_accuracy:.4f} "
            "positive={positive_accuracy:.4f} negative={negative_accuracy:.4f}".format(**row)
        )
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
