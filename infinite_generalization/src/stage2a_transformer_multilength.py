"""Train Stage 2A: the Stage 1 transformer with multi-length training."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from attention import save_attention_analysis
from audit import save_multilength_audit_examples
from config import (
    Stage2AConfig,
    TaskConfig,
    build_config,
    load_yaml_config,
    resolve_device,
    split_experiment_config,
)
from data import make_balanced_token_presence_dataset
from models import TransformerTokenPresenceClassifier, count_parameters, format_trainable_parameters
from train import (
    evaluate_by_length,
    evaluate_dataset,
    evaluate_diagnostic_slices_by_length,
    make_loader,
    run_loader_sequence,
    set_reproducibility,
    write_diagnostic_slices_csv,
    write_json,
    write_metrics_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Stage 2A experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--eval-lengths", type=int, nargs="+", default=None)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=None)
    parser.add_argument("--train-examples-per-length", type=int, default=None)
    parser.add_argument("--val-examples-per-length", type=int, default=None)
    parser.add_argument("--test-examples", type=int, default=None)
    parser.add_argument("--diagnostic-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a tiny configuration that verifies the pipeline quickly.",
    )
    parser.add_argument(
        "--save-attention",
        action="store_true",
        help="Save attention summary CSV files for selected diagnostic examples.",
    )
    parser.add_argument(
        "--save-examples",
        action="store_true",
        help="Save audit CSV files with sample sequences and model outputs.",
    )
    parser.add_argument(
        "--save-raw-attention",
        action="store_true",
        help="Save raw attention tensors for selected examples. Can use substantial disk space.",
    )
    parser.add_argument(
        "--attention-examples-per-class",
        type=int,
        default=2,
        help="Number of examples per diagnostic slice for attention analysis.",
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
    parser.add_argument(
        "--print-parameters",
        action="store_true",
        help="Print trainable parameter names, shapes, dtypes, devices, and counts.",
    )
    return parser.parse_args()


def make_configs(args: argparse.Namespace) -> tuple[TaskConfig, Stage2AConfig]:
    """Build task and Stage 2A configs from defaults, YAML, and CLI overrides."""

    task_cli_values = {
        "eval_lengths": args.eval_lengths,
    }
    stage_cli_values = {
        "seed": args.seed,
        "device": args.device,
        "train_lengths": args.train_lengths,
        "train_examples_per_length": args.train_examples_per_length,
        "val_examples_per_length": args.val_examples_per_length,
        "test_examples": args.test_examples,
        "diagnostic_examples": args.diagnostic_examples,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "output_dir": args.output_dir,
    }
    task_yaml_values, stage_yaml_values = split_experiment_config(load_yaml_config(args.config))
    task = build_config(
        TaskConfig,
        yaml_values=task_yaml_values,
        cli_values={key: value for key, value in task_cli_values.items() if value is not None},
    )
    config = build_config(
        Stage2AConfig,
        yaml_values=stage_yaml_values,
        cli_values={key: value for key, value in stage_cli_values.items() if value is not None},
    )

    if args.smoke_test:
        config = replace(
            config,
            train_examples_per_length=512,
            val_examples_per_length=128,
            test_examples=512,
            diagnostic_examples=128,
            batch_size=128,
            eval_batch_size=64,
            epochs=2,
            output_dir="runs/stage2a_smoke_test",
        )
    return task, config


def make_length_loaders(
    *,
    lengths: tuple[int, ...],
    examples_per_length: int,
    task: TaskConfig,
    config: Stage2AConfig,
    seed_offset: int,
    shuffle: bool,
) -> list[DataLoader]:
    """Create one balanced DataLoader per training length."""

    loaders: list[DataLoader] = []
    for length_index, length in enumerate(lengths):
        data_generator = torch.Generator().manual_seed(config.seed + seed_offset + length_index)
        loader_generator = torch.Generator().manual_seed(
            config.seed + seed_offset + 1_000 + length_index
        )
        inputs, labels = make_balanced_token_presence_dataset(
            num_examples=examples_per_length,
            length=length,
            task=task,
            generator=data_generator,
        )
        loaders.append(
            make_loader(
                inputs,
                labels,
                batch_size=config.batch_size,
                shuffle=shuffle,
                generator=loader_generator,
            )
        )
    return loaders


@torch.no_grad()
def evaluate_train_lengths(
    model: nn.Module,
    *,
    task: TaskConfig,
    config: Stage2AConfig,
    device: torch.device,
) -> list[dict[str, float | int]]:
    """Evaluate held-out validation data for each Stage 2A training length."""

    criterion = nn.BCEWithLogitsLoss()
    rows: list[dict[str, float | int]] = []

    for length_index, length in enumerate(config.train_lengths):
        generator = torch.Generator().manual_seed(config.seed + 30_000 + length_index)
        inputs, labels = make_balanced_token_presence_dataset(
            num_examples=config.val_examples_per_length,
            length=length,
            task=task,
            generator=generator,
        )
        loader = make_loader(
            inputs,
            labels,
            batch_size=config.eval_batch_size,
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
            }
        )

    return rows


def main() -> None:
    """Run Stage 2A multi-length transformer training and evaluation."""

    args = parse_args()
    task, config = make_configs(args)
    set_reproducibility(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config.device)
    train_loaders = make_length_loaders(
        lengths=config.train_lengths,
        examples_per_length=config.train_examples_per_length,
        task=task,
        config=config,
        seed_offset=0,
        shuffle=True,
    )
    val_loaders = make_length_loaders(
        lengths=config.train_lengths,
        examples_per_length=config.val_examples_per_length,
        task=task,
        config=config,
        seed_offset=20_000,
        shuffle=False,
    )

    model = TransformerTokenPresenceClassifier(
        vocab_size=task.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)
    if args.print_parameters:
        print(format_trainable_parameters(model))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    run_metadata = {
        "task": task.to_dict(),
        "stage2a": config.to_dict(),
        "device": str(device),
        "parameter_count": count_parameters(model),
    }
    write_json(output_dir / "config.json", run_metadata)

    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_metrics = run_loader_sequence(
            model,
            train_loaders,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_metrics = run_loader_sequence(
            model,
            val_loaders,
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

    train_length_metrics = evaluate_train_lengths(model, task=task, config=config, device=device)
    metrics_by_length = evaluate_by_length(model, task=task, config=config, device=device)
    diagnostic_slices = evaluate_diagnostic_slices_by_length(
        model,
        task=task,
        config=config,
        device=device,
    )
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "train_lengths_metrics.json", train_length_metrics)
    write_metrics_csv(output_dir / "train_lengths_metrics.csv", train_length_metrics)
    write_json(output_dir / "metrics_by_length.json", metrics_by_length)
    write_metrics_csv(output_dir / "metrics_by_length.csv", metrics_by_length)
    write_json(output_dir / "diagnostic_slices_by_length.json", diagnostic_slices)
    write_diagnostic_slices_csv(
        output_dir / "diagnostic_slices_by_length.csv",
        diagnostic_slices,
    )
    torch.save(model.state_dict(), output_dir / "model.pt")

    if args.save_examples:
        save_multilength_audit_examples(
            model,
            task=task,
            config=config,
            device=device,
            output_dir=output_dir,
            examples_per_class=args.examples_per_class,
            preview_tokens=args.preview_tokens,
        )

    if args.save_attention:
        save_attention_analysis(
            model,
            task=task,
            config=config,
            device=device,
            output_dir=output_dir,
            examples_per_class=args.attention_examples_per_class,
            save_raw=args.save_raw_attention,
        )

    print("\nTrain-length validation:")
    for row in train_length_metrics:
        print(
            "length={length:4d} overall={overall_accuracy:.4f} "
            "positive={positive_accuracy:.4f} negative={negative_accuracy:.4f}".format(**row)
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
