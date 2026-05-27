"""Train and evaluate the Stage 1 no-position transformer baseline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from audit import save_audit_examples
from attention import save_attention_analysis
from config import Stage1Config, TaskConfig
from data import make_balanced_token_presence_dataset
from models import TransformerTokenPresenceClassifier, count_parameters
from train import (
    evaluate_by_length,
    make_loader,
    run_epoch,
    set_reproducibility,
    write_json,
    write_metrics_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Stage 1 transformer experiment."""

    defaults = Stage1Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--train-examples", type=int, default=defaults.train_examples)
    parser.add_argument("--val-examples", type=int, default=defaults.val_examples)
    parser.add_argument("--test-examples", type=int, default=defaults.test_examples)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=defaults.eval_batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--d-model", type=int, default=defaults.d_model)
    parser.add_argument("--num-heads", type=int, default=defaults.num_heads)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--dim-feedforward", type=int, default=defaults.dim_feedforward)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
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
        "--save-attention",
        action="store_true",
        help="Save attention summary CSV files for selected evaluation examples.",
    )
    parser.add_argument(
        "--save-raw-attention",
        action="store_true",
        help="Save raw attention tensors for selected examples. Can use substantial disk space.",
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=4,
        help="Number of positive and negative examples to save per split or length.",
    )
    parser.add_argument(
        "--attention-examples-per-class",
        type=int,
        default=2,
        help="Number of positive and negative examples per length for attention analysis.",
    )
    parser.add_argument(
        "--preview-tokens",
        type=int,
        default=12,
        help="Number of tokens to keep at each edge for long sequence previews.",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Stage1Config:
    """Build the training config, with small overrides for smoke tests."""

    config = Stage1Config(
        seed=args.seed,
        train_examples=args.train_examples,
        val_examples=args.val_examples,
        test_examples=args.test_examples,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        output_dir=args.output_dir,
    )

    if args.smoke_test:
        config = replace(
            config,
            train_examples=2_048,
            val_examples=512,
            test_examples=512,
            batch_size=128,
            eval_batch_size=64,
            epochs=3,
            output_dir="runs/stage1_smoke_test",
        )
    return config


def main() -> None:
    """Run the full Stage 1 transformer training and evaluation pipeline."""

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

    model = TransformerTokenPresenceClassifier(
        vocab_size=task.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    run_metadata = {
        "task": task.to_dict(),
        "stage1": config.to_dict(),
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

    print("\nLength sweep:")
    for row in metrics_by_length:
        print(
            "length={length:4d} overall={overall_accuracy:.4f} "
            "positive={positive_accuracy:.4f} negative={negative_accuracy:.4f}".format(**row)
        )
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
