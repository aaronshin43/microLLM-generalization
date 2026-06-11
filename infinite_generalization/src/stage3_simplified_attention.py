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
    train_examples: int = 2_000
    val_examples: int = 500
    test_examples: int = 50
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
    parser.add_argument("--train-examples", type=int, default=Stage3Config.train_examples)
    parser.add_argument("--val-examples", type=int, default=Stage3Config.val_examples)
    parser.add_argument("--test-examples", type=int, default=Stage3Config.test_examples)
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

    config = Stage3Config(
        seed=args.seed,
        device=args.device,
        output_dir=output_dir,
        alpha_mode=args.alpha_mode,
        train_length=args.train_length,
        train_lengths=train_lengths,
        train_examples=args.train_examples,
        val_examples=args.val_examples,
        test_examples=args.test_examples,
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
        train_examples=64,
        val_examples=32,
        test_examples=40,
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


def make_two_token_dataset(
    *,
    length: int,
    examples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a balanced two-token target-presence dataset.

    Positive examples are [t, u, ..., u]. Negative examples are [u, u, ..., u].
    The token id convention is t=0 and u=1.
    """

    if length < 2:
        raise ValueError("length must be at least 2 for the exactly-one-target setup.")
    if examples < 2:
        raise ValueError("examples must be at least 2.")

    positive_count = examples // 2
    negative_count = examples - positive_count

    positive_inputs = torch.full((positive_count, length), NON_TARGET_TOKEN_ID, dtype=torch.long)
    positive_inputs[:, TARGET_POSITION] = TARGET_TOKEN_ID
    positive_labels = torch.ones(positive_count, dtype=torch.float32)

    negative_inputs = torch.full((negative_count, length), NON_TARGET_TOKEN_ID, dtype=torch.long)
    negative_labels = torch.zeros(negative_count, dtype=torch.float32)

    inputs = torch.cat([positive_inputs, negative_inputs], dim=0)
    labels = torch.cat([positive_labels, negative_labels], dim=0)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(examples, generator=generator)
    return inputs[permutation], labels[permutation]


def make_loader(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic DataLoader."""

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(inputs, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def resolved_train_lengths(config: Stage3Config) -> tuple[int, ...]:
    """Return the active training lengths, preserving backward compatibility."""

    return config.train_lengths if config.train_lengths else (config.train_length,)


def make_multilength_loaders(
    *,
    lengths: tuple[int, ...],
    examples_per_length: int,
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
        inputs, labels = make_two_token_dataset(
            length=length,
            examples=examples_per_length,
            seed=seed + offset,
        )
        loaders.append(
            make_loader(
                inputs,
                labels,
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
    ) -> None:
        super().__init__()
        if d_head < 1:
            raise ValueError("d_head must be positive.")
        if alpha_mode not in {"constant", "log", "learned_log"}:
            raise ValueError(f"Unsupported alpha_mode: {alpha_mode}")

        self.d_head = d_head
        self.alpha_mode = alpha_mode
        self.query_projection = nn.Linear(2, d_head, bias=False)
        self.key_projection = nn.Linear(2, d_head, bias=False)
        self.classifier = nn.Linear(2, 1)
        self.alpha_log_scale_unconstrained = nn.Parameter(
            torch.tensor(float(alpha_log_scale_init))
        )

    def token_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return fixed one-hot embeddings with t=[1,0] and u=[0,1]."""

        return F.one_hot(tokens, num_classes=2).to(dtype=torch.float32)

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
        embeddings = self.token_embeddings(tokens)
        queries = self.query_projection(embeddings)
        keys = self.key_projection(embeddings)

        last_query = queries[:, -1, :]
        raw_scores = torch.einsum("bd,bld->bl", last_query, keys) / math.sqrt(self.d_head)
        alpha = self.alpha_for_length(length, tokens.device)
        corrected_scores = alpha * raw_scores
        attention_weights = torch.softmax(corrected_scores, dim=-1)
        attention_output = torch.einsum("bl,bld->bd", attention_weights, embeddings)
        logits = self.classifier(attention_output).squeeze(-1)

        if not return_details:
            return logits

        details = {
            "raw_scores": raw_scores,
            "corrected_scores": corrected_scores,
            "attention_weights": attention_weights,
            "attention_output": attention_output,
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

    for tokens, labels in loader:
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

    for tokens, labels in batches:
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
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if total_examples == 0:
        return float("nan"), {"accuracy": float("nan"), "positive_accuracy": float("nan"), "negative_accuracy": float("nan")}, update_count

    logits_cpu = torch.cat(all_logits)
    labels_cpu = torch.cat(all_labels)
    return total_loss / total_examples, binary_accuracy(logits_cpu, labels_cpu), update_count


@torch.no_grad()
def evaluate_length(
    model: SimplifiedLastQueryAttentionClassifier,
    *,
    length: int,
    examples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    """Evaluate one length and collect mechanism metrics."""

    inputs, labels = make_two_token_dataset(length=length, examples=examples, seed=seed)
    loader = make_loader(inputs, labels, batch_size=batch_size, shuffle=False, seed=seed)

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
    empirical_attentions: list[torch.Tensor] = []
    empirical_theory_attentions: list[torch.Tensor] = []
    alpha_values: list[torch.Tensor] = []

    for tokens, batch_labels in loader:
        tokens = tokens.to(device)
        batch_labels = batch_labels.to(device)
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
            alpha = details["alpha"][positive_mask]
            target_score = raw_scores[:, TARGET_POSITION]
            non_target_scores = raw_scores[:, 1:]
            non_target_mean = non_target_scores.mean(dim=1)
            delta = target_score - non_target_mean
            empirical_attention = attention_weights[:, TARGET_POSITION]
            theory_attention = target_attention_theory(
                length=length,
                alpha=alpha,
                delta=delta,
            )

            target_scores.append(target_score.cpu())
            non_target_means.append(non_target_mean.cpu())
            deltas.append(delta.cpu())
            non_target_stds.append(non_target_scores.std(dim=1, unbiased=False).cpu())
            empirical_attentions.append(empirical_attention.cpu())
            empirical_theory_attentions.append(theory_attention.cpu())
            alpha_values.append(alpha.cpu())

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
    empirical_attentions_cpu = torch.cat(empirical_attentions)
    empirical_theory_attentions_cpu = torch.cat(empirical_theory_attentions)
    alpha_values_cpu = torch.cat(alpha_values)

    attention_abs_error = (empirical_attentions_cpu - empirical_theory_attentions_cpu).abs()
    classifier_weight = model.classifier.weight.detach().cpu().squeeze(0)

    return {
        "length": length,
        "split": "test",
        "alpha_mode": model.alpha_mode,
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
        "mean_empirical_target_attention": mean_or_nan(empirical_attentions_cpu),
        "mean_theory_target_attention_using_empirical_delta": mean_or_nan(
            empirical_theory_attentions_cpu
        ),
        "mean_theory_target_attention_using_train_delta": float("nan"),
        "mean_attention_absolute_error_empirical_vs_theory": mean_or_nan(attention_abs_error),
        "learned_alpha_coefficient": model.learned_alpha_coefficient(),
        "classifier_weight_target_coord": classifier_weight[0].item(),
        "classifier_weight_non_target_coord": classifier_weight[1].item(),
        "classifier_bias": model.classifier.bias.detach().cpu().item(),
    }


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
            "non_target_token_id": NON_TARGET_TOKEN_ID,
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
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed + 1,
    )
    val_loaders = make_multilength_loaders(
        lengths=train_lengths,
        examples_per_length=config.val_examples,
        batch_size=config.eval_batch_size,
        shuffle=False,
        seed=config.seed + 2,
    )

    model = SimplifiedLastQueryAttentionClassifier(
        d_head=config.d_head,
        alpha_mode=config.alpha_mode,
        alpha_log_scale_init=config.alpha_log_scale_init,
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
            "non_target_token_id": NON_TARGET_TOKEN_ID,
        },
    )

    rows = [
        evaluate_length(
            model,
            length=length,
            examples=config.test_examples,
            batch_size=config.eval_batch_size,
            seed=config.seed + 10_000 + length,
            device=device,
        )
        for length in config.eval_lengths
    ]
    add_train_delta_theory(rows, train_length=train_lengths[0])
    for row in rows:
        row["train_lengths"] = " ".join(str(length) for length in train_lengths)
        row["train_length_count"] = len(train_lengths)
        row["optimizer_updates"] = optimizer_updates
        row["examples_per_train_length"] = config.train_examples
    write_csv(output_dir / "metrics_by_length.csv", rows)
    write_figures(output_dir, rows)

    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
