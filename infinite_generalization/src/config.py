"""Shared configuration for the token-presence experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskConfig:
    """Specification for the first token-presence detection task."""

    vocab_size: int = 16
    target_token: int = 1
    train_length: int = 10
    eval_lengths: tuple[int, ...] = (10, 20, 50, 100, 200, 500, 1000)
    positive_fraction: float = 0.5

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the task config."""

        data = asdict(self)
        data["eval_lengths"] = list(self.eval_lengths)
        return data


@dataclass(frozen=True)
class Stage0Config:
    """Training configuration for the non-transformer max-pooling baseline."""

    seed: int = 1234
    train_examples: int = 50_000
    val_examples: int = 10_000
    test_examples: int = 10_000
    batch_size: int = 512
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    embedding_dim: int = 32
    hidden_dim: int = 64
    output_dir: str = "runs/stage0_maxpool_baseline"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the training config."""

        return asdict(self)
