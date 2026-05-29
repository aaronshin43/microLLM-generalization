"""Shared configuration for the token-presence experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
import yaml


@dataclass(frozen=True)
class TaskConfig:
    """Specification for the first token-presence detection task."""

    vocab_size: int = 16
    target_token: int = 1
    eval_lengths: tuple[int, ...] = (10, 20, 50, 100, 200, 500, 700, 800, 850, 900, 950, 1000, 1100)
    positive_fraction: float = 0.5

    def __post_init__(self) -> None:
        """Normalize sequence-like config values loaded from YAML."""

        object.__setattr__(self, "eval_lengths", tuple(self.eval_lengths))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the task config."""

        data = asdict(self)
        data["eval_lengths"] = list(self.eval_lengths)
        return data


@dataclass(frozen=True)
class Stage0Config:
    """Training configuration for the non-transformer max-pooling baseline."""

    seed: int = 1234
    device: str = "auto"
    train_length: int = 10
    train_examples: int = 50_000
    val_examples: int = 10_000
    test_examples: int = 10_000
    diagnostic_examples: int = 2_000
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


@dataclass(frozen=True)
class Stage1Config:
    """Training configuration for the minimal transformer baseline."""

    seed: int = 1234
    device: str = "auto"
    train_length: int = 10
    train_examples: int = 50_000
    val_examples: int = 10_000
    test_examples: int = 10_000
    diagnostic_examples: int = 2_000
    batch_size: int = 512
    eval_batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    d_model: int = 64
    num_heads: int = 1
    num_layers: int = 1
    dim_feedforward: int = 128
    dropout: float = 0.0
    output_dir: str = "runs/stage1_transformer_maxpool"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the training config."""

        return asdict(self)


@dataclass(frozen=True)
class Stage2AConfig:
    """Training configuration for the multi-length transformer intervention."""

    seed: int = 1234
    device: str = "auto"
    train_lengths: tuple[int, ...] = (10, 20, 50, 100)
    train_examples_per_length: int = 12_500
    val_examples_per_length: int = 2_500
    test_examples: int = 10_000
    diagnostic_examples: int = 2_000
    batch_size: int = 256
    eval_batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    d_model: int = 64
    num_heads: int = 1
    num_layers: int = 1
    dim_feedforward: int = 128
    dropout: float = 0.0
    output_dir: str = "runs/stage2a_transformer_multilength"

    def __post_init__(self) -> None:
        """Normalize sequence-like config values loaded from YAML."""

        object.__setattr__(self, "train_lengths", tuple(self.train_lengths))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the training config."""

        data = asdict(self)
        data["train_lengths"] = list(self.train_lengths)
        return data


ConfigT = TypeVar("ConfigT", TaskConfig, Stage0Config, Stage1Config, Stage2AConfig)


def load_yaml_config(path: str | None) -> dict[str, Any]:
    """Load YAML config values from disk."""

    if path is None:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping: {config_path}")
    return data


def build_config(
    config_class: type[ConfigT],
    *,
    yaml_values: dict[str, Any],
    cli_values: dict[str, Any],
) -> ConfigT:
    """Build a dataclass config from defaults, YAML values, and explicit CLI overrides."""

    valid_fields = set(config_class.__dataclass_fields__)
    unknown_yaml_fields = set(yaml_values) - valid_fields
    unknown_cli_fields = set(cli_values) - valid_fields

    if unknown_yaml_fields:
        raise ValueError(f"unknown config fields in YAML: {sorted(unknown_yaml_fields)}")
    if unknown_cli_fields:
        raise ValueError(f"unknown config fields in CLI overrides: {sorted(unknown_cli_fields)}")

    merged = {**yaml_values, **cli_values}
    return config_class(**merged)


def split_experiment_config(
    raw_values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a YAML experiment config into task and stage sections."""

    if not raw_values:
        return {}, {}

    if "task" not in raw_values and "stage" not in raw_values:
        # Backward-compatible path for older stage-only YAML files.
        return {}, raw_values

    unknown_sections = set(raw_values) - {"task", "stage"}
    if unknown_sections:
        raise ValueError(f"unknown top-level config sections: {sorted(unknown_sections)}")

    task_values = raw_values.get("task", {})
    stage_values = raw_values.get("stage", {})
    if not isinstance(task_values, dict):
        raise ValueError("config section 'task' must contain a mapping")
    if not isinstance(stage_values, dict):
        raise ValueError("config section 'stage' must contain a mapping")
    return task_values, stage_values


def resolve_device(device_name: str) -> torch.device:
    """Resolve a user-facing device option into a torch.device."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but CUDA is not available.")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(device_name)
