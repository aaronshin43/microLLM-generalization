"""Tests for YAML-backed experiment configuration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import (  # noqa: E402
    Stage0Config,
    Stage1Config,
    Stage2AConfig,
    TaskConfig,
    build_config,
    resolve_device,
    split_experiment_config,
)


class ExperimentConfigTest(unittest.TestCase):
    """Validate config merging and device selection behavior."""

    def test_split_experiment_config_accepts_task_and_stage_sections(self) -> None:
        task_values, stage_values = split_experiment_config(
            {
                "task": {"eval_lengths": [10, 20]},
                "stage": {"device": "cpu", "epochs": 2},
            }
        )

        self.assertEqual(task_values, {"eval_lengths": [10, 20]})
        self.assertEqual(stage_values, {"device": "cpu", "epochs": 2})

    def test_build_config_applies_cli_overrides_after_yaml_values(self) -> None:
        config = build_config(
            Stage1Config,
            yaml_values={"device": "cpu", "epochs": 2},
            cli_values={"epochs": 3},
        )

        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.epochs, 3)

    def test_stage0_train_length_lives_in_stage_config(self) -> None:
        config = build_config(
            Stage0Config,
            yaml_values={"train_length": 20},
            cli_values={"train_length": 30},
        )

        self.assertEqual(config.train_length, 30)

    def test_task_config_normalizes_yaml_eval_lengths_to_tuple(self) -> None:
        task = build_config(
            TaskConfig,
            yaml_values={"eval_lengths": [10, 20, 50]},
            cli_values={},
        )

        self.assertEqual(task.eval_lengths, (10, 20, 50))

    def test_stage2a_config_normalizes_yaml_train_lengths_to_tuple(self) -> None:
        config = build_config(
            Stage2AConfig,
            yaml_values={"train_lengths": [10, 20, 50, 100]},
            cli_values={},
        )

        self.assertEqual(config.train_lengths, (10, 20, 50, 100))

    def test_resolve_device_accepts_cpu(self) -> None:
        self.assertEqual(resolve_device("cpu").type, "cpu")


if __name__ == "__main__":
    unittest.main()
