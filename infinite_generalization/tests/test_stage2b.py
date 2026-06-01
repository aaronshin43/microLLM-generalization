"""Tests for Stage 2B length-aware attention training components."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import Stage2BConfig, TaskConfig  # noqa: E402
from stage2b_length_aware_attention import (  # noqa: E402
    evaluate_train_lengths,
    make_length_loaders,
    make_model,
    save_stage2b_analysis_tables,
)
from train import run_loader_sequence  # noqa: E402


class Stage2BTrainingComponentTest(unittest.TestCase):
    """Validate Stage 2B runner contracts without launching full experiments."""

    def test_fixed_length_training_components_run_smoke_step(self) -> None:
        task = TaskConfig(eval_lengths=(10, 20))
        config = Stage2BConfig(
            train_lengths=(10,),
            train_examples_per_length=64,
            val_examples_per_length=32,
            batch_size=32,
            eval_batch_size=32,
            attention_variant="global_log_temperature",
        )
        device = torch.device("cpu")
        model = make_model(task=task, config=config, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        train_loaders = make_length_loaders(
            lengths=config.train_lengths,
            examples_per_length=config.train_examples_per_length,
            task=task,
            config=config,
            seed_offset=0,
            shuffle=True,
        )

        loss, metrics = run_loader_sequence(
            model,
            train_loaders,
            criterion=nn.BCEWithLogitsLoss(),
            device=device,
            optimizer=optimizer,
        )
        train_length_metrics = evaluate_train_lengths(
            model,
            task=task,
            config=config,
            device=device,
        )

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertIn("overall_accuracy", metrics)
        self.assertEqual(len(train_length_metrics), 1)
        self.assertEqual(train_length_metrics[0]["length"], 10)

    def test_multilength_training_components_run_smoke_step(self) -> None:
        task = TaskConfig(eval_lengths=(10, 20))
        config = Stage2BConfig(
            train_lengths=(10, 20),
            train_examples_per_length=64,
            val_examples_per_length=32,
            batch_size=32,
            eval_batch_size=32,
            attention_variant="target_key_log_bias",
        )
        device = torch.device("cpu")
        model = make_model(task=task, config=config, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        train_loaders = make_length_loaders(
            lengths=config.train_lengths,
            examples_per_length=config.train_examples_per_length,
            task=task,
            config=config,
            seed_offset=0,
            shuffle=True,
        )

        loss, metrics = run_loader_sequence(
            model,
            train_loaders,
            criterion=nn.BCEWithLogitsLoss(),
            device=device,
            optimizer=optimizer,
        )
        train_length_metrics = evaluate_train_lengths(
            model,
            task=task,
            config=config,
            device=device,
        )

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertIn("positive_accuracy", metrics)
        self.assertEqual([row["length"] for row in train_length_metrics], [10, 20])

    def test_stage2b_forward_has_no_nans_on_long_input(self) -> None:
        task = TaskConfig()
        config = Stage2BConfig(attention_variant="target_key_log_bias")
        model = make_model(task=task, config=config, device=torch.device("cpu"))
        tokens = torch.randint(0, task.vocab_size, (1, 512))

        logits, pooled, attention_layers, attention_details = model.forward_with_attention_details(
            tokens
        )

        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(pooled).all())
        self.assertTrue(torch.isfinite(attention_layers[0]).all())
        self.assertTrue(torch.isfinite(attention_details[0]["base_scores"]).all())
        self.assertTrue(torch.isfinite(attention_details[0]["corrected_scores"]).all())

    def test_stage2b_analysis_tables_are_written(self) -> None:
        task = TaskConfig(eval_lengths=(10, 20))
        config = Stage2BConfig(attention_variant="target_key_log_bias")
        model = make_model(task=task, config=config, device=torch.device("cpu"))

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_stage2b_analysis_tables(model, task=task, output_dir=output_dir)

            self.assertTrue((output_dir / "length_scales.csv").exists())
            self.assertTrue((output_dir / "target_detector_vocab.csv").exists())

    def test_length_scale_is_positive_and_changes_with_length(self) -> None:
        task = TaskConfig()
        config = Stage2BConfig(attention_variant="global_log_temperature")
        model = make_model(task=task, config=config, device=torch.device("cpu"))

        rows = model.length_scale_rows((10, 100))

        self.assertGreater(rows[0]["learned_positive_scale"], 0.0)
        self.assertGreater(rows[0]["length_correction"], 0.0)
        self.assertGreater(rows[1]["length_correction"], rows[0]["length_correction"])


if __name__ == "__main__":
    unittest.main()
