"""Tests for numerical analysis helpers."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from analysis import manual_layer0_forward, write_csv  # noqa: E402
from models import TransformerTokenPresenceClassifier  # noqa: E402


class NumericalAnalysisTest(unittest.TestCase):
    """Validate core contracts used by the Stage 1 analysis pipeline."""

    def test_write_csv_accepts_rows_with_different_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"

            write_csv(path, [{"a": 1}, {"a": 2, "b": 3}])

            with path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(rows[0], {"a": "1", "b": ""})
        self.assertEqual(rows[1], {"a": "2", "b": "3"})

    def test_manual_layer0_forward_returns_expected_shapes(self) -> None:
        model = TransformerTokenPresenceClassifier(
            vocab_size=16,
            d_model=64,
            num_heads=1,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.0,
        )
        tokens = torch.randint(0, 16, (7,))

        tensors = manual_layer0_forward(model, tokens)

        self.assertEqual(tuple(tensors["scores"].shape), (1, 7, 7))
        self.assertEqual(tuple(tensors["attention"].shape), (1, 7, 7))
        self.assertEqual(tuple(tensors["encoded"].shape), (7, 64))
        self.assertEqual(tuple(tensors["pooled"].shape), (64,))
        self.assertEqual(tuple(tensors["argmax_positions"].shape), (64,))
        self.assertEqual(tuple(tensors["logit"].shape), ())


if __name__ == "__main__":
    unittest.main()
