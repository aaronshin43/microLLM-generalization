"""Tests for shared training-loop utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from train import run_loader_sequence  # noqa: E402


class CountingSequenceModel(nn.Module):
    """Tiny model that records every input sequence length it sees."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.seen_lengths: list[int] = []
        self.seen_examples = 0

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return one trainable logit per input row."""

        self.seen_lengths.append(tokens.shape[1])
        self.seen_examples += tokens.shape[0]
        return self.bias.expand(tokens.shape[0])


class RunLoaderSequenceTest(unittest.TestCase):
    """Validate Stage 2A multi-loader epoch behavior."""

    def test_run_loader_sequence_consumes_all_batches_when_loader_lengths_differ(self) -> None:
        short_loader = DataLoader(
            TensorDataset(torch.zeros((4, 3), dtype=torch.long), torch.zeros(4)),
            batch_size=2,
            shuffle=False,
        )
        long_loader = DataLoader(
            TensorDataset(torch.zeros((10, 7), dtype=torch.long), torch.ones(10)),
            batch_size=2,
            shuffle=False,
        )
        model = CountingSequenceModel()

        loss, metrics = run_loader_sequence(
            model,
            [short_loader, long_loader],
            criterion=nn.BCEWithLogitsLoss(),
            device=torch.device("cpu"),
        )

        self.assertAlmostEqual(loss, 0.693147, places=5)
        self.assertEqual(model.seen_examples, 14)
        self.assertEqual(model.seen_lengths.count(3), 2)
        self.assertEqual(model.seen_lengths.count(7), 5)
        self.assertAlmostEqual(metrics["overall_accuracy"], 10 / 14, places=6)


if __name__ == "__main__":
    unittest.main()
