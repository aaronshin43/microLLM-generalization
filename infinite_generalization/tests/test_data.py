"""Tests for the token-presence synthetic data generator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import TaskConfig  # noqa: E402
from data import _sample_non_target_tokens, make_balanced_token_presence_dataset  # noqa: E402


class TokenPresenceDataTest(unittest.TestCase):
    """Validate the dataset invariants required by TASK.md."""

    def test_non_target_sampler_excludes_target_token(self) -> None:
        task = TaskConfig()
        generator = torch.Generator().manual_seed(123)

        tokens = _sample_non_target_tokens(
            (10_000,),
            vocab_size=task.vocab_size,
            target_token=task.target_token,
            generator=generator,
        )

        self.assertFalse(tokens.eq(task.target_token).any())
        self.assertGreaterEqual(int(tokens.min().item()), 0)
        self.assertLess(int(tokens.max().item()), task.vocab_size)

    def test_balanced_dataset_has_zero_target_negatives_and_exactly_one_positives(self) -> None:
        task = TaskConfig()
        generator = torch.Generator().manual_seed(456)

        inputs, labels = make_balanced_token_presence_dataset(
            num_examples=1_000,
            length=task.train_length,
            task=task,
            generator=generator,
        )

        target_counts = inputs.eq(task.target_token).sum(dim=1)
        negative_counts = target_counts[labels.eq(0)]
        positive_counts = target_counts[labels.eq(1)]

        self.assertEqual(int(labels.eq(0).sum().item()), 500)
        self.assertEqual(int(labels.eq(1).sum().item()), 500)
        self.assertTrue(negative_counts.eq(0).all())
        self.assertTrue(positive_counts.eq(1).all())

    def test_invalid_generation_arguments_raise_errors(self) -> None:
        task = TaskConfig()
        generator = torch.Generator().manual_seed(789)

        with self.assertRaises(ValueError):
            make_balanced_token_presence_dataset(
                num_examples=1,
                length=task.train_length,
                task=task,
                generator=generator,
            )

        with self.assertRaises(ValueError):
            make_balanced_token_presence_dataset(
                num_examples=2,
                length=0,
                task=task,
                generator=generator,
            )


if __name__ == "__main__":
    unittest.main()

