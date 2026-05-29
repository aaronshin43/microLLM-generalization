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
from data import (  # noqa: E402
    _sample_non_target_tokens,
    make_balanced_token_presence_dataset,
    make_negative_dataset,
    make_positive_dataset,
)


class TokenPresenceDataTest(unittest.TestCase):
    """Validate the dataset invariants for token-presence detection."""

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
            length=10,
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
                length=10,
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

    def test_positive_diagnostic_dataset_controls_target_count_and_region(self) -> None:
        task = TaskConfig()
        generator = torch.Generator().manual_seed(321)

        inputs, labels = make_positive_dataset(
            num_examples=100,
            length=30,
            task=task,
            generator=generator,
            target_count=3,
            target_region="begin",
        )

        target_positions = torch.nonzero(inputs.eq(task.target_token), as_tuple=False)
        self.assertTrue(labels.eq(1).all())
        self.assertTrue(inputs.eq(task.target_token).sum(dim=1).eq(3).all())
        self.assertTrue(target_positions[:, 1].lt(10).all())

    def test_negative_diagnostic_dataset_contains_no_targets(self) -> None:
        task = TaskConfig()
        generator = torch.Generator().manual_seed(654)

        inputs, labels = make_negative_dataset(
            num_examples=100,
            length=30,
            task=task,
            generator=generator,
        )

        self.assertTrue(labels.eq(0).all())
        self.assertFalse(inputs.eq(task.target_token).any())


if __name__ == "__main__":
    unittest.main()
