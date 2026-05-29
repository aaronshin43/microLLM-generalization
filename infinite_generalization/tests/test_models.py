"""Tests for model output shapes and attention export."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models import TransformerTokenPresenceClassifier, format_trainable_parameters  # noqa: E402


class TransformerModelTest(unittest.TestCase):
    """Validate Stage 1 transformer output contracts."""

    def test_forward_with_attention_returns_expected_shapes(self) -> None:
        model = TransformerTokenPresenceClassifier(
            vocab_size=16,
            d_model=64,
            num_heads=1,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.0,
        )
        tokens = torch.randint(0, 16, (3, 10))

        logits, pooled, attention_layers = model.forward_with_attention(tokens)

        self.assertEqual(tuple(logits.shape), (3,))
        self.assertEqual(tuple(pooled.shape), (3, 64))
        self.assertEqual(len(attention_layers), 1)
        self.assertEqual(tuple(attention_layers[0].shape), (3, 1, 10, 10))

    def test_format_trainable_parameters_includes_core_descriptions(self) -> None:
        model = TransformerTokenPresenceClassifier(
            vocab_size=16,
            d_model=64,
            num_heads=1,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.0,
        )

        description = format_trainable_parameters(model)

        self.assertIn("embedding.weight", description)
        self.assertIn("layers.0.self_attn.in_proj_weight", description)
        self.assertIn("classifier.weight", description)
        self.assertIn("(16, 64)", description)
        self.assertIn("Total trainable parameters: 34561", description)


if __name__ == "__main__":
    unittest.main()
