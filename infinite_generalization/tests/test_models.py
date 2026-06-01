"""Tests for model output shapes and attention export."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models import (  # noqa: E402
    LengthAwareTransformerTokenPresenceClassifier,
    TransformerTokenPresenceClassifier,
    format_trainable_parameters,
)


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

    def test_length_aware_global_temperature_returns_attention_probabilities(self) -> None:
        model = LengthAwareTransformerTokenPresenceClassifier(
            vocab_size=16,
            d_model=64,
            num_heads=1,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.0,
            attention_variant="global_log_temperature",
            log_scale_init=-5.0,
            target_detector="linear",
        )
        tokens = torch.randint(0, 16, (2, 11))

        logits, pooled, attention_layers = model.forward_with_attention(tokens)

        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(pooled.shape), (2, 64))
        self.assertEqual(tuple(attention_layers[0].shape), (2, 1, 11, 11))
        attention_sums = attention_layers[0].sum(dim=-1)
        self.assertTrue(torch.allclose(attention_sums, torch.ones_like(attention_sums)))

    def test_length_scale_parameter_receives_gradient_at_length_ten(self) -> None:
        model = LengthAwareTransformerTokenPresenceClassifier(
            vocab_size=16,
            d_model=64,
            num_heads=1,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.0,
            attention_variant="global_log_temperature",
            log_scale_init=-5.0,
            target_detector="linear",
        )
        tokens = torch.randint(0, 16, (2, 10))

        loss = model(tokens).sum()
        loss.backward()

        gradient = model.layers[0].self_attn.log_scale.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().item()), 0.0)

    def test_target_key_log_bias_runs_without_oracle_target_mask(self) -> None:
        model = LengthAwareTransformerTokenPresenceClassifier(
            vocab_size=16,
            d_model=64,
            num_heads=1,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.0,
            attention_variant="target_key_log_bias",
            log_scale_init=-5.0,
            target_detector="linear",
        )
        tokens = torch.randint(0, 16, (2, 9))

        logits, _, attention_layers = model.forward_with_attention(tokens)
        rows = model.length_scale_rows((10, 100))

        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(attention_layers[0].shape), (2, 1, 9, 9))
        self.assertIn("target_detector.weight", format_trainable_parameters(model))
        self.assertEqual(rows[0]["attention_variant"], "target_key_log_bias")
        self.assertIn("beta_length_scale", rows[0])


if __name__ == "__main__":
    unittest.main()
