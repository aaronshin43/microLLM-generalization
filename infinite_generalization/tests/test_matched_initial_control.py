"""Check matched behavior, factorization, and passive tracing."""

from __future__ import annotations

import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_matched_initial_control import (
    MatchedUpdateTrace, apply_matched_initialization,
    initial_metrics, make_model, rng_digest,
)
from run_training_length_init import softplus_inverse
from stage3_simplified_attention import Stage3Config, project_dir, set_reproducibility, train_model


class MatchedInitialControlTest(unittest.TestCase):
    def config(self, seed=5, coefficient=1.0):
        return Stage3Config(seed=seed, device="cpu", alpha_mode="learned_log",
                            alpha_log_scale_init=softplus_inverse(coefficient),
                            train_length=10, train_lengths=(10,), train_examples=64,
                            val_examples=32, batch_size=16, eval_batch_size=16,
                            max_train_steps=8)

    def test_negative_gap_matches_reference_only(self):
        model = make_model(self.config(), torch.device("cpu"))
        before_rng = rng_digest()
        result = apply_matched_initialization(model, 100)
        self.assertEqual(before_rng, rng_digest())
        self.assertLess(result["baseline"]["delta"], 0)
        self.assertLess(result["matched"]["delta"], 0)
        self.assertAlmostEqual(result["baseline"]["effective_margin_100"],
                               result["matched"]["effective_margin_100"], places=6)
        self.assertNotAlmostEqual(result["baseline"]["effective_margin_10"],
                                  result["matched"]["effective_margin_10"], places=3)
        self.assertTrue(result["head_bitwise_equal"])

    def test_factorizations_match_all_lengths_with_different_norms(self):
        symmetric = make_model(self.config(seed=3), torch.device("cpu"))
        apply_matched_initialization(symmetric, 10)
        for factor in ("query_only", "key_only"):
            model = make_model(self.config(seed=3), torch.device("cpu"))
            result = apply_matched_initialization(model, 10, factor,
                                                 comparison_state=symmetric.state_dict())
            self.assertIsNotNone(result["factorization_function_residuals"])
            self.assertNotAlmostEqual(initial_metrics(model)["query_norm"],
                                      initial_metrics(symmetric)["query_norm"], places=4)

    def test_zero_gap_and_baseline_bypass(self):
        config = replace(self.config(), alpha_log_scale_init=-5.0)
        baseline = make_model(config, torch.device("cpu"))
        old = copy.deepcopy(baseline.state_dict())
        result = apply_matched_initialization(baseline, 100)
        self.assertEqual(result["ratio"], 1.0)
        self.assertTrue(all(torch.equal(value, baseline.state_dict()[name]) for name, value in old.items()))
        model = make_model(self.config(), torch.device("cpu"))
        with torch.no_grad():
            model.key_projection.weight[:, 0].copy_(model.key_projection.weight[:, 1])
        apply_matched_initialization(model, 100)
        self.assertEqual(initial_metrics(model)["delta"], 0.0)

    def test_trace_preserves_matched_training_path(self):
        output = project_dir() / "runs/_test_matched_initial_control"
        plain_dir, traced_dir = output / "plain", output / "traced"
        plain_dir.mkdir(parents=True, exist_ok=True)
        traced_dir.mkdir(parents=True, exist_ok=True)
        config = self.config(seed=3)
        def initialize(model):
            apply_matched_initialization(model, 10)
        set_reproducibility(config.seed)
        plain, a = train_model(config, device=torch.device("cpu"), output_dir=plain_dir,
                               initial_state_transform=initialize)
        plain_rng = rng_digest()
        trace = MatchedUpdateTrace(traced_dir, set(range(1, 9)))
        set_reproducibility(config.seed)
        traced, b = train_model(config, device=torch.device("cpu"), output_dir=traced_dir,
                                initial_state_transform=initialize, update_observer=trace)
        self.assertEqual((a, b), (8, 8))
        self.assertEqual(plain_rng, rng_digest())
        self.assertTrue(all(torch.equal(value, traced.state_dict()[name])
                            for name, value in plain.state_dict().items()))
        self.assertEqual((plain_dir / "train_history.csv").read_bytes(),
                         (traced_dir / "train_history.csv").read_bytes())
        self.assertEqual(len(trace.rows), 8)
        self.assertLess(max(trace.max_residual.values()), 1e-10)
        self.assertIn("query_full_update_norm", trace.rows[-1])


if __name__ == "__main__":
    unittest.main()
