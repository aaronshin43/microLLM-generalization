"""Focused checks for the learned-log mechanism follow-up."""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_training_length_init import initial_state, softplus_inverse
from run_training_length_trace import UpdateTrace
from stage3_simplified_attention import Stage3Config, project_dir, set_reproducibility, train_model


class MechanismFollowupTest(unittest.TestCase):
    def test_initial_coefficient_pairing(self):
        base = Stage3Config(seed=137, alpha_mode="learned_log", train_lengths=(10,))
        states = [initial_state(replace(base, train_lengths=lengths, alpha_log_scale_init=u))
                  for u in (-5, softplus_inverse(0.1), softplus_inverse(1.0))
                  for lengths in ((10,), (100,))]
        self.assertEqual(len({state["noncoefficient_state_sha256"] for state in states}), 1)
        self.assertEqual(len({state["initial_state_sha256"] for state in states}), 3)
        self.assertTrue(math.isclose(states[2]["c_initial"], 0.1, rel_tol=1e-6))
        self.assertTrue(math.isclose(states[4]["c_initial"], 1.0, rel_tol=1e-6))

    def test_passive_update_trace_preserves_weights(self):
        output = project_dir() / "runs/_test_training_length_mechanism"
        plain_dir, traced_dir = output / "plain", output / "traced"
        plain_dir.mkdir(parents=True, exist_ok=True)
        traced_dir.mkdir(parents=True, exist_ok=True)
        config = Stage3Config(seed=138, device="cpu", alpha_mode="learned_log",
                              train_length=10, train_lengths=(10,), train_examples=64,
                              val_examples=32, batch_size=16, eval_batch_size=16,
                              max_train_steps=8)
        set_reproducibility(config.seed)
        plain, plain_steps = train_model(config, device=torch.device("cpu"), output_dir=plain_dir)
        trace = UpdateTrace(set(range(1, 9)), traced_dir)
        set_reproducibility(config.seed)
        traced, traced_steps = train_model(config, device=torch.device("cpu"),
                                           output_dir=traced_dir, update_observer=trace)
        self.assertEqual((plain_steps, traced_steps), (8, 8))
        self.assertTrue(all(torch.equal(value, traced.state_dict()[name])
                            for name, value in plain.state_dict().items()))
        self.assertEqual((plain_dir / "train_history.csv").read_bytes(),
                         (traced_dir / "train_history.csv").read_bytes())
        self.assertEqual(len(trace.rows), 8)
        self.assertLess(max(trace.max_residual.values()), 1e-10)


if __name__ == "__main__":
    unittest.main()
