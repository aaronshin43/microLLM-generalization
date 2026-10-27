"""Validate history reconstruction and censored or unfitted loss comparisons."""

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analyze_training_length_sweep import INTERPOLATED_FIELDS, first_loss_crossing, reconstruct_second_moment


class SweepAnalysisTest(unittest.TestCase):
    def test_reconstruction_matches_actual_float32_adam_state(self):
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.AdamW([parameter], lr=.003, weight_decay=0, betas=(.8, .997))
        rows = []
        for step in range(1, 6401):
            parameter.grad = torch.tensor(math.sin(step / 29) * .03 + .01)
            optimizer.step()
            rows.append({"optimizer_updates": step, "beta2": .997,
                "u_gradient": float(parameter.grad), "v_after": float(optimizer.state[parameter]["exp_avg_sq"])})
        for endpoint in (1600, 6400):
            parts, check = reconstruct_second_moment(rows, endpoint)
            self.assertTrue(check["passed"])
            self.assertAlmostEqual(sum(p["fraction_recorded_second_moment"] for p in parts), 1, places=4)
        with self.assertRaises(AssertionError):
            reconstruct_second_moment(rows[1:], 1600)

    def test_first_crossing_retains_bracket_and_fitting_eligibility(self):
        rows = []
        for step, loss, r in ((0, .03, .2), (200, .005, .8), (400, .02, 1.2), (600, .005, 1.4)):
            row = {key: 0.0 for key in INTERPOLATED_FIELDS}
            row.update(optimizer_updates=step, common100_bce=loss, c_delta=r,
                own_positive=2, own_negative=-2)
            rows.append(row)
        crossing = first_loss_crossing(rows, .01)
        self.assertEqual((crossing["update_lower"], crossing["update_upper"]), (0, 200))
        self.assertAlmostEqual(crossing["c_delta"], .68)
        self.assertTrue(crossing["eligible_fitted"])
        self.assertIsNone(first_loss_crossing(rows, .001))
        for row in rows:
            row["own_positive"] = 0
        self.assertFalse(first_loss_crossing(rows, .01)["eligible_fitted"])


if __name__ == "__main__":
    unittest.main()
