"""Independent checks for loss matching and head-aware failure classification."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analyze_training_length_confirmation import failure_status, match_loss


class ConfirmationAnalysisTest(unittest.TestCase):
    def test_loss_matching_interpolates_brackets(self):
        rows = [dict(optimizer_updates=0, loss=0.04, c=0.1, delta=2.0,
                     c_delta=0.2, head_threshold=0.3),
                dict(optimizer_updates=200, loss=0.02, c=0.2, delta=3.0,
                     c_delta=0.6, head_threshold=0.5)]
        matched = match_loss(rows, "loss", 0.03)
        self.assertAlmostEqual(matched["estimated_update"], 100)
        self.assertAlmostEqual(matched["c_delta"], 0.4)
        self.assertEqual((matched["update_lower"], matched["update_upper"]), (0, 200))
        self.assertIsNone(match_loss(rows, "loss", 0.01))

    def test_supercritical_mass_still_requires_head_check(self):
        values = dict(c=10.0, delta=0.2, c_delta=2.0, head_slope=1.0, head_threshold=0.8)
        self.assertEqual(failure_status(values)[0], "no_finite_crossing")
        values["head_threshold"] = 0.99
        self.assertEqual(failure_status(values)[0], "boundary_or_nonmonotone_threshold_case")

    def test_finite_prediction_satisfies_independent_attention_formula(self):
        values = dict(c=0.25, delta=2.0, c_delta=0.5, head_slope=1.0, head_threshold=0.3)
        status, log10_length = failure_status(values)
        self.assertEqual(status, "finite")
        length = 10 ** log10_length
        mass = 1 / (1 + (length - 1) * math.exp(-(1 + 0.25 * math.log1p(length)) * 2))
        self.assertAlmostEqual(mass, 0.3, places=12)


if __name__ == "__main__":
    unittest.main()
