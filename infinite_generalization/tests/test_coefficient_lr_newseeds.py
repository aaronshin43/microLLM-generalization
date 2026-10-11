"""Protect conditional forecasts, failed-parent handling, and censoring."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_coefficient_lr_branches import FIELDS
from run_coefficient_lr_newseeds import predict_from_control


class IndependentSeedPredictionTest(unittest.TestCase):
    def inputs(self):
        rule = {"coefficients": [
            {"length": 10, "comparison": "update6400", "normalized_effect_mean": 0.75},
            {"length": 10, "comparison": "common100_bce0.01", "normalized_effect_mean": 0.8}]}
        path = []
        for step, r, loss in ((1600, 0.4, 0.03), (6400, 1.0, 0.005)):
            row = {key: 0.0 for key in FIELDS}
            row.update(optimizer_updates=step, c_delta=r, common100_bce=loss)
            path.append(row)
        return rule, path

    def test_prediction_uses_post_branch_growth_without_seed_adjustment(self):
        rule, path = self.inputs()
        a = predict_from_control(rule, 8, 10, 0.4, path, True)
        b = predict_from_control(rule, 9, 10, 0.4, path, True)
        self.assertAlmostEqual(a[0]["predicted_effect"], 0.45)
        self.assertAlmostEqual(a[0]["predicted_r"], 1.45)
        self.assertAlmostEqual(a[0]["effect_lower"], 0.36)
        self.assertAlmostEqual(a[0]["effect_upper"], 0.54)
        self.assertEqual(a[0]["predicted_effect"], b[0]["predicted_effect"])
        self.assertAlmostEqual(a[1]["control_r"], 0.88)
        self.assertAlmostEqual(a[1]["predicted_effect"], 0.384)

    def test_failed_parent_is_not_scored_despite_later_control_success(self):
        rule, path = self.inputs()
        rows = predict_from_control(rule, 8, 10, 0.4, path, False)
        self.assertTrue(all(not row["evaluable"] and row["predicted_effect"] is None for row in rows))
        self.assertTrue(all(row["reason"] == "parent_not_fitted_at1600" for row in rows))

    def test_unreached_loss_and_nonpositive_growth_are_not_extrapolated(self):
        rule, path = self.inputs()
        path[-1]["common100_bce"] = 0.02
        rows = predict_from_control(rule, 9, 10, 1.0, path, True)
        self.assertEqual(rows[0]["reason"], "nonpositive_control_growth")
        self.assertEqual(rows[1]["reason"], "control_target_censored")
        self.assertTrue(all(not row["evaluable"] for row in rows))


if __name__ == "__main__":
    unittest.main()
