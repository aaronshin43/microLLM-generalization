"""Check multi-group state restoration and fresh global-step trace headers."""

import copy
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_coefficient_lr_branches import capture_parent, read_rows, tree_digest
from run_training_length_extension import COEFFICIENT, continue_path, parameter_groups, restore_start
from run_training_length_pilot import TrainingRecorder
from run_training_length_sweep import CoefficientTrace
from analyze_training_length_extension import crossing_summary, reconstruct_boundary_memory
from stage3_simplified_attention import Stage3Config, set_reproducibility, train_model


class ExtensionRestorationTest(unittest.TestCase):
    def parent(self, output):
        config = Stage3Config(seed=3, device="cpu", train_length=10, train_lengths=(10,),
            alpha_mode="learned_log", train_examples=64, val_examples=32,
            batch_size=16, eval_batch_size=16, max_train_steps=4)
        recorder = TrainingRecorder(config, output, 200)
        trace = CoefficientTrace(output, 4)
        captured = []
        def boundary(model, optimizer, train, val, epoch, updates, history):
            captured.append(capture_parent(config, model, optimizer, train, val, epoch, updates, history, recorder, trace))
        set_reproducibility(3)
        train_model(config, device=torch.device("cpu"), output_dir=output,
            training_observer=recorder, update_observer=trace, epoch_observer=boundary)
        return config, captured[-1]

    def two_groups(self, bundle):
        bundle = copy.deepcopy(bundle)
        mapping = [[n for n in bundle["parameter_names"] if n != COEFFICIENT], [COEFFICIENT]]
        options = {k: v for k, v in bundle["optimizer"]["param_groups"][0].items() if k != "params"}
        groups, states, counter = [], {}, 0
        for names in mapping:
            identifiers = []
            for name in names:
                identifiers.append(counter)
                states[counter] = copy.deepcopy(bundle["named_optimizer"][name])
                counter += 1
            groups.append({**copy.deepcopy(options), "params": identifiers})
        bundle["optimizer"] = {"state": states, "param_groups": groups}
        return bundle

    def test_restore_preserves_source_group_layout_and_next_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "parent"
            source.mkdir()
            config, original = self.parent(source)
            grouped = self.two_groups(original)
            self.assertEqual(parameter_groups(grouped)[1], [COEFFICIENT])
            model, optimizer, _, _ = restore_start(grouped, torch.device("cpu"))
            self.assertEqual(len(optimizer.param_groups), 2)
            self.assertEqual(tree_digest(optimizer.state_dict()), tree_digest(grouped["optimizer"]))
            results = []
            for label, bundle in (("one", original), ("two", grouped)):
                directory = root / label
                directory.mkdir()
                results.append(continue_path(bundle, replace(config, max_train_steps=8), torch.device("cpu"), directory, witness=True))
            for field in ("model", "named_optimizer", "rng", "train_loaders", "val_loaders", "history"):
                self.assertEqual(tree_digest(results[0][3][field]), tree_digest(results[1][3][field]))
            self.assertEqual(results[0][5], results[1][5])
            rows = read_rows(root / "two/coefficient_trace.csv")
            self.assertEqual([int(r["optimizer_updates"]) for r in rows], [5, 6, 7, 8])
            self.assertIn("u_gradient", rows[0])
            bad = copy.deepcopy(grouped)
            bad["optimizer_group_parameter_names"] = [[COEFFICIENT, *grouped["parameter_names"][2:]], [grouped["parameter_names"][1]]]
            with self.assertRaises(AssertionError):
                parameter_groups(bad)


class ExtensionAnalysisTest(unittest.TestCase):
    def test_entry_is_not_confused_with_an_already_above_start(self):
        earlier = [{"optimizer_updates": 0, "c_delta": .1}, {"optimizer_updates": 200, "c_delta": 1.1}]
        above = [{"optimizer_updates": step, "r_before": 1.2, "r_after": 1.2} for step in range(6401, 12801)]
        summary, events = crossing_summary(above, earlier)
        self.assertIsNone(summary["first_new_entry"])
        self.assertFalse(summary["right_censored_at12800"])
        self.assertEqual(summary["historical_first_entry_grid_bracket"], "(0, 200]")
        self.assertEqual(events, [])
        trace, before = [], .9
        for step in range(6401, 12801):
            after = .99 if step >= 6800 else 1.01 if step >= 6404 else .9
            trace.append({"optimizer_updates": step, "r_before": before, "r_after": after})
            before = after
        summary, events = crossing_summary(trace, [{"optimizer_updates": 6400, "c_delta": .9}])
        self.assertEqual(summary["first_new_entry"], 6404)
        self.assertEqual(summary["exit_count"], 1)
        self.assertFalse(summary["all_steps_above_after_start_or_entry"])
        for row in trace:
            row["r_before"] = row["r_after"] = .9
        self.assertTrue(crossing_summary(trace, [])[0]["right_censored_at12800"])

    def test_second_moment_preserves_nonzero_boundary_and_missing_history(self):
        beta, initial = .996, .004
        v = torch.tensor(initial)
        rows = []
        for step in range(6401, 12801):
            gradient = torch.tensor(.002 + .001 * ((step % 13) / 13))
            v.mul_(beta).addcmul_(gradient, gradient, value=1-beta)
            rows.append({"optimizer_updates": step, "beta2": beta, "u_gradient": float(gradient), "v_after": float(v)})
        for endpoint in (9600, 12800):
            result = reconstruct_boundary_memory(rows, float(torch.tensor(initial)), endpoint)
            self.assertTrue(result["passed"])
            self.assertAlmostEqual(result["inherited_contribution"], beta ** (endpoint - 6400) * float(torch.tensor(initial)))
            self.assertAlmostEqual(result["inherited_fraction"] + result["new_gradient_fraction"], 1, places=4)
        with self.assertRaises(AssertionError):
            reconstruct_boundary_memory(rows[1:], initial, 9600)


if __name__ == "__main__":
    unittest.main()
