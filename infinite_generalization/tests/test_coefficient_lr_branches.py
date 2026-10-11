"""Check serialized continuation, optimizer grouping, and intervention isolation."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_coefficient_lr_branches import (
    BranchTrace, capture_parent, capture_rng, continue_training, forecast_first_update,
    loader_state, named_optimizer_state, restore_parent, restored_recorder,
    split_coefficient_group, tree_digest,
)
from run_training_length_pilot import TrainingRecorder, state_digest
from run_coefficient_lr_holdout import first_update_proposal
from stage3_simplified_attention import Stage3Config, project_dir, set_reproducibility, train_model


class CoefficientBranchTest(unittest.TestCase):
    def setUp(self):
        self.output = project_dir() / "runs/_test_coefficient_lr_branches" / self._testMethodName
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = Stage3Config(seed=3, device="cpu", alpha_mode="learned_log",
            alpha_log_scale_init=-5.0, train_length=10, train_lengths=(10,),
            train_examples=64, val_examples=32, batch_size=16, eval_batch_size=16,
            max_train_steps=8)
        self.device = torch.device("cpu")

    def parent(self):
        directory = self.output / "parent"
        directory.mkdir(exist_ok=True)
        config = replace(self.config, max_train_steps=4)
        recorder = TrainingRecorder(config, directory, 200)
        trace = BranchTrace(directory, {1, 2, 3, 4})
        saved = []
        def boundary(model, optimizer, train, val, epoch, updates, history):
            saved.append(capture_parent(self.config, model, optimizer, train, val,
                                       epoch, updates, history, recorder, trace))
        set_reproducibility(config.seed)
        train_model(config, device=self.device, output_dir=directory,
                    training_observer=recorder, update_observer=trace, epoch_observer=boundary)
        torch.save(saved[0], directory / "state.pt")
        return torch.load(directory / "state.pt", map_location="cpu", weights_only=False)

    def resume(self, bundle, tag, multiplier, traced=True):
        directory = self.output / tag
        directory.mkdir(exist_ok=True)
        recorder = restored_recorder(bundle, self.config, directory)
        trace = BranchTrace(directory, {5, 6, 7, 8}, 4) if traced else None
        result = continue_training(bundle, self.config, self.device, directory, multiplier, recorder, trace)
        return result, trace, capture_rng()

    def test_serialized_continuation_equals_uninterrupted(self):
        bundle = self.parent()
        directory = self.output / "uninterrupted"
        directory.mkdir(exist_ok=True)
        final = []
        def boundary(model, optimizer, train, val, epoch, updates, history):
            if updates == 8:
                final.append((named_optimizer_state(model, optimizer), loader_state(train), capture_rng()))
        set_reproducibility(self.config.seed)
        model, _ = train_model(self.config, device=self.device, output_dir=directory, epoch_observer=boundary)
        resumed, trace, rng = self.resume(bundle, "restored", 1.0)
        self.assertEqual(state_digest(model), state_digest(resumed[0]))
        self.assertEqual(tree_digest(final[0]), tree_digest((named_optimizer_state(resumed[0], resumed[1]), loader_state(resumed[2]), rng)))
        self.assertEqual((directory / "train_history.csv").read_bytes(), (self.output / "restored/train_history.csv").read_bytes())
        self.assertEqual(len(trace.rows), 4)
        self.assertLess(trace.max_allocation_residual, 1e-10)

    def test_grouping_and_trace_are_passive(self):
        bundle = self.parent()
        plain, _, a_rng = self.resume(bundle, "original", None, False)
        grouped, trace, b_rng = self.resume(bundle, "grouped", 1.0)
        self.assertEqual(state_digest(plain[0]), state_digest(grouped[0]))
        self.assertEqual(tree_digest(named_optimizer_state(plain[0], plain[1])), tree_digest(named_optimizer_state(grouped[0], grouped[1])))
        self.assertEqual(tree_digest(a_rng), tree_digest(b_rng))
        self.assertEqual((self.output / "original/train_history.csv").read_bytes(), (self.output / "grouped/train_history.csv").read_bytes())

    def test_optimizer_clones_and_lr_isolation(self):
        bundle = self.parent()
        digest = tree_digest(bundle)
        a, a_opt, _, _ = restore_parent(bundle, self.device)
        b, b_opt, _, _ = restore_parent(bundle, self.device)
        before = tree_digest(named_optimizer_state(a, a_opt))
        split_coefficient_group(a, a_opt, 3.0)
        self.assertEqual(before, tree_digest(named_optimizer_state(a, a_opt)))
        self.assertEqual(a_opt.param_groups[0]["lr"], self.config.learning_rate)
        self.assertEqual(a_opt.param_groups[1]["lr"], self.config.learning_rate * 3)
        for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
            self.assertNotEqual(pa.data_ptr(), pb.data_ptr())
            self.assertNotEqual(a_opt.state[pa]["exp_avg"].data_ptr(), b_opt.state[pb]["exp_avg"].data_ptr())
        with torch.no_grad():
            a.alpha_log_scale_unconstrained.add_(10)
            a_opt.state[a.alpha_log_scale_unconstrained]["exp_avg"].add_(100)
        self.assertEqual(digest, tree_digest(bundle))
        self.assertEqual(tree_digest(named_optimizer_state(b, b_opt)), tree_digest(bundle["named_optimizer"]))

    def test_frozen_forecast_and_first_update_isolation(self):
        bundle = self.parent()
        digest = tree_digest(bundle)
        forecast = forecast_first_update(bundle, self.device)
        first = []
        batches = []
        for tag, multiplier in (("slow", 1/3), ("control", 1.0), ("fast", 3.0)):
            _, trace, _ = self.resume(bundle, tag, multiplier)
            expected = next(item for item in forecast["predictions"] if item["tag"] == tag)
            self.assertEqual(trace.rows[0]["c_delta_after"], expected["r_after"])
            first.append(torch.load(self.output / tag / "first_update.pt", weights_only=False))
            batches.append(trace.batch_digest.hexdigest())
        self.assertEqual(digest, tree_digest(bundle))
        self.assertEqual(len(set(batches)), 1)
        for item in first[1:]:
            self.assertEqual(tree_digest(item["before"]), tree_digest(first[0]["before"]))
            self.assertEqual(tree_digest(item["after_moments"]), tree_digest(first[0]["after_moments"]))
            for name, weights in item["after_weights"].items():
                if name != "alpha_log_scale_unconstrained":
                    self.assertTrue(torch.equal(weights, first[0]["after_weights"][name]))
        self.assertEqual(len({item["delta_after"] for item in forecast["predictions"]}), 1)
        self.assertEqual(len({item["u_after"] for item in forecast["predictions"]}), 3)

    def test_multiplier2_proposal_matches_realized_continuation(self):
        bundle = self.parent()
        digest = tree_digest(bundle)
        proposal = first_update_proposal(bundle, self.device)
        _, trace, _ = self.resume(bundle, "x2", 2.0)
        first = torch.load(self.output / "x2/first_update.pt", weights_only=False)
        self.assertEqual(tree_digest(proposal["weights"]), tree_digest(first["after_weights"]))
        self.assertEqual(tree_digest(proposal["moments"]), tree_digest(first["after_moments"]))
        self.assertEqual(digest, tree_digest(bundle))
        self.assertEqual(trace.rows[0]["coefficient_lr"], 0.006)


if __name__ == "__main__":
    unittest.main()
