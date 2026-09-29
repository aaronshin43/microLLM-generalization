"""Verify passive training observation preserves the existing experiment path."""

import csv
import sys
import unittest
import uuid
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stage3_simplified_attention import Stage3Config, set_reproducibility, train_model
from run_training_length_pilot import predicted_failure_log10
from run_training_length_pilot import run_one


class TrainingObservationTest(unittest.TestCase):
    def test_observer_preserves_weights_rng_and_history(self):
        config = Stage3Config(
            seed=7, train_lengths=(10, 20), train_examples=64, val_examples=32,
            batch_size=16, eval_batch_size=16, max_train_steps=5, alpha_mode="learned_log",
        )
        observed_batches = []
        initial_states = []

        def observe(model, tokens, labels):
            if tokens is None:
                initial_states.append({key: value.clone() for key, value in model.state_dict().items()})
            else:
                observed_batches.append((tokens.shape[0], tokens.shape[1], int(labels.sum())))
                with torch.no_grad():
                    model(torch.tensor([[0] + [1] * 9, [1] * 10]))

        temporary_root = Path(__file__).resolve().parents[1] / "runs" / "_test_training_observer"
        temporary_root.mkdir(parents=True, exist_ok=True)
        temporary = temporary_root / uuid.uuid4().hex
        temporary.mkdir()
        with nullcontext(temporary):
            paths = [Path(temporary) / "plain", Path(temporary) / "observed"]
            models = []
            rng_states = []
            histories = []
            for path, observer in zip(paths, [None, observe]):
                path.mkdir()
                set_reproducibility(config.seed)
                model, updates = train_model(config, device=torch.device("cpu"), output_dir=path,
                                             training_observer=observer)
                self.assertEqual(updates, 5)
                models.append(model)
                rng_states.append(torch.get_rng_state())
                with (path / "train_history.csv").open(newline="") as stream:
                    histories.append(list(csv.DictReader(stream)))
            self.assertEqual(len(initial_states), 1)
            self.assertEqual(len(observed_batches), 5)
            self.assertEqual(sum(batch[0] for batch in observed_batches), 80)
            self.assertEqual(histories[0], histories[1])
            self.assertTrue(torch.equal(rng_states[0], rng_states[1]))
            for key, value in models[0].state_dict().items():
                self.assertTrue(torch.equal(value, models[1].state_dict()[key]), key)

    def test_predicted_failure_matches_constant_closed_form(self):
        import math

        values = {"delta": 2.0, "head_slope": 3.0, "head_threshold": 0.3}
        expected = math.log10(1 + math.exp(2) * (1 - 0.3) / 0.3)
        self.assertAlmostEqual(predicted_failure_log10("constant", values), expected, places=12)
        self.assertIsNone(predicted_failure_log10("log", values))

    def test_fresh_longer_run_verifies_reference_prefix(self):
        from dataclasses import replace
        import json

        root = Path(__file__).resolve().parents[1] / "runs" / "_test_training_observer" / uuid.uuid4().hex
        root.mkdir(parents=True)
        config = Stage3Config(seed=9, alpha_mode="learned_log", train_lengths=(10, 20),
                              train_examples=64, val_examples=32, batch_size=64,
                              test_examples=4, eval_chunk_examples=4, eval_batch_size=4,
                              max_train_steps=4)
        run_one(config, root / "reference", torch.device("cpu"), 2)
        run_one(replace(config, max_train_steps=8), root / "longer", torch.device("cpu"), 2,
                reference_run=root / "reference", verify_step=4)
        observed_rng = torch.get_rng_state()
        run_one(replace(config, max_train_steps=8), root / "plain_longer", torch.device("cpu"), 2)
        self.assertTrue(torch.equal(observed_rng, torch.get_rng_state()))
        observed_state = torch.load(root / "longer/model.pt", weights_only=False)["state_dict"]
        plain_state = torch.load(root / "plain_longer/model.pt", weights_only=False)["state_dict"]
        self.assertTrue(all(torch.equal(value, plain_state[key]) for key, value in observed_state.items()))
        self.assertEqual((root / "longer/train_history.csv").read_bytes(),
                         (root / "plain_longer/train_history.csv").read_bytes())
        record = json.loads((root / "longer/pilot_prefix_verification.json").read_text())
        self.assertTrue(record["weights_bitwise_equal"])
        self.assertEqual(record["diagnostic_rows_exactly_equal"], 12)
        self.assertEqual(record["train_history_rows_exactly_equal"], 2)


if __name__ == "__main__":
    unittest.main()
