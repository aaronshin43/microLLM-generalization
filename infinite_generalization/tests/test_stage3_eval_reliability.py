"""Tests for Stage 3 chunked and stratified evaluation."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from math import isnan
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from stage3_simplified_attention import (  # noqa: E402
    Stage3Config,
    SimplifiedLastQueryAttentionClassifier,
    evaluate_length,
    iter_eval_batches,
    make_stratified_eval_dataset,
    make_two_token_dataset,
    target_position_bucket,
)


class Stage3EvalReliabilityTest(unittest.TestCase):
    """Validate the reliable Stage 3 evaluation path."""

    def make_model(self) -> SimplifiedLastQueryAttentionClassifier:
        """Create a deterministic tiny model for evaluation tests."""

        torch.manual_seed(123)
        return SimplifiedLastQueryAttentionClassifier(
            d_head=2,
            alpha_mode="learned_log",
            alpha_log_scale_init=-5.0,
            target_token_count=1,
            non_target_token_count=1,
        )

    def evaluate_tiny_model(
        self,
        *,
        examples: int,
        eval_chunk_examples: int,
        eval_sampling_mode: str = "random",
    ) -> dict[str, float | int | str]:
        """Run a small deterministic evaluation and return the length-level row."""

        row, _, _, _ = evaluate_length(
            self.make_model(),
            length=10,
            examples=examples,
            eval_chunk_examples=eval_chunk_examples,
            eval_sampling_mode=eval_sampling_mode,
            batch_size=4,
            seed=777,
            target_position_mode="fixed_start",
            target_token_count=1,
            non_target_token_count=1,
            non_target_sampling="uniform",
            device=torch.device("cpu"),
        )
        return row

    def test_single_chunk_random_evaluation_is_stable(self) -> None:
        row_exact_chunk = self.evaluate_tiny_model(examples=24, eval_chunk_examples=24)
        row_larger_chunk = self.evaluate_tiny_model(examples=24, eval_chunk_examples=100)

        comparable_keys = [
            key
            for key in row_exact_chunk
            if key not in {"eval_chunk_examples"}
        ]
        for key in comparable_keys:
            if isinstance(row_exact_chunk[key], float) and isnan(row_exact_chunk[key]):
                self.assertTrue(isinstance(row_larger_chunk[key], float))
                self.assertTrue(isnan(row_larger_chunk[key]))
                continue
            self.assertEqual(row_exact_chunk[key], row_larger_chunk[key])

    def test_single_chunk_random_matches_unchunked_dataset(self) -> None:
        # The reliable evaluation path with a single chunk must reproduce the
        # original unchunked random dataset byte-for-byte. This guards against
        # the refactor silently shifting the examples behind existing Stage 3
        # conclusions. A non-trivial config exercises random target positions
        # and multiple target / non-target token types.
        kwargs = dict(
            length=37,
            target_position_mode="nonfinal_random",
            target_token_count=3,
            non_target_token_count=4,
            non_target_sampling="uniform",
        )
        base_seed = 4242
        expected = make_two_token_dataset(examples=50, seed=base_seed, **kwargs)

        batches = list(
            iter_eval_batches(
                examples=50,
                eval_chunk_examples=50,
                batch_size=16,
                seed=base_seed,
                eval_sampling_mode="random",
                **kwargs,
            )
        )
        actual_inputs = torch.cat([batch[0] for batch in batches])
        actual_labels = torch.cat([batch[1] for batch in batches])
        actual_positions = torch.cat([batch[2] for batch in batches])
        actual_target_ids = torch.cat([batch[3] for batch in batches])

        self.assertTrue(torch.equal(actual_inputs, expected[0]))
        self.assertTrue(torch.equal(actual_labels, expected[1]))
        self.assertTrue(torch.equal(actual_positions, expected[2]))
        self.assertTrue(torch.equal(actual_target_ids, expected[3]))

    def test_chunked_random_evaluation_consumes_total_examples(self) -> None:
        row = self.evaluate_tiny_model(examples=24, eval_chunk_examples=6)

        self.assertEqual(row["test_examples"], 24)
        self.assertEqual(row["positive_examples"], 12)
        self.assertEqual(row["negative_examples"], 12)

    def test_chunked_random_odd_total_consumes_all_examples(self) -> None:
        # An odd total that does not divide evenly into chunks must still
        # consume exactly test_examples, with the balanced split preserved.
        row = self.evaluate_tiny_model(examples=25, eval_chunk_examples=6)

        self.assertEqual(row["test_examples"], 25)
        self.assertEqual(row["positive_examples"], 12)
        self.assertEqual(row["negative_examples"], 13)
        self.assertEqual(
            row["positive_examples"] + row["negative_examples"],
            25,
        )

    def test_stratified_stage3c_balances_target_position_buckets(self) -> None:
        _, labels, target_positions, _ = make_stratified_eval_dataset(
            length=10,
            positive_count=12,
            negative_count=12,
            seed=1,
            target_position_mode="nonfinal_random",
            target_token_count=1,
            non_target_token_count=1,
            non_target_sampling="uniform",
        )

        positive_positions = target_positions[labels.eq(1)]
        buckets = [target_position_bucket(int(position), 10) for position in positive_positions]
        self.assertEqual(Counter(buckets), Counter({"beginning": 4, "middle": 4, "end_nonfinal": 4}))

    def test_stratified_stage3d_balances_final_query_token_ids(self) -> None:
        inputs, labels, _, _ = make_stratified_eval_dataset(
            length=10,
            positive_count=12,
            negative_count=12,
            seed=2,
            target_position_mode="fixed_start",
            target_token_count=1,
            non_target_token_count=4,
            non_target_sampling="uniform",
        )

        final_ids = inputs[:, -1]
        self.assertEqual(Counter(final_ids[labels.eq(1)].tolist()), Counter({1: 3, 2: 3, 3: 3, 4: 3}))
        self.assertEqual(Counter(final_ids[labels.eq(0)].tolist()), Counter({1: 3, 2: 3, 3: 3, 4: 3}))

    def test_stratified_stage3e_balances_positive_target_token_ids(self) -> None:
        _, labels, _, target_ids = make_stratified_eval_dataset(
            length=10,
            positive_count=12,
            negative_count=12,
            seed=3,
            target_position_mode="fixed_start",
            target_token_count=3,
            non_target_token_count=1,
            non_target_sampling="uniform",
        )

        self.assertEqual(Counter(target_ids[labels.eq(1)].tolist()), Counter({0: 4, 1: 4, 2: 4}))

    def test_combined_stratification_covers_positive_cartesian_product(self) -> None:
        inputs, labels, target_positions, target_ids = make_stratified_eval_dataset(
            length=10,
            positive_count=36,
            negative_count=36,
            seed=4,
            target_position_mode="nonfinal_random",
            target_token_count=3,
            non_target_token_count=4,
            non_target_sampling="uniform",
        )

        positive_mask = labels.eq(1)
        positive_positions = target_positions[positive_mask]
        positive_final_ids = inputs[positive_mask, -1]
        positive_target_ids = target_ids[positive_mask]
        strata = Counter(
            (
                target_position_bucket(int(position), 10),
                int(final_id),
                int(target_id),
            )
            for position, final_id, target_id in zip(
                positive_positions,
                positive_final_ids,
                positive_target_ids,
                strict=True,
            )
        )

        self.assertEqual(len(strata), 36)
        self.assertTrue(all(count == 1 for count in strata.values()))

    def test_chunked_combined_stratification_preserves_global_balance(self) -> None:
        batches = list(
            iter_eval_batches(
                length=10,
                examples=72,
                eval_chunk_examples=12,
                batch_size=12,
                seed=5,
                target_position_mode="nonfinal_random",
                target_token_count=3,
                non_target_token_count=4,
                non_target_sampling="uniform",
                eval_sampling_mode="stratified",
            )
        )
        inputs = torch.cat([batch[0] for batch in batches])
        labels = torch.cat([batch[1] for batch in batches])
        target_positions = torch.cat([batch[2] for batch in batches])
        target_ids = torch.cat([batch[3] for batch in batches])

        positive_mask = labels.eq(1)
        strata = Counter(
            (
                target_position_bucket(int(position), 10),
                int(final_id),
                int(target_id),
            )
            for position, final_id, target_id in zip(
                target_positions[positive_mask],
                inputs[positive_mask, -1],
                target_ids[positive_mask],
                strict=True,
            )
        )

        self.assertEqual(len(strata), 36)
        self.assertTrue(all(count == 1 for count in strata.values()))

    def test_chunked_stratified_balances_negative_final_query_ids(self) -> None:
        # Negative examples carry no target, but worst-case non-target analysis
        # still needs them balanced over the final query token id. Verify that
        # balance holds globally across chunk boundaries, not just per chunk.
        batches = list(
            iter_eval_batches(
                length=10,
                examples=72,
                eval_chunk_examples=12,
                batch_size=12,
                seed=8,
                target_position_mode="fixed_start",
                target_token_count=1,
                non_target_token_count=4,
                non_target_sampling="uniform",
                eval_sampling_mode="stratified",
            )
        )
        inputs = torch.cat([batch[0] for batch in batches])
        labels = torch.cat([batch[1] for batch in batches])

        negative_final_ids = inputs[labels.eq(0), -1]
        self.assertEqual(
            Counter(negative_final_ids.tolist()),
            Counter({1: 9, 2: 9, 3: 9, 4: 9}),
        )

    def test_default_config_uses_random_single_chunk_evaluation(self) -> None:
        config = Stage3Config()

        self.assertEqual(config.eval_sampling_mode, "random")
        self.assertEqual(config.eval_chunk_examples, 50)


if __name__ == "__main__":
    unittest.main()
