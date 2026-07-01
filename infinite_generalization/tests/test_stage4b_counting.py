"""Tests for Stage 4B target occurrence counting."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
import math
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from stage4b_counting import (  # noqa: E402
    SimplifiedLastQueryAttentionCounter,
    Stage4BConfig,
    count_class_labels,
    evaluate_length,
    make_count_dataset,
    train_model,
)


def softmax_weights(scores: torch.Tensor) -> torch.Tensor:
    """Return per-row softmax weights that sum to 1."""

    return torch.softmax(scores, dim=-1)


def _without_metadata(
    rows: list[dict[str, Any]],
    metadata_keys: set[str],
) -> list[dict[str, Any]]:
    """Drop run metadata that is expected to differ across equivalent eval paths."""

    return [
        {
            key: _normalize_for_comparison(value)
            for key, value in row.items()
            if key not in metadata_keys
        }
        for row in rows
    ]


def _normalize_for_comparison(value: Any) -> Any:
    """Normalize NaN values because NaN is intentionally not self-equal."""

    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


class Stage4BDataTest(unittest.TestCase):
    """Validate Stage 4B count dataset generation."""

    def test_generated_labels_equal_true_target_count(self) -> None:
        inputs, labels, _, _ = make_count_dataset(
            length=9,
            examples=32,
            seed=123,
            target_position_mode="nonfinal_random",
            target_token_count=3,
            non_target_token_count=2,
            non_target_sampling="uniform",
            max_target_count=3,
        )
        true_counts = inputs.lt(3).sum(dim=1)
        self.assertEqual(labels.tolist(), true_counts.tolist())

    def test_count_labels_are_balanced(self) -> None:
        labels = count_class_labels(examples=24, max_target_count=3)
        self.assertEqual(Counter(labels.tolist()), Counter({0: 6, 1: 6, 2: 6, 3: 6}))

    def test_target_positions_are_distinct_when_count_is_greater_than_one(self) -> None:
        _, labels, target_positions, _ = make_count_dataset(
            length=12,
            examples=28,
            seed=456,
            target_position_mode="nonfinal_random",
            target_token_count=2,
            non_target_token_count=1,
            non_target_sampling="uniform",
            max_target_count=3,
        )
        for label, positions in zip(labels.tolist(), target_positions.tolist(), strict=True):
            active_positions = positions[:label]
            self.assertTrue(all(0 <= position < 11 for position in active_positions))
            if label > 1:
                self.assertEqual(len(active_positions), len(set(active_positions)))


class Stage4BModelTest(unittest.TestCase):
    """Validate the counting model value pathway and output head."""

    def make_model(
        self,
        *,
        target_token_count: int = 3,
        non_target_token_count: int = 2,
        max_target_count: int = 4,
        alpha_mode: str = "constant",
        readout_mode: str = "softmax_mass",
    ) -> SimplifiedLastQueryAttentionCounter:
        torch.manual_seed(0)
        return SimplifiedLastQueryAttentionCounter(
            d_head=2,
            alpha_mode=alpha_mode,
            alpha_log_scale_init=-5.0,
            target_token_count=target_token_count,
            non_target_token_count=non_target_token_count,
            max_target_count=max_target_count,
            readout_mode=readout_mode,
        )

    def test_value_masses_sum_to_one_per_example(self) -> None:
        model = self.make_model(target_token_count=3, non_target_token_count=2)
        tokens = torch.tensor([[0, 3, 1, 4, 2], [3, 4, 3, 4, 0]])
        torch.manual_seed(1)
        attn = softmax_weights(torch.randn(2, 5))
        value = model.token_value_output(tokens, attn)

        self.assertEqual(value.shape, (2, 4))
        self.assertTrue(torch.allclose(value.sum(dim=1), torch.ones(2), atol=1e-6))
        self.assertAlmostEqual(value[0, 0].item(), attn[0, 0].item(), places=6)

    def test_count_head_output_dimension_is_k_plus_one(self) -> None:
        max_target_count = 5
        model = self.make_model(
            target_token_count=1,
            non_target_token_count=1,
            max_target_count=max_target_count,
            readout_mode="target_numerator_only",
        )
        tokens = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
        logits = model(tokens)

        self.assertEqual(model.classifier.in_features, 1)
        self.assertEqual(model.classifier.out_features, max_target_count + 1)
        self.assertEqual(logits.shape, (2, max_target_count + 1))

    def test_target_numerator_only_output_shape_is_single_feature(self) -> None:
        model = self.make_model(
            target_token_count=1,
            non_target_token_count=1,
            max_target_count=3,
            readout_mode="target_numerator_only",
        )
        tokens = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
        _, details = model(tokens, return_details=True)

        self.assertEqual(details["attention_output"].shape, (2, 1))

    def test_unnormalized_sum_matches_exp_corrected_score_sums(self) -> None:
        model = self.make_model(
            target_token_count=2,
            non_target_token_count=1,
            max_target_count=3,
            readout_mode="unnormalized_sum",
        )
        tokens = torch.tensor([[0, 2, 1, 2], [2, 0, 0, 2]])
        _, details = model(tokens, return_details=True)
        numerators = torch.exp(details["corrected_scores"])
        expected = torch.stack(
            [
                numerators.masked_fill(tokens.ne(0), 0.0).sum(dim=1),
                numerators.masked_fill(tokens.ne(1), 0.0).sum(dim=1),
                numerators.masked_fill(tokens.lt(2), 0.0).sum(dim=1),
            ],
            dim=1,
        )

        self.assertTrue(torch.allclose(details["attention_output"], expected, atol=1e-6))
        self.assertTrue(
            torch.allclose(
                details["normalized_attention_output"].sum(dim=1),
                torch.ones(tokens.shape[0]),
                atol=1e-6,
            )
        )

    def test_unnormalized_target_readout_increases_with_true_count(self) -> None:
        model = self.make_model(
            target_token_count=1,
            non_target_token_count=1,
            max_target_count=3,
            readout_mode="unnormalized_sum",
        )
        tokens = torch.tensor(
            [
                [0, 1, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
            ]
        )
        _, details = model(tokens, return_details=True)
        target_readout = details["attention_output"][:, 0]

        self.assertGreater(target_readout[1].item(), target_readout[0].item())
        self.assertGreater(target_readout[2].item(), target_readout[1].item())

    def test_target_numerator_only_readout_is_proportional_to_true_count(self) -> None:
        model = self.make_model(
            target_token_count=1,
            non_target_token_count=1,
            max_target_count=3,
            readout_mode="target_numerator_only",
        )
        tokens = torch.tensor(
            [
                [0, 1, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
            ]
        )
        _, details = model(tokens, return_details=True)
        target_readout = details["attention_output"][:, 0]

        self.assertTrue(
            torch.allclose(
                target_readout,
                target_readout[0] * torch.tensor([1.0, 2.0, 3.0]),
                atol=1e-6,
            )
        )


class Stage4BEvalTest(unittest.TestCase):
    """Validate Stage 4B evaluation and chunking."""

    def make_model(
        self,
        *,
        readout_mode: str = "softmax_mass",
        target_token_count: int = 2,
    ) -> SimplifiedLastQueryAttentionCounter:
        torch.manual_seed(7)
        return SimplifiedLastQueryAttentionCounter(
            d_head=2,
            alpha_mode="learned_log",
            alpha_log_scale_init=-5.0,
            target_token_count=target_token_count,
            non_target_token_count=1,
            max_target_count=3,
            readout_mode=readout_mode,
        )

    def evaluate(
        self,
        *,
        examples: int,
        eval_chunk_examples: int,
        readout_mode: str = "softmax_mass",
        target_token_count: int = 2,
    ):
        return evaluate_length(
            self.make_model(
                readout_mode=readout_mode,
                target_token_count=target_token_count,
            ),
            length=10,
            examples=examples,
            eval_chunk_examples=eval_chunk_examples,
            eval_sampling_mode="stratified",
            batch_size=5,
            seed=789,
            target_position_mode="nonfinal_random",
            target_token_count=target_token_count,
            non_target_token_count=1,
            non_target_sampling="uniform",
            max_target_count=3,
            device=torch.device("cpu"),
        )

    def test_chunked_evaluation_matches_single_chunk_evaluation(self) -> None:
        single = self.evaluate(examples=24, eval_chunk_examples=24)
        chunked = self.evaluate(examples=24, eval_chunk_examples=5)
        self.assert_evaluations_match(single, chunked)

    def test_unnormalized_chunked_evaluation_matches_single_chunk_evaluation(self) -> None:
        single = self.evaluate(
            examples=24,
            eval_chunk_examples=24,
            readout_mode="unnormalized_sum",
        )
        chunked = self.evaluate(
            examples=24,
            eval_chunk_examples=5,
            readout_mode="unnormalized_sum",
        )
        self.assert_evaluations_match(single, chunked)

    def test_target_only_chunked_evaluation_matches_single_chunk_evaluation(self) -> None:
        single = self.evaluate(
            examples=24,
            eval_chunk_examples=24,
            readout_mode="target_numerator_only",
            target_token_count=1,
        )
        chunked = self.evaluate(
            examples=24,
            eval_chunk_examples=5,
            readout_mode="target_numerator_only",
            target_token_count=1,
        )
        self.assert_evaluations_match(single, chunked)

    def assert_evaluations_match(self, single, chunked) -> None:
        """Assert equal evaluation outputs except chunk-size metadata."""

        single_metrics, single_counts, single_confusion, single_types = single
        chunked_metrics, chunked_counts, chunked_confusion, chunked_types = chunked

        metadata_keys = {"eval_chunk_examples"}
        self.assertEqual(len(single_metrics), 1)
        self.assertEqual(len(chunked_metrics), 1)
        for key, single_value in single_metrics[0].items():
            if key in metadata_keys:
                continue
            chunked_value = chunked_metrics[0][key]
            if isinstance(single_value, float):
                if math.isnan(single_value) and math.isnan(chunked_value):
                    continue
                self.assertAlmostEqual(single_value, chunked_value, places=6, msg=key)
            else:
                self.assertEqual(single_value, chunked_value, key)
        self.assertEqual(
            _without_metadata(single_counts, metadata_keys),
            _without_metadata(chunked_counts, metadata_keys),
        )
        self.assertEqual(single_confusion, chunked_confusion)
        self.assertEqual(single_types, chunked_types)


class Stage4BDefaultRunTest(unittest.TestCase):
    """End-to-end smoke at the unit-test scale for each alpha mode."""

    def test_tiny_train_and_eval_runs_for_each_alpha_mode(self) -> None:
        base_output_dir = ROOT / "runs" / "_test_stage4b_counting"
        base_output_dir.mkdir(parents=True, exist_ok=True)

        for alpha_mode in ("constant", "log", "learned_log"):
            with self.subTest(alpha_mode=alpha_mode):
                output_dir = base_output_dir / alpha_mode
                output_dir.mkdir(exist_ok=True)
                config = Stage4BConfig(
                    seed=42,
                    device="cpu",
                    output_dir=str(output_dir),
                    alpha_mode=alpha_mode,
                    train_length=8,
                    target_position_mode="nonfinal_random",
                    target_token_count=1,
                    non_target_token_count=1,
                    max_target_count=2,
                    train_examples=24,
                    val_examples=12,
                    test_examples=12,
                    eval_chunk_examples=5,
                    eval_sampling_mode="stratified",
                    eval_lengths=(8, 12),
                    batch_size=8,
                    eval_batch_size=4,
                    epochs=1,
                    max_train_steps=2,
                )
                model, updates = train_model(
                    config,
                    device=torch.device("cpu"),
                    output_dir=output_dir,
                )
                self.assertGreater(updates, 0)
                metric_rows, count_rows, confusion_rows, target_type_rows = evaluate_length(
                    model,
                    length=12,
                    examples=config.test_examples,
                    eval_chunk_examples=config.eval_chunk_examples,
                    eval_sampling_mode=config.eval_sampling_mode,
                    batch_size=config.eval_batch_size,
                    seed=config.seed + 10_000 + 12,
                    target_position_mode=config.target_position_mode,
                    target_token_count=config.target_token_count,
                    non_target_token_count=config.non_target_token_count,
                    non_target_sampling=config.non_target_sampling,
                    max_target_count=config.max_target_count,
                    device=torch.device("cpu"),
                )
                self.assertEqual(len(metric_rows), 1)
                self.assertEqual(len(count_rows), config.max_target_count + 1)
                self.assertEqual(
                    len(confusion_rows),
                    (config.max_target_count + 1) * (config.max_target_count + 1),
                )
                self.assertEqual(len(target_type_rows), config.target_token_count)
                self.assertTrue(0.0 <= metric_rows[0]["accuracy"] <= 1.0)

    def test_tiny_train_and_eval_runs_for_unnormalized_sum(self) -> None:
        base_output_dir = ROOT / "runs" / "_test_stage4b_counting"
        base_output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = base_output_dir / "unnormalized_sum"
        output_dir.mkdir(exist_ok=True)
        config = Stage4BConfig(
            seed=43,
            device="cpu",
            output_dir=str(output_dir),
            alpha_mode="constant",
            readout_mode="unnormalized_sum",
            train_length=8,
            target_position_mode="nonfinal_random",
            target_token_count=1,
            non_target_token_count=1,
            max_target_count=2,
            train_examples=24,
            val_examples=12,
            test_examples=12,
            eval_chunk_examples=5,
            eval_sampling_mode="stratified",
            eval_lengths=(8, 12),
            batch_size=8,
            eval_batch_size=4,
            epochs=1,
            max_train_steps=2,
        )

        model, updates = train_model(
            config,
            device=torch.device("cpu"),
            output_dir=output_dir,
        )
        self.assertGreater(updates, 0)
        metric_rows, count_rows, confusion_rows, target_type_rows = evaluate_length(
            model,
            length=12,
            examples=config.test_examples,
            eval_chunk_examples=config.eval_chunk_examples,
            eval_sampling_mode=config.eval_sampling_mode,
            batch_size=config.eval_batch_size,
            seed=config.seed + 10_000 + 12,
            target_position_mode=config.target_position_mode,
            target_token_count=config.target_token_count,
            non_target_token_count=config.non_target_token_count,
            non_target_sampling=config.non_target_sampling,
            max_target_count=config.max_target_count,
            device=torch.device("cpu"),
        )

        self.assertEqual(metric_rows[0]["readout_mode"], "unnormalized_sum")
        self.assertEqual(len(count_rows), config.max_target_count + 1)
        self.assertEqual(
            len(confusion_rows),
            (config.max_target_count + 1) * (config.max_target_count + 1),
        )
        self.assertEqual(len(target_type_rows), config.target_token_count)
        self.assertTrue(0.0 <= metric_rows[0]["accuracy"] <= 1.0)

    def test_tiny_train_and_eval_runs_for_target_numerator_only(self) -> None:
        base_output_dir = ROOT / "runs" / "_test_stage4b_counting"
        base_output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = base_output_dir / "target_numerator_only"
        output_dir.mkdir(exist_ok=True)
        config = Stage4BConfig(
            seed=44,
            device="cpu",
            output_dir=str(output_dir),
            alpha_mode="constant",
            readout_mode="target_numerator_only",
            train_length=8,
            target_position_mode="nonfinal_random",
            target_token_count=1,
            non_target_token_count=1,
            max_target_count=2,
            train_examples=24,
            val_examples=12,
            test_examples=12,
            eval_chunk_examples=5,
            eval_sampling_mode="stratified",
            eval_lengths=(8, 12),
            batch_size=8,
            eval_batch_size=4,
            epochs=1,
            max_train_steps=2,
        )

        model, updates = train_model(
            config,
            device=torch.device("cpu"),
            output_dir=output_dir,
        )
        self.assertGreater(updates, 0)
        metric_rows, count_rows, confusion_rows, target_type_rows = evaluate_length(
            model,
            length=12,
            examples=config.test_examples,
            eval_chunk_examples=config.eval_chunk_examples,
            eval_sampling_mode=config.eval_sampling_mode,
            batch_size=config.eval_batch_size,
            seed=config.seed + 10_000 + 12,
            target_position_mode=config.target_position_mode,
            target_token_count=config.target_token_count,
            non_target_token_count=config.non_target_token_count,
            non_target_sampling=config.non_target_sampling,
            max_target_count=config.max_target_count,
            device=torch.device("cpu"),
        )

        self.assertEqual(metric_rows[0]["readout_mode"], "target_numerator_only")
        self.assertEqual(len(count_rows), config.max_target_count + 1)
        self.assertEqual(
            len(confusion_rows),
            (config.max_target_count + 1) * (config.max_target_count + 1),
        )
        self.assertEqual(len(target_type_rows), config.target_token_count)
        self.assertTrue(0.0 <= metric_rows[0]["accuracy"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
