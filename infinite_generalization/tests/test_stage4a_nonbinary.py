"""Tests for Stage 4A non-binary target classification."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from stage3_simplified_attention import (  # noqa: E402
    SimplifiedLastQueryAttentionClassifier,
    make_two_token_dataset,
)
from stage4a_nonbinary_classification import (  # noqa: E402
    SimplifiedLastQueryAttentionMultiClass,
    Stage4AConfig,
    class_labels_from_presence,
    evaluate_length,
    none_class_index,
    train_model,
)


def softmax_weights(scores: torch.Tensor) -> torch.Tensor:
    """Return per-row softmax weights that sum to 1."""

    return torch.softmax(scores, dim=-1)


class Stage4AModelTest(unittest.TestCase):
    """Validate the multi-class value pathway and label mapping."""

    def make_model(
        self, *, target_token_count: int, non_target_token_count: int
    ) -> SimplifiedLastQueryAttentionMultiClass:
        torch.manual_seed(0)
        return SimplifiedLastQueryAttentionMultiClass(
            d_head=2,
            alpha_mode="constant",
            alpha_log_scale_init=-5.0,
            target_token_count=target_token_count,
            non_target_token_count=non_target_token_count,
        )

    def test_value_output_dim_and_mass_conservation(self) -> None:
        model = self.make_model(target_token_count=3, non_target_token_count=2)
        # tokens use target ids 0,1,2 and non-target ids 3,4.
        tokens = torch.tensor([[0, 3, 1, 4, 2], [3, 4, 3, 4, 0]])
        torch.manual_seed(1)
        attn = softmax_weights(torch.randn(2, 5))
        value = model.token_value_output(tokens, attn)

        self.assertEqual(value.shape, (2, 4))  # H + 1 = 4
        self.assertTrue(torch.allclose(value.sum(dim=1), torch.ones(2), atol=1e-6))
        # Column 0 must equal the attention mass on target id 0 in row 0.
        self.assertAlmostEqual(value[0, 0].item(), attn[0, 0].item(), places=6)

    def test_h1_reduces_to_binary_value_output(self) -> None:
        multi = self.make_model(target_token_count=1, non_target_token_count=1)
        torch.manual_seed(2)
        binary = SimplifiedLastQueryAttentionClassifier(
            d_head=2,
            alpha_mode="constant",
            alpha_log_scale_init=-5.0,
            target_token_count=1,
            non_target_token_count=1,
        )
        tokens = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
        torch.manual_seed(3)
        attn = softmax_weights(torch.randn(2, 4))

        self.assertTrue(
            torch.allclose(
                multi.token_value_output(tokens, attn),
                binary.token_value_output(tokens, attn),
                atol=1e-6,
            )
        )

    def test_class_labels_from_presence(self) -> None:
        presence = torch.tensor([1.0, 0.0, 1.0, 0.0])
        target_ids = torch.tensor([2, -1, 0, -1])
        labels = class_labels_from_presence(presence, target_ids, target_token_count=3)
        # none class index is 3 for H = 3.
        self.assertEqual(labels.tolist(), [2, 3, 0, 3])
        self.assertEqual(none_class_index(3), 3)


class Stage4AEvalTest(unittest.TestCase):
    """Validate the Stage 4A evaluation path."""

    def make_model(self) -> SimplifiedLastQueryAttentionMultiClass:
        torch.manual_seed(7)
        return SimplifiedLastQueryAttentionMultiClass(
            d_head=2,
            alpha_mode="learned_log",
            alpha_log_scale_init=-5.0,
            target_token_count=3,
            non_target_token_count=1,
        )

    def evaluate(self, *, examples: int, eval_chunk_examples: int, mode: str):
        return evaluate_length(
            self.make_model(),
            length=10,
            examples=examples,
            eval_chunk_examples=eval_chunk_examples,
            eval_sampling_mode=mode,
            batch_size=4,
            seed=123,
            target_position_mode="fixed_start",
            target_token_count=3,
            non_target_token_count=1,
            non_target_sampling="uniform",
            device=torch.device("cpu"),
        )

    def test_chunked_evaluation_consumes_all_examples(self) -> None:
        row, _ = self.evaluate(examples=24, eval_chunk_examples=6, mode="random")
        self.assertEqual(row["positive_examples"] + row["negative_examples"], 24)
        self.assertEqual(row["num_classes"], 4)

    def test_stratified_balances_per_target_type(self) -> None:
        # 24 examples -> 12 positives over 3 target types -> 4 per type.
        _, type_rows = self.evaluate(examples=24, eval_chunk_examples=24, mode="stratified")
        counts = Counter()
        for r in type_rows:
            counts[r["target_token_id"]] = r["positive_examples"]
        self.assertEqual(counts, Counter({0: 4, 1: 4, 2: 4}))

    def test_chunked_stratified_matches_single_chunk_counts(self) -> None:
        # Per-type positive counts must be preserved across chunk boundaries.
        _, single = self.evaluate(examples=36, eval_chunk_examples=36, mode="stratified")
        _, chunked = self.evaluate(examples=36, eval_chunk_examples=12, mode="stratified")
        single_counts = {r["target_token_id"]: r["positive_examples"] for r in single}
        chunked_counts = {r["target_token_id"]: r["positive_examples"] for r in chunked}
        self.assertEqual(single_counts, chunked_counts)

    def test_single_chunk_random_matches_direct_dataset_metrics(self) -> None:
        # The single-chunk random evaluation path must reproduce the metrics obtained by
        # running the same model directly on the canonical unchunked dataset. This pins the
        # Stage 4A metric layer (label mapping and multi-class accumulation); the underlying
        # dataset equivalence itself is guaranteed by the Stage 3 dataset tests.
        model = self.make_model()
        shared = dict(
            length=10,
            target_position_mode="fixed_start",
            target_token_count=3,
            non_target_token_count=1,
            non_target_sampling="uniform",
        )
        examples = 48
        seed = 4242

        # Reference: one-shot dataset, one-shot forward, hand-computed metrics.
        inputs, presence, _, target_ids = make_two_token_dataset(
            examples=examples, seed=seed, **shared
        )
        labels = class_labels_from_presence(presence, target_ids, 3)
        with torch.no_grad():
            ref_preds = model(inputs).argmax(dim=1)
        ref_accuracy = ref_preds.eq(labels).float().mean().item()
        positive_mask = presence.eq(1)
        ref_positive_correct = (
            ref_preds[positive_mask].eq(labels[positive_mask]).float().mean().item()
        )

        # Single-chunk random evaluation through the Stage 4A evaluator.
        row, type_rows = evaluate_length(
            model,
            examples=examples,
            eval_chunk_examples=examples,
            eval_sampling_mode="random",
            batch_size=5,
            seed=seed,
            device=torch.device("cpu"),
            **shared,
        )

        self.assertAlmostEqual(row["accuracy"], ref_accuracy, places=6)
        self.assertAlmostEqual(row["positive_correct_fraction"], ref_positive_correct, places=6)

        self.assertGreater(len(type_rows), 0)
        for type_row in type_rows:
            token_id = type_row["target_token_id"]
            type_mask = positive_mask & target_ids.eq(token_id)
            ref_recall = ref_preds[type_mask].eq(token_id).float().mean().item()
            self.assertEqual(type_row["positive_examples"], int(type_mask.sum().item()))
            self.assertAlmostEqual(type_row["recall"], ref_recall, places=6)


class Stage4ADefaultRunTest(unittest.TestCase):
    """End-to-end smoke at the unit-test scale."""

    def test_tiny_train_and_eval_runs(self) -> None:
        config = Stage4AConfig(
            seed=42,
            device="cpu",
            alpha_mode="learned_log",
            train_length=10,
            target_token_count=3,
            non_target_token_count=1,
            train_examples=64,
            val_examples=32,
            test_examples=24,
            eval_chunk_examples=12,
            eval_sampling_mode="stratified",
            eval_lengths=(10, 20),
            batch_size=16,
            eval_batch_size=8,
            epochs=2,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            model, updates = train_model(config, device=torch.device("cpu"), output_dir=Path(tmp))
        self.assertGreater(updates, 0)
        row, type_rows = evaluate_length(
            model,
            length=20,
            examples=config.test_examples,
            eval_chunk_examples=config.eval_chunk_examples,
            eval_sampling_mode=config.eval_sampling_mode,
            batch_size=config.eval_batch_size,
            seed=config.seed + 10_000 + 20,
            target_position_mode=config.target_position_mode,
            target_token_count=config.target_token_count,
            non_target_token_count=config.non_target_token_count,
            non_target_sampling=config.non_target_sampling,
            device=torch.device("cpu"),
        )
        self.assertTrue(0.0 <= row["accuracy"] <= 1.0)
        self.assertEqual(len(type_rows), 3)


if __name__ == "__main__":
    unittest.main()
