"""Run numerical analysis for the Stage 1 transformer failure mechanism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from analysis import (
    attention_length_rows,
    hidden_evidence_rows,
    load_stage1_checkpoint,
    manual_layer0_forward,
    make_controlled_sequence,
    maxpool_and_logit_rows,
    parameter_stat_rows,
    token_qkv_geometry_rows,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Stage 1 numerical analysis."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/stage1_transformer_maxpool"),
        help="Stage 1 run directory containing config.json and model.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for analysis CSVs. Defaults to <run-dir>/numerical_analysis.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[10, 100, 500, 900, 1000, 1100],
        help="Sequence lengths to analyze.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used for checkpoint loading and analysis.",
    )
    return parser.parse_args()


@torch.no_grad()
def controlled_example_rows(
    *,
    model,
    task,
    lengths: list[int],
) -> list[dict[str, object]]:
    """Summarize logits and pooled activations for positive and negative examples."""

    rows: list[dict[str, object]] = []
    for length_index, length in enumerate(lengths):
        for label_type, label in (("negative", 0), ("positive", 1)):
            seed = 50_000 + length_index if label_type == "positive" else 70_000 + length_index
            tokens, target_index = make_controlled_sequence(
                length=length,
                label_type=label_type,
                task=task,
                seed=seed,
            )
            tensors = manual_layer0_forward(model, tokens)
            logit = float(tensors["logit"].cpu().item())
            pooled = tensors["pooled"].detach().cpu()
            rows.append(
                {
                    "length": length,
                    "label_type": label_type,
                    "label": label,
                    "target_index": "" if target_index is None else target_index,
                    "logit": logit,
                    "probability": float(torch.sigmoid(torch.tensor(logit)).item()),
                    "prediction": int(logit >= 0.0),
                    "correct": int((logit >= 0.0) == bool(label)),
                    "pooled_l2_norm": float(torch.linalg.vector_norm(pooled).item()),
                    "pooled_abs_mean": float(pooled.abs().mean().item()),
                    "pooled_dim_min": float(pooled.min().item()),
                    "pooled_dim_max": float(pooled.max().item()),
                }
            )
    return rows


def write_summary(
    path: Path,
    *,
    lengths: list[int],
    controlled_rows: list[dict[str, object]],
    attention_rows: list[dict[str, object]],
    maxpool_rows: list[dict[str, object]],
) -> None:
    """Write a compact markdown summary of the numerical analysis outputs."""

    positive_rows = [row for row in controlled_rows if row["label_type"] == "positive"]
    mean_attention_rows = [row for row in attention_rows if row["query"] == "mean_over_queries"]

    lines = [
        "# Stage 1 Numerical Analysis Summary",
        "",
        f"Analyzed lengths: {', '.join(str(length) for length in lengths)}",
        "",
        "## Positive Logit By Length",
        "",
        "| Length | Logit | Probability | Prediction |",
        "|---:|---:|---:|---:|",
    ]
    for row in positive_rows:
        lines.append(
            "| {length} | {logit:.4f} | {probability:.4f} | {prediction} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Mean Target Attention By Length",
            "",
            "| Length | Target Attention Mean | Target Attention Max | Entropy Mean | Logit |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mean_attention_rows:
        lines.append(
            "| {length} | {target_attention_mean:.4f} | {target_attention_max:.4f} | "
            "{attention_entropy_mean:.4f} | {logit:.4f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Max-Pool Target Source Fraction",
            "",
            "| Length | Target-Sourced Dim Fraction | Target Contribution Sum | Non-Target Contribution Sum |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in maxpool_rows:
        lines.append(
            "| {length} | {target_sourced_dim_fraction:.4f} | "
            "{target_sourced_contribution_sum:.4f} | "
            "{non_target_sourced_contribution_sum:.4f} |".format(**row)
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Generate numerical analysis CSVs for the trained Stage 1 model."""

    args = parse_args()
    device = torch.device(args.device)
    output_dir = args.output_dir or (args.run_dir / "numerical_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, task, config, metadata = load_stage1_checkpoint(args.run_dir, device=device)
    lengths = args.lengths

    parameter_rows = parameter_stat_rows(model)
    qkv_rows = token_qkv_geometry_rows(model, task=task)
    controlled_rows = controlled_example_rows(model=model, task=task, lengths=lengths)
    attention_rows = attention_length_rows(model, task=task, lengths=lengths)
    evidence_rows = hidden_evidence_rows(model, task=task, lengths=lengths)
    maxpool_rows, logit_rows = maxpool_and_logit_rows(model, task=task, lengths=lengths)

    write_csv(output_dir / "parameter_stats.csv", parameter_rows)
    write_csv(output_dir / "token_qkv_geometry.csv", qkv_rows)
    write_csv(output_dir / "controlled_examples.csv", controlled_rows)
    write_csv(output_dir / "attention_length_summary.csv", attention_rows)
    write_csv(output_dir / "hidden_evidence_summary.csv", evidence_rows)
    write_csv(output_dir / "maxpool_source_summary.csv", maxpool_rows)
    write_csv(output_dir / "logit_decomposition.csv", logit_rows)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "lengths": lengths,
                "task": task.to_dict(),
                "stage1": config.to_dict(),
                "source_metadata": metadata,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_summary(
        output_dir / "summary.md",
        lengths=lengths,
        controlled_rows=controlled_rows,
        attention_rows=attention_rows,
        maxpool_rows=maxpool_rows,
    )

    print(f"Saved Stage 1 numerical analysis to {output_dir}")


if __name__ == "__main__":
    main()
