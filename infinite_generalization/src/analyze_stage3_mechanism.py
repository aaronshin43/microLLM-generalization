"""Analyze the weight-level mechanism of a trained Stage 3 model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from stage3_simplified_attention import (
    NON_TARGET_TOKEN_ID,
    TARGET_TOKEN_ID,
    SimplifiedLastQueryAttentionClassifier,
)


def project_dir() -> Path:
    """Return the infinite_generalization project directory."""

    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Stage 3 run directory containing model.pt and metrics_by_length.csv.",
    )
    parser.add_argument(
        "--output-json",
        default="mechanism_analysis.json",
        help="Output JSON filename, relative to --run-dir unless absolute.",
    )
    parser.add_argument(
        "--output-csv",
        default="mechanism_decomposition.csv",
        help="Output CSV filename, relative to --run-dir unless absolute.",
    )
    return parser.parse_args()


def resolve_run_dir(value: str) -> Path:
    """Resolve a run directory path."""

    path = Path(value)
    if path.is_absolute():
        return path
    project_relative = project_dir() / path
    if project_relative.exists():
        return project_relative
    return Path.cwd() / path


def resolve_output_path(run_dir: Path, value: str) -> Path:
    """Resolve an output path relative to the run directory."""

    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a Stage 3 model checkpoint."""

    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model(checkpoint: dict[str, Any]) -> SimplifiedLastQueryAttentionClassifier:
    """Reconstruct the Stage 3 model from checkpoint metadata."""

    model = SimplifiedLastQueryAttentionClassifier(
        d_head=int(checkpoint["d_head"]),
        alpha_mode=str(checkpoint["alpha_mode"]),
        alpha_log_scale_init=float(checkpoint["alpha_log_scale_init"]),
        target_token_count=int(checkpoint.get("target_token_count", 1)),
        non_target_token_count=int(checkpoint.get("non_target_token_count", 1)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def read_metric_delta(run_dir: Path) -> float | None:
    """Read the recorded mean delta from metrics_by_length.csv when available."""

    path = run_dir / "metrics_by_length.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        return None
    train_length_rows = [row for row in rows if int(row["length"]) == 10]
    row = train_length_rows[0] if train_length_rows else rows[0]
    return float(row["mean_delta"])


def analyze_model(model: SimplifiedLastQueryAttentionClassifier) -> dict[str, Any]:
    """Compute the direct query/key mechanism for target-vs-non-target margin."""

    w_q = model.query_projection.weight.detach().cpu()
    w_k = model.key_projection.weight.detach().cpu()
    d_head = int(model.d_head)
    scale = math.sqrt(d_head)

    x_t = torch.zeros(model.score_vocab_size)
    x_u = torch.zeros(model.score_vocab_size)
    x_t[TARGET_TOKEN_ID] = 1.0
    x_u[int(checkpoint_non_target_token_id(model))] = 1.0
    q_u = w_q @ x_u
    k_t = w_k @ x_t
    k_u = w_k @ x_u
    key_difference = k_t - k_u

    target_products = q_u * k_t
    non_target_products = q_u * k_u
    margin_products = q_u * key_difference
    target_contributions = target_products / scale
    non_target_contributions = non_target_products / scale
    delta_contributions = margin_products / scale

    a = target_contributions.sum().item()
    b = non_target_contributions.sum().item()
    delta = delta_contributions.sum().item()

    return {
        "d_head": d_head,
        "scale": scale,
        "w_q": w_q.tolist(),
        "w_k": w_k.tolist(),
        "q_u": q_u.tolist(),
        "k_t": k_t.tolist(),
        "k_u": k_u.tolist(),
        "k_t_minus_k_u": key_difference.tolist(),
        "q_u_times_k_t": target_products.tolist(),
        "q_u_times_k_u": non_target_products.tolist(),
        "q_u_times_k_t_minus_k_u": margin_products.tolist(),
        "target_score_a": a,
        "non_target_score_b": b,
        "delta": delta,
        "delta_contributions": delta_contributions.tolist(),
        "target_score_contributions": target_contributions.tolist(),
        "non_target_score_contributions": non_target_contributions.tolist(),
    }


def build_decomposition_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Build per-dimension decomposition rows."""

    rows = []
    for index, q_value in enumerate(analysis["q_u"]):
        rows.append(
            {
                "dimension": index,
                "q_u": q_value,
                "k_t": analysis["k_t"][index],
                "k_u": analysis["k_u"][index],
                "k_t_minus_k_u": analysis["k_t_minus_k_u"][index],
                "q_u_times_k_t": analysis["q_u_times_k_t"][index],
                "q_u_times_k_u": analysis["q_u_times_k_u"][index],
                "q_u_times_k_t_minus_k_u": analysis["q_u_times_k_t_minus_k_u"][index],
                "target_score_contribution": analysis["target_score_contributions"][index],
                "non_target_score_contribution": analysis["non_target_score_contributions"][
                    index
                ],
                "delta_contribution": analysis["delta_contributions"][index],
            }
        )
    return rows


def checkpoint_non_target_token_id(model: SimplifiedLastQueryAttentionClassifier) -> int:
    """Return the first non-target token id for a reconstructed Stage 3 model."""

    return int(model.target_token_count)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the mechanism analysis."""

    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    checkpoint = load_checkpoint(run_dir / "model.pt")
    model = load_model(checkpoint)
    analysis = analyze_model(model)
    metric_delta = read_metric_delta(run_dir)

    summary = {
        "run_dir": str(run_dir),
        "target_token_id": TARGET_TOKEN_ID,
        "target_token_count": checkpoint.get("target_token_count", 1),
        "non_target_token_id": checkpoint.get("non_target_token_id", NON_TARGET_TOKEN_ID),
        "non_target_token_count": checkpoint.get("non_target_token_count", 1),
        "alpha_mode": checkpoint["alpha_mode"],
        "optimizer_updates": checkpoint.get("optimizer_updates"),
        "train_lengths": checkpoint.get("train_lengths"),
        **analysis,
        "recorded_metric_delta": metric_delta,
        "delta_abs_error_vs_metric": (
            None if metric_delta is None else abs(float(analysis["delta"]) - metric_delta)
        ),
    }
    rows = build_decomposition_rows(analysis)

    output_json = resolve_output_path(run_dir, args.output_json)
    output_csv = resolve_output_path(run_dir, args.output_csv)
    write_json(output_json, summary)
    write_csv(output_csv, rows)

    print(f"Wrote mechanism JSON to: {output_json}")
    print(f"Wrote mechanism CSV to: {output_csv}")
    print(
        "a={:.6f}, b={:.6f}, delta={:.6f}, metric_delta={}".format(
            analysis["target_score_a"],
            analysis["non_target_score_b"],
            analysis["delta"],
            "n/a" if metric_delta is None else f"{metric_delta:.6f}",
        )
    )


if __name__ == "__main__":
    main()
