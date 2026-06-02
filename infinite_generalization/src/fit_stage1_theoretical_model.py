"""Fit reduced theoretical formulas to Stage 1 numerical analysis outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("runs/stage1_transformer_maxpool/numerical_analysis"),
        help="Directory containing Stage 1 numerical analysis CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for fitted formula outputs. Defaults to <analysis-dir>/theoretical_fit.",
    )
    parser.add_argument(
        "--attention-query",
        default="mean_over_queries",
        help="Query row from attention_length_summary.csv used for attention fitting.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""

    if not path.exists():
        raise FileNotFoundError(f"missing input CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV, preserving first-seen field order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def target_attention_mass(sequence_length: float, target_score_margin: float) -> float:
    """Compute the reduced target attention mass formula."""

    return 1.0 / (1.0 + (sequence_length - 1.0) * math.exp(-target_score_margin))


def mean(values: list[float]) -> float:
    """Return the arithmetic mean."""

    return sum(values) / len(values)


def regression_metrics(observed: list[float], predicted: list[float]) -> dict[str, float]:
    """Compute simple regression fit metrics."""

    residuals = [actual - estimate for actual, estimate in zip(observed, predicted)]
    absolute_errors = [abs(value) for value in residuals]
    squared_errors = [value * value for value in residuals]
    observed_mean = mean(observed)
    total_sum_of_squares = sum((value - observed_mean) ** 2 for value in observed)
    residual_sum_of_squares = sum(squared_errors)
    r_squared = (
        1.0 - residual_sum_of_squares / total_sum_of_squares
        if total_sum_of_squares > 0.0
        else float("nan")
    )
    return {
        "mean_absolute_error": mean(absolute_errors),
        "root_mean_squared_error": math.sqrt(mean(squared_errors)),
        "residual_sum_of_squares": residual_sum_of_squares,
        "r_squared": r_squared,
    }


def fit_attention_margin(lengths: list[float], observed_attention: list[float]) -> float:
    """Fit the one-parameter attention dilution formula by golden-section search."""

    def objective(target_score_margin: float) -> float:
        predictions = [
            target_attention_mass(length, target_score_margin) for length in lengths
        ]
        return sum(
            (observed - predicted) ** 2
            for observed, predicted in zip(observed_attention, predictions)
        )

    lower = -20.0
    upper = 20.0
    golden_ratio_conjugate = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - golden_ratio_conjugate * (upper - lower)
    right = lower + golden_ratio_conjugate * (upper - lower)
    left_value = objective(left)
    right_value = objective(right)

    for _ in range(200):
        if left_value > right_value:
            lower = left
            left = right
            left_value = right_value
            right = lower + golden_ratio_conjugate * (upper - lower)
            right_value = objective(right)
        else:
            upper = right
            right = left
            right_value = left_value
            left = upper - golden_ratio_conjugate * (upper - lower)
            left_value = objective(left)

    return (lower + upper) / 2.0


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense linear system by Gaussian elimination."""

    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]

    for pivot_index in range(size):
        best_row = max(
            range(pivot_index, size),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        if abs(augmented[best_row][pivot_index]) < 1e-12:
            raise ValueError("linear system is singular")
        if best_row != pivot_index:
            augmented[pivot_index], augmented[best_row] = (
                augmented[best_row],
                augmented[pivot_index],
            )

        pivot = augmented[pivot_index][pivot_index]
        for column_index in range(pivot_index, size + 1):
            augmented[pivot_index][column_index] /= pivot

        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            for column_index in range(pivot_index, size + 1):
                augmented[row_index][column_index] -= factor * augmented[pivot_index][column_index]

    return [augmented[row_index][size] for row_index in range(size)]


def fit_linear_model(features: list[list[float]], targets: list[float]) -> list[float]:
    """Fit ordinary least squares coefficients for a small feature matrix."""

    feature_count = len(features[0])
    normal_matrix = [
        [
            sum(row[left] * row[right] for row in features)
            for right in range(feature_count)
        ]
        for left in range(feature_count)
    ]
    normal_vector = [
        sum(row[index] * target for row, target in zip(features, targets))
        for index in range(feature_count)
    ]
    return solve_linear_system(normal_matrix, normal_vector)


def predict_linear_model(features: list[list[float]], coefficients: list[float]) -> list[float]:
    """Return predictions for a fitted linear model."""

    return [
        sum(value * coefficient for value, coefficient in zip(row, coefficients))
        for row in features
    ]


def log_growth(sequence_length: float) -> float:
    """Return logarithmic non-target interference growth."""

    return math.log(sequence_length)


def sqrt_log_growth(sequence_length: float) -> float:
    """Return extreme-value-style non-target interference growth."""

    return math.sqrt(2.0 * math.log(sequence_length))


def load_fit_data(
    analysis_dir: Path,
    *,
    attention_query: str,
) -> tuple[list[float], list[float], list[float]]:
    """Load lengths, observed attention, and observed positive logits."""

    attention_rows = read_csv(analysis_dir / "attention_length_summary.csv")
    controlled_rows = read_csv(analysis_dir / "controlled_examples.csv")

    attention_by_length: dict[int, float] = {}
    for row in attention_rows:
        if row.get("query") != attention_query:
            continue
        length = int(row["length"])
        attention_by_length[length] = float(row["target_attention_mean"])

    positive_logit_by_length: dict[int, float] = {}
    for row in controlled_rows:
        if row.get("label_type") != "positive":
            continue
        length = int(row["length"])
        positive_logit_by_length[length] = float(row["logit"])

    shared_lengths = sorted(set(attention_by_length) & set(positive_logit_by_length))
    if len(shared_lengths) < 3:
        raise ValueError(
            "need at least three shared lengths in attention_length_summary.csv "
            "and controlled_examples.csv"
        )

    return (
        [float(length) for length in shared_lengths],
        [attention_by_length[length] for length in shared_lengths],
        [positive_logit_by_length[length] for length in shared_lengths],
    )


def load_maxpool_fit_data(analysis_dir: Path) -> tuple[list[float], list[float], list[float]]:
    """Load measured target and non-target max-pool source contributions."""

    maxpool_rows = read_csv(analysis_dir / "maxpool_source_summary.csv")
    if len(maxpool_rows) < 3:
        raise ValueError("need at least three rows in maxpool_source_summary.csv")

    maxpool_rows.sort(key=lambda row: int(row["length"]))
    return (
        [float(row["length"]) for row in maxpool_rows],
        [float(row["target_sourced_contribution_sum"]) for row in maxpool_rows],
        [float(row["non_target_sourced_contribution_sum"]) for row in maxpool_rows],
    )


def logit_model_specs(
    *,
    observed_attention: list[float],
    fitted_attention: list[float],
) -> list[dict[str, object]]:
    """Return the candidate final-logit formulas to fit."""

    return [
        {
            "model_name": "observed_attention_only",
            "attention_values": observed_attention,
            "growth_name": "none",
            "growth_function": None,
        },
        {
            "model_name": "fitted_attention_only",
            "attention_values": fitted_attention,
            "growth_name": "none",
            "growth_function": None,
        },
        {
            "model_name": "observed_attention_plus_log_length",
            "attention_values": observed_attention,
            "growth_name": "log_length",
            "growth_function": log_growth,
        },
        {
            "model_name": "fitted_attention_plus_log_length",
            "attention_values": fitted_attention,
            "growth_name": "log_length",
            "growth_function": log_growth,
        },
        {
            "model_name": "observed_attention_plus_sqrt_2_log_length",
            "attention_values": observed_attention,
            "growth_name": "sqrt_2_log_length",
            "growth_function": sqrt_log_growth,
        },
        {
            "model_name": "fitted_attention_plus_sqrt_2_log_length",
            "attention_values": fitted_attention,
            "growth_name": "sqrt_2_log_length",
            "growth_function": sqrt_log_growth,
        },
    ]


def fit_logit_models(
    *,
    lengths: list[float],
    observed_attention: list[float],
    fitted_attention: list[float],
    observed_logits: list[float],
    target_score_margin: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Fit candidate reduced formulas for the positive final logit."""

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for spec in logit_model_specs(
        observed_attention=observed_attention,
        fitted_attention=fitted_attention,
    ):
        growth_function = spec["growth_function"]
        attention_values = spec["attention_values"]
        if growth_function is None:
            features = [[1.0, attention] for attention in attention_values]
            coefficients = fit_linear_model(features, observed_logits)
            classifier_bias, target_signal_strength = coefficients
            non_target_interference_strength = 0.0
            growth_values = [0.0 for _ in lengths]
        else:
            growth_values = [growth_function(length) for length in lengths]
            # The negative sign makes a positive coefficient mean stronger interference.
            features = [
                [1.0, attention, -growth]
                for attention, growth in zip(attention_values, growth_values)
            ]
            coefficients = fit_linear_model(features, observed_logits)
            classifier_bias, target_signal_strength, non_target_interference_strength = (
                coefficients
            )

        predicted_logits = predict_linear_model(features, coefficients)
        metrics = regression_metrics(observed_logits, predicted_logits)
        summary_rows.append(
            {
                "model_name": spec["model_name"],
                "attention_source": (
                    "observed" if str(spec["model_name"]).startswith("observed") else "fitted"
                ),
                "growth_name": spec["growth_name"],
                "target_score_margin": target_score_margin,
                "classifier_bias": classifier_bias,
                "target_signal_strength": target_signal_strength,
                "non_target_interference_strength": non_target_interference_strength,
                **metrics,
            }
        )

        for length, observed_attention_value, fitted_attention_value, attention_value, growth_value, observed_logit, predicted_logit in zip(
            lengths,
            observed_attention,
            fitted_attention,
            attention_values,
            growth_values,
            observed_logits,
            predicted_logits,
        ):
            prediction_rows.append(
                {
                    "model_name": spec["model_name"],
                    "length": int(length),
                    "observed_attention": observed_attention_value,
                    "fitted_attention": fitted_attention_value,
                    "attention_used_by_model": attention_value,
                    "interference_growth": growth_value,
                    "observed_logit": observed_logit,
                    "predicted_logit": predicted_logit,
                    "residual": observed_logit - predicted_logit,
                }
            )

    return summary_rows, prediction_rows


def fit_maxpool_contribution_models(
    *,
    lengths: list[float],
    target_contributions: list[float],
    non_target_contributions: list[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Fit reduced formulas to measured max-pool source contributions."""

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    model_specs = [
        {
            "contribution_name": "target_sourced_contribution_sum",
            "model_name": "target_constant",
            "growth_name": "none",
            "growth_function": None,
            "targets": target_contributions,
        },
        {
            "contribution_name": "non_target_sourced_contribution_sum",
            "model_name": "non_target_constant",
            "growth_name": "none",
            "growth_function": None,
            "targets": non_target_contributions,
        },
        {
            "contribution_name": "non_target_sourced_contribution_sum",
            "model_name": "non_target_log_length",
            "growth_name": "log_length",
            "growth_function": log_growth,
            "targets": non_target_contributions,
        },
        {
            "contribution_name": "non_target_sourced_contribution_sum",
            "model_name": "non_target_sqrt_2_log_length",
            "growth_name": "sqrt_2_log_length",
            "growth_function": sqrt_log_growth,
            "targets": non_target_contributions,
        },
    ]

    for spec in model_specs:
        growth_function = spec["growth_function"]
        observed_values = spec["targets"]
        if growth_function is None:
            growth_values = [0.0 for _ in lengths]
            features = [[1.0] for _ in lengths]
            coefficients = fit_linear_model(features, observed_values)
            intercept = coefficients[0]
            growth_strength = 0.0
        else:
            growth_values = [growth_function(length) for length in lengths]
            # The negative sign makes a positive coefficient mean length-growing suppression.
            features = [[1.0, -growth] for growth in growth_values]
            coefficients = fit_linear_model(features, observed_values)
            intercept, growth_strength = coefficients

        predicted_values = predict_linear_model(features, coefficients)
        metrics = regression_metrics(observed_values, predicted_values)
        summary_rows.append(
            {
                "contribution_name": spec["contribution_name"],
                "model_name": spec["model_name"],
                "growth_name": spec["growth_name"],
                "intercept": intercept,
                "growth_strength": growth_strength,
                **metrics,
            }
        )

        for length, growth_value, observed_value, predicted_value in zip(
            lengths,
            growth_values,
            observed_values,
            predicted_values,
        ):
            prediction_rows.append(
                {
                    "contribution_name": spec["contribution_name"],
                    "model_name": spec["model_name"],
                    "length": int(length),
                    "interference_growth": growth_value,
                    "observed_contribution": observed_value,
                    "predicted_contribution": predicted_value,
                    "residual": observed_value - predicted_value,
                }
            )

    return summary_rows, prediction_rows


def write_markdown_summary(
    path: Path,
    *,
    attention_summary: dict[str, float],
    logit_summary_rows: list[dict[str, object]],
    maxpool_summary_rows: list[dict[str, object]],
) -> None:
    """Write a compact markdown report for the formula fits."""

    best_logit_row = min(
        logit_summary_rows,
        key=lambda row: float(row["root_mean_squared_error"]),
    )
    non_target_rows = [
        row
        for row in maxpool_summary_rows
        if row["contribution_name"] == "non_target_sourced_contribution_sum"
    ]
    best_non_target_row = min(
        non_target_rows,
        key=lambda row: float(row["root_mean_squared_error"]),
    )
    target_constant_row = next(
        row for row in maxpool_summary_rows if row["model_name"] == "target_constant"
    )
    lines = [
        "# Stage 1 Reduced Formula Fit",
        "",
        "## Attention Fit",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| target_score_margin | {attention_summary['target_score_margin']:.6f} |",
        f"| r_squared | {attention_summary['r_squared']:.6f} |",
        f"| mean_absolute_error | {attention_summary['mean_absolute_error']:.6f} |",
        f"| root_mean_squared_error | {attention_summary['root_mean_squared_error']:.6f} |",
        "",
        "## Best Logit Fit",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| model_name | {best_logit_row['model_name']} |",
        f"| r_squared | {float(best_logit_row['r_squared']):.6f} |",
        f"| mean_absolute_error | {float(best_logit_row['mean_absolute_error']):.6f} |",
        f"| root_mean_squared_error | {float(best_logit_row['root_mean_squared_error']):.6f} |",
        f"| classifier_bias | {float(best_logit_row['classifier_bias']):.6f} |",
        f"| target_signal_strength | {float(best_logit_row['target_signal_strength']):.6f} |",
        f"| non_target_interference_strength | {float(best_logit_row['non_target_interference_strength']):.6f} |",
        "",
        "## Max-Pool Contribution Fit",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| target_constant_r_squared | {float(target_constant_row['r_squared']):.6f} |",
        f"| target_constant_rmse | {float(target_constant_row['root_mean_squared_error']):.6f} |",
        f"| best_non_target_model | {best_non_target_row['model_name']} |",
        f"| best_non_target_r_squared | {float(best_non_target_row['r_squared']):.6f} |",
        f"| best_non_target_rmse | {float(best_non_target_row['root_mean_squared_error']):.6f} |",
        f"| best_non_target_growth_strength | {float(best_non_target_row['growth_strength']):.6f} |",
        "",
        "See `attention_fit_by_length.csv`, `logit_fit_summary.csv`, "
        "`logit_fit_by_length.csv`, `maxpool_contribution_fit_summary.csv`, and "
        "`maxpool_contribution_fit_by_length.csv` for full results.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_write_figures(
    output_dir: Path,
    *,
    attention_rows: list[dict[str, object]],
    logit_prediction_rows: list[dict[str, object]],
    maxpool_prediction_rows: list[dict[str, object]],
) -> None:
    """Write PNG figures when matplotlib is available."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    lengths = [int(row["length"]) for row in attention_rows]
    observed_attention = [float(row["observed_attention"]) for row in attention_rows]
    predicted_attention = [float(row["predicted_attention"]) for row in attention_rows]

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    ax.plot(lengths, observed_attention, marker="o", label="observed")
    ax.plot(lengths, predicted_attention, marker="o", label="fitted formula")
    ax.set_xscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("target attention mass")
    ax.set_title("Target attention formula fit")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "target_attention_fit.png", bbox_inches="tight")
    plt.close(fig)

    best_model_name = min(
        {row["model_name"] for row in logit_prediction_rows},
        key=lambda model_name: sum(
            float(row["residual"]) ** 2
            for row in logit_prediction_rows
            if row["model_name"] == model_name
        ),
    )
    best_rows = [
        row for row in logit_prediction_rows if row["model_name"] == best_model_name
    ]
    best_rows.sort(key=lambda row: int(row["length"]))
    lengths = [int(row["length"]) for row in best_rows]
    observed_logits = [float(row["observed_logit"]) for row in best_rows]
    predicted_logits = [float(row["predicted_logit"]) for row in best_rows]

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    ax.plot(lengths, observed_logits, marker="o", label="observed")
    ax.plot(lengths, predicted_logits, marker="o", label="fitted formula")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("positive logit")
    ax.set_title(f"Best logit formula fit: {best_model_name}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "positive_logit_fit.png", bbox_inches="tight")
    plt.close(fig)

    target_rows = [
        row for row in maxpool_prediction_rows if row["model_name"] == "target_constant"
    ]
    target_rows.sort(key=lambda row: int(row["length"]))
    lengths = [int(row["length"]) for row in target_rows]
    observed_values = [float(row["observed_contribution"]) for row in target_rows]
    predicted_values = [float(row["predicted_contribution"]) for row in target_rows]

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    ax.plot(lengths, observed_values, marker="o", label="observed")
    ax.plot(lengths, predicted_values, marker="o", label="constant fit")
    ax.set_xscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("target-sourced contribution")
    ax.set_title("Target max-pool contribution fit")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "maxpool_target_contribution_fit.png", bbox_inches="tight")
    plt.close(fig)

    non_target_model_name = min(
        {
            row["model_name"]
            for row in maxpool_prediction_rows
            if row["contribution_name"] == "non_target_sourced_contribution_sum"
        },
        key=lambda model_name: sum(
            float(row["residual"]) ** 2
            for row in maxpool_prediction_rows
            if row["model_name"] == model_name
        ),
    )
    non_target_rows = [
        row for row in maxpool_prediction_rows if row["model_name"] == non_target_model_name
    ]
    non_target_rows.sort(key=lambda row: int(row["length"]))
    lengths = [int(row["length"]) for row in non_target_rows]
    observed_values = [float(row["observed_contribution"]) for row in non_target_rows]
    predicted_values = [float(row["predicted_contribution"]) for row in non_target_rows]

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    ax.plot(lengths, observed_values, marker="o", label="observed")
    ax.plot(lengths, predicted_values, marker="o", label="fitted formula")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("non-target-sourced contribution")
    ax.set_title(f"Non-target max-pool contribution fit: {non_target_model_name}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "maxpool_non_target_contribution_fit.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Fit reduced theoretical formulas to existing Stage 1 analysis CSVs."""

    args = parse_args()
    output_dir = args.output_dir or (args.analysis_dir / "theoretical_fit")
    output_dir.mkdir(parents=True, exist_ok=True)

    lengths, observed_attention, observed_logits = load_fit_data(
        args.analysis_dir,
        attention_query=args.attention_query,
    )
    target_score_margin = fit_attention_margin(lengths, observed_attention)
    fitted_attention = [
        target_attention_mass(length, target_score_margin) for length in lengths
    ]
    attention_metrics = regression_metrics(observed_attention, fitted_attention)
    attention_summary = {
        "target_score_margin": target_score_margin,
        **attention_metrics,
    }

    attention_rows = [
        {
            "length": int(length),
            "observed_attention": observed,
            "predicted_attention": predicted,
            "residual": observed - predicted,
            "target_score_margin": target_score_margin,
        }
        for length, observed, predicted in zip(
            lengths,
            observed_attention,
            fitted_attention,
        )
    ]

    logit_summary_rows, logit_prediction_rows = fit_logit_models(
        lengths=lengths,
        observed_attention=observed_attention,
        fitted_attention=fitted_attention,
        observed_logits=observed_logits,
        target_score_margin=target_score_margin,
    )
    (
        maxpool_lengths,
        target_contributions,
        non_target_contributions,
    ) = load_maxpool_fit_data(args.analysis_dir)
    maxpool_summary_rows, maxpool_prediction_rows = fit_maxpool_contribution_models(
        lengths=maxpool_lengths,
        target_contributions=target_contributions,
        non_target_contributions=non_target_contributions,
    )

    write_csv(output_dir / "attention_fit_summary.csv", [attention_summary])
    write_csv(output_dir / "attention_fit_by_length.csv", attention_rows)
    write_csv(output_dir / "logit_fit_summary.csv", logit_summary_rows)
    write_csv(output_dir / "logit_fit_by_length.csv", logit_prediction_rows)
    write_csv(output_dir / "maxpool_contribution_fit_summary.csv", maxpool_summary_rows)
    write_csv(output_dir / "maxpool_contribution_fit_by_length.csv", maxpool_prediction_rows)
    write_markdown_summary(
        output_dir / "summary.md",
        attention_summary=attention_summary,
        logit_summary_rows=logit_summary_rows,
        maxpool_summary_rows=maxpool_summary_rows,
    )
    maybe_write_figures(
        output_dir,
        attention_rows=attention_rows,
        logit_prediction_rows=logit_prediction_rows,
        maxpool_prediction_rows=maxpool_prediction_rows,
    )

    print(f"Saved Stage 1 theoretical formula fits to {output_dir}")


if __name__ == "__main__":
    main()
