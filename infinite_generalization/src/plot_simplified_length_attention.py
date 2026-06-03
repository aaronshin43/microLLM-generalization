"""Plot target attention mass in the simplified length-aware attention model."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


AlphaSchedule = Callable[[int], float]
MarginSchedule = Callable[[int], float]


@dataclass(frozen=True)
class CurveSpec:
    """Description of one curve in the simplified attention plot."""

    name: str
    effective_margin: MarginSchedule


def repository_dir() -> Path:
    """Return the infinite_generalization project directory."""

    return Path(__file__).resolve().parents[1]


def target_attention_from_margin(length: int, effective_margin: float) -> float:
    """Compute p_t(n) from the effective target-vs-non-target margin."""

    if length < 2:
        raise ValueError("length must be at least 2")

    # Use the logistic form to avoid unnecessary overflow for large margins.
    log_non_target_ratio = math.log(length - 1) - effective_margin
    if log_non_target_ratio > 700:
        return 0.0
    if log_non_target_ratio < -700:
        return 1.0
    return 1.0 / (1.0 + math.exp(log_non_target_ratio))


def target_attention(length: int, delta: float, alpha: float) -> float:
    """Compute p_t(n) for a global inverse temperature alpha."""

    return target_attention_from_margin(length, alpha * delta)


def build_lengths(min_length: int, max_length: int, points: int) -> list[int]:
    """Build unique integer lengths on a logarithmic grid."""

    if min_length < 2:
        raise ValueError("min_length must be at least 2")
    if max_length < min_length:
        raise ValueError("max_length must be greater than or equal to min_length")
    if points < 2:
        raise ValueError("points must be at least 2")

    start = math.log(min_length)
    stop = math.log(max_length)
    lengths = {
        int(round(math.exp(start + (stop - start) * index / (points - 1))))
        for index in range(points)
    }
    lengths.update({min_length, max_length})
    return sorted(lengths)


def build_curve_specs() -> list[CurveSpec]:
    """Define representative regimes for the simplified model."""

    return [
        CurveSpec(
            name="constant alpha, Delta=2",
            effective_margin=lambda n: 2.0,
        ),
        CurveSpec(
            name="log n, Delta=0.5",
            effective_margin=lambda n: 0.5 * math.log(n),
        ),
        CurveSpec(
            name="log n, Delta=1",
            effective_margin=lambda n: math.log(n),
        ),
        CurveSpec(
            name="log n, Delta=2",
            effective_margin=lambda n: 2.0 * math.log(n),
        ),
        CurveSpec(
            name="c log n, c Delta=0.75",
            effective_margin=lambda n: 0.75 * math.log(n),
        ),
        CurveSpec(
            name="c log n, c Delta=1.25",
            effective_margin=lambda n: 1.25 * math.log(n),
        ),
    ]


def write_csv(output_csv: Path, lengths: list[int], curves: list[CurveSpec]) -> None:
    """Write the plotted curve values to CSV for auditability."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "schedule",
                "length",
                "effective_margin",
                "target_attention_mass",
            ],
        )
        writer.writeheader()
        for curve in curves:
            for length in lengths:
                margin = curve.effective_margin(length)
                writer.writerow(
                    {
                        "schedule": curve.name,
                        "length": length,
                        "effective_margin": f"{margin:.12g}",
                        "target_attention_mass": (
                            f"{target_attention_from_margin(length, margin):.12g}"
                        ),
                    }
                )


def write_plot(output_png: Path, lengths: list[int], curves: list[CurveSpec]) -> bool:
    """Write a PNG plot if matplotlib is installed."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))

    for curve in curves:
        values = [
            target_attention_from_margin(length, curve.effective_margin(length))
            for length in lengths
        ]
        ax.plot(lengths, values, label=curve.name, linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel("Sequence length n")
    ax.set_ylabel("Target attention mass p_t(n)")
    ax.set_title("Simplified length-aware attention regimes")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    fig.savefig(output_png, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    project_dir = repository_dir()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1_000_000)
    parser.add_argument("--points", type=int, default=240)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_dir / "runs" / "simplified_length_aware_attention" / "attention_curves.csv",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=project_dir
        / "documents"
        / "figures"
        / "simplified_length_aware_attention_pt.png",
    )
    return parser.parse_args()


def main() -> None:
    """Generate CSV and optional PNG outputs for the simplified model."""

    args = parse_args()
    lengths = build_lengths(args.min_length, args.max_length, args.points)
    curves = build_curve_specs()

    write_csv(args.output_csv, lengths, curves)
    wrote_plot = write_plot(args.output_png, lengths, curves)

    print(f"Wrote CSV: {args.output_csv}")
    if wrote_plot:
        print(f"Wrote PNG: {args.output_png}")
    else:
        print("matplotlib is not installed; skipped PNG generation.")


if __name__ == "__main__":
    main()
