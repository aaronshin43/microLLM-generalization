"""Generate poster-specific Stage 3 results figures.

The script reads the five-seed Stage 3 evaluation CSVs and writes two figures:

1. Target attention versus sequence length for four representative runs.
2. The learned-log growth rate across the four training checkpoints.

Each figure is saved as a vector PDF for the poster and as a high-resolution
PNG for quick visual review.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "runs" / "stage3_seeds"
DEFAULT_OUT = PROJECT / "documents" / "poster" / "figures"
SEEDS = (0, 1, 2, 3, 4)

TEXT = "#202124"
MUTED = "#5F6368"
GRID = "#DADCE0"
NAVY = "#19324D"
CONSTANT = "#4D4D4D"
LOG = "#0072B2"
LEARNED_BELOW = "#C47A35"
LEARNED_ABOVE = "#A64D79"
THRESHOLD = "#D55E00"


matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
    }
)


@dataclass(frozen=True)
class AttentionRun:
    directory: str
    label: str
    color: str
    linestyle: Any
    marker: str
    linewidth: float = 3.0
    markevery: tuple[int, int] | None = None


ATTENTION_RUNS = (
    AttentionRun("constant_e50", "Constant (50)", CONSTANT, "-", "o"),
    AttentionRun(
        "learned_log_e50",
        "Learned log (50), $c\\Delta=0.58$",
        LEARNED_BELOW,
        (0, (5, 3)),
        "v",
    ),
    AttentionRun(
        "learned_log_e200",
        "Learned log (200), $c\\Delta=1.14$",
        LEARNED_ABOVE,
        "-",
        "^",
        4.4,
        (1, 2),
    ),
    # Draw Log last with an alternating marker cadence so that both successful
    # curves remain visible where they overlap near target attention one.
    AttentionRun(
        "log_e50",
        "Log (50)",
        LOG,
        (0, (5, 2, 1, 2)),
        "D",
        2.7,
        (0, 2),
    ),
)

LEARNED_CHECKPOINTS = (
    (50, "learned_log_e50"),
    (100, "learned_log_e100"),
    (200, "learned_log_e200"),
    (400, "learned_log_e400"),
)


def mean_and_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def load_attention(run: str) -> dict[int, list[float]]:
    per_length: dict[int, list[float]] = {}
    for seed in SEEDS:
        path = RUNS / f"{run}_s{seed}" / "metrics_by_length.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation metrics: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                length = int(float(row["length"]))
                attention = float(row["mean_empirical_target_attention"])
                per_length.setdefault(length, []).append(attention)
    return per_length


def attention_series(
    per_length: dict[int, list[float]],
) -> tuple[list[int], list[float], list[float]]:
    lengths = sorted(per_length)
    summaries = [mean_and_std(per_length[length]) for length in lengths]
    means = [mean for mean, _ in summaries]
    stds = [std for _, std in summaries]
    return lengths, means, stds


def load_seed_scalar(run: str, field: str) -> list[float]:
    values: list[float] = []
    for seed in SEEDS:
        path = RUNS / f"{run}_s{seed}" / "metrics_by_length.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation metrics: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No evaluation rows in {path}")
        seed_values = {float(row[field]) for row in rows}
        if len(seed_values) != 1:
            raise ValueError(f"Expected one {field} value across lengths in {path}")
        values.append(seed_values.pop())
    return values


def style_axis(ax: plt.Axes, *, horizontal_grid_only: bool = False) -> None:
    ax.set_axisbelow(True)
    if horizontal_grid_only:
        ax.grid(axis="y", which="major", color=GRID, linewidth=1.0)
    else:
        ax.grid(which="major", color=GRID, linewidth=1.0)
    ax.grid(which="minor", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", labelsize=14, width=1.1, length=5)
    ax.tick_params(axis="x", which="minor", width=0.8, length=3)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


def plot_target_attention(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 7.4))
    handles = []

    for spec in ATTENTION_RUNS:
        lengths, means, stds = attention_series(load_attention(spec.directory))
        lower = [max(0.0, mean - std) for mean, std in zip(means, stds)]
        upper = [min(1.0, mean + std) for mean, std in zip(means, stds)]
        ax.fill_between(
            lengths,
            lower,
            upper,
            color=spec.color,
            alpha=0.13,
            linewidth=0,
            zorder=1,
        )
        (line,) = ax.plot(
            lengths,
            means,
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
            marker=spec.marker,
            markersize=8.5,
            markeredgecolor="white",
            markeredgewidth=1.1,
            markevery=spec.markevery,
            zorder=3,
        )
        handles.append(line)

    ax.set_xscale("log")
    ax.set_xlim(8, 1.2e7)
    ax.set_ylim(-0.025, 1.035)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("Sequence length $n$", fontsize=18, labelpad=10)
    ax.set_ylabel("Target attention $p_t(n)$", fontsize=18, labelpad=10)
    style_axis(ax)

    # The poster card heading carries the message, so the figure itself has no
    # title; the former callout lives inside the empty central plot region.
    ax.annotate(
        "",
        xy=(4.0e6, 0.955),
        xytext=(1.3e6, 0.575),
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.7, shrinkA=4, shrinkB=2),
        zorder=2,
    )
    ax.annotate(
        "",
        xy=(4.0e6, 0.055),
        xytext=(1.3e6, 0.33),
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.7, shrinkA=4, shrinkB=2),
        zorder=2,
    )
    ax.text(
        5.0e5,
        0.505,
        "Same margin, opposite outcomes",
        fontsize=16.5,
        fontweight="semibold",
        color=NAVY,
        ha="center",
        va="center",
    )
    ax.text(
        5.0e5,
        0.455,
        "Constant (50) vs Learned log (200)\n"
        "both learn $\\Delta\\approx9$\n"
        "positive accuracy at $n=10^7$: 0% vs 100%",
        fontsize=13.2,
        color=TEXT,
        ha="center",
        va="top",
        linespacing=1.5,
    )

    legend_order = (0, 3, 1, 2)
    fig.legend(
        [handles[index] for index in legend_order],
        [ATTENTION_RUNS[index].label for index in legend_order],
        loc="upper center",
        bbox_to_anchor=(0.545, 1.0),
        ncol=4,
        frameon=False,
        fontsize=12.6,
        handlelength=2.5,
        handletextpad=0.5,
        columnspacing=1.25,
    )

    # Keep the poster footprint unchanged while giving the data region more
    # vertical space by tightening the internal top and bottom margins.
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.125, top=0.91)
    save_figure(fig, output_dir, "poster_target_attention_by_length")


def plot_learned_threshold(output_dir: Path) -> None:
    epochs: list[int] = []
    means: list[float] = []
    stds: list[float] = []

    for epoch, run in LEARNED_CHECKPOINTS:
        values = load_seed_scalar(run, "learned_log_c_delta_min_mean")
        mean, std = mean_and_std(values)
        epochs.append(epoch)
        means.append(mean)
        stds.append(std)
        print(f"{run}: c*Delta = {mean:.3f} +/- {std:.3f}")

    positions = list(range(len(epochs)))
    fig, ax = plt.subplots(figsize=(7.0, 6.2))

    ax.axhspan(0.4, 1.0, color="#FBF6EC", zorder=0)
    ax.axhspan(1.0, 1.72, color="#F0F7F5", zorder=0)
    ax.axhline(1.0, color=THRESHOLD, linewidth=2.0, linestyle=(0, (5, 3)), zorder=2)
    ax.plot(positions, means, color=LEARNED_ABOVE, linewidth=2.6, zorder=3)
    ax.errorbar(
        positions,
        means,
        yerr=stds,
        fmt="o",
        markersize=9.5,
        color=LEARNED_ABOVE,
        markerfacecolor=LEARNED_ABOVE,
        markeredgecolor="white",
        markeredgewidth=1.2,
        ecolor=LEARNED_ABOVE,
        elinewidth=2.0,
        capsize=5,
        capthick=2.0,
        zorder=4,
    )

    for position, mean, std in zip(positions, means, stds):
        ax.text(
            position,
            mean + std + 0.055,
            f"{mean:.2f}",
            fontsize=14,
            fontweight="semibold",
            ha="center",
            va="bottom",
            color=TEXT,
        )

    ax.text(
        0.98,
        1.015,
        "$c\\Delta=1$ threshold",
        transform=ax.get_yaxis_transform(),
        fontsize=12.5,
        color=THRESHOLD,
        ha="right",
        va="bottom",
    )
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(0.4, 1.72)
    ax.set_xticks(positions, [str(epoch) for epoch in epochs])
    ax.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("Training checkpoint (epochs)", fontsize=17, labelpad=10)
    ax.set_ylabel("Learned growth rate $c\\Delta$", fontsize=17, labelpad=10)
    style_axis(ax, horizontal_grid_only=True)

    # The benchmark fact lives inside the otherwise empty below-threshold band
    # so the poster callout can stay a single interpretive sentence.
    ax.text(
        1.5,
        0.475,
        "every checkpoint reaches 100% accuracy at $n=10^7$",
        fontsize=13.2,
        color=MUTED,
        ha="center",
        va="center",
    )

    fig.subplots_adjust(left=0.16, right=0.975, bottom=0.14, top=0.965)
    save_figure(fig, output_dir, "poster_learned_log_threshold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_target_attention(args.output_dir)
    plot_learned_threshold(args.output_dir)


if __name__ == "__main__":
    main()
