"""Regenerate the main FINAL_REPORT results figure.

The two panels show mean curves over seeds 0-4 with a +/- standard-deviation
band at the seven decade lengths (no 5e6). The script writes a vector PDF for
LaTeX and a PNG preview for Markdown.
"""

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

BASE = Path("D:/03_Coding/microLLM-generalization/infinite_generalization")
RUNS = BASE / "runs" / "stage3_seeds"
OUT = BASE / "documents" / "latex"
SEEDS = [0, 1, 2, 3, 4]

# Keep each run's color, line style, and marker identical across both panels.
# The palette is colorblind-safe, while line styles and markers preserve run
# identity in grayscale. Display labels describe the experiment rather than the
# internal run-directory names.
RUN_SPECS = [
    ("constant_e50", "Constant (50)", "#0072B2", "-", "o"),
    ("constant_e100", "Constant (100)", "#E69F00", "-", "s"),
    ("constant_e1000", "Constant (1000)", "#009E73", "-", "^"),
    ("log_e50", "Log (50)", "#D55E00", "-.", "D"),
    ("learned_log_e50", "Learned log (50)", "#CC79A7", "--", "v"),
    ("learned_log_e200", "Learned log (200)", "#56B4E9", (0, (3, 2)), "X"),
]


def load_run(run: str) -> dict[int, dict[str, list[float]]]:
    per_len: dict[int, dict[str, list[float]]] = {}
    for seed in SEEDS:
        path = RUNS / f"{run}_s{seed}" / "metrics_by_length.csv"
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                length = int(float(row["length"]))
                bucket = per_len.setdefault(length, {"att": [], "logit": []})
                bucket["att"].append(float(row["mean_empirical_target_attention"]))
                bucket["logit"].append(float(row["mean_logit_positive"]))
    return per_len


def series(per_len, key):
    lengths = sorted(per_len)
    means = [statistics.mean(per_len[L][key]) for L in lengths]
    stds = [statistics.stdev(per_len[L][key]) if len(per_len[L][key]) > 1 else 0.0 for L in lengths]
    return lengths, means, stds


DATA = {run: load_run(run) for run, *_ in RUN_SPECS}


def plot_panel(ax, key, ylabel, title, *, clip01=False, zero_line=False):
    for run, display_label, color, linestyle, marker in RUN_SPECS:
        lengths, means, stds = series(DATA[run], key)
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        if clip01:
            lo = [max(0.0, v) for v in lo]
            hi = [min(1.0, v) for v in hi]
        ax.plot(
            lengths,
            means,
            marker=marker,
            color=color,
            label=display_label,
            linewidth=1.25,
            markersize=3.4,
            markeredgecolor="white",
            markeredgewidth=0.35,
            linestyle=linestyle,
        )
        ax.fill_between(lengths, lo, hi, color=color, alpha=0.12, linewidth=0)
    if zero_line:
        ax.axhline(0.0, color="#404040", linewidth=0.85, linestyle=(0, (4, 2)))
    ax.set_xscale("log")
    ax.set_ylabel(ylabel, fontsize=8.2)
    ax.set_title(title, fontsize=8.4, pad=5)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#D8D8D8", linewidth=0.55)
    ax.grid(False, which="minor")
    ax.tick_params(axis="both", which="major", labelsize=7, width=0.7, length=3)
    ax.tick_params(axis="x", which="minor", width=0.55, length=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    if clip01:
        ax.set_ylim(-0.03, 1.03)


def plot_combined(*, preview=False):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.9), sharex=True)
    plot_panel(
        axes[0],
        "att",
        "$p_t(n)$",
        "(a) Target attention",
        clip01=True,
    )
    plot_panel(
        axes[1],
        "logit",
        "$z(n)$",
        "(b) Positive-example logit",
        zero_line=True,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.supxlabel("Sequence length $n$", fontsize=8.2, y=0.155)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        fontsize=6.5,
        frameon=False,
        handlelength=2.6,
        handletextpad=0.55,
        columnspacing=1.15,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.89, bottom=0.315, wspace=0.24)

    suffix = "_preview" if preview else ""
    stem = f"final_report_attention_and_logit_by_length{suffix}"
    pdf_path = OUT / f"{stem}.pdf"
    png_path = OUT / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Write separate preview files without replacing report figures.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_combined(preview=args.preview)
