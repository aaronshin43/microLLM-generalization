"""Regenerate the main FINAL_REPORT results figure.

The two panels show mean curves over seeds 0-4 with a +/- standard-deviation
band at the seven decade lengths (no 5e6). The script writes a vector PDF for
LaTeX and a PNG preview for Markdown.
"""

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

# (run label, color, linestyle) -- keep run->style identical across both panels.
# learned_log_e200 is dashed so it stays visible where it coincides with log_e50
# at p_t = 1 in the target-attention panel.
RUN_SPECS = [
    ("constant_e50", "C0", "-"),
    ("constant_e100", "C1", "-"),
    ("constant_e1000", "C2", "-"),
    ("log_e50", "C3", "-"),
    ("learned_log_e50", "C4", "-"),
    ("learned_log_e200", "C9", (0, (3, 2))),
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
    for run, color, linestyle in RUN_SPECS:
        lengths, means, stds = series(DATA[run], key)
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        if clip01:
            lo = [max(0.0, v) for v in lo]
            hi = [min(1.0, v) for v in hi]
        ax.plot(lengths, means, marker="o", color=color, label=run,
                linewidth=1.15, markersize=3.1, linestyle=linestyle)
        ax.fill_between(lengths, lo, hi, color=color, alpha=0.15, linewidth=0)
    if zero_line:
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)
    if clip01:
        ax.set_ylim(-0.03, 1.03)


def plot_combined():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.05), sharex=True)
    plot_panel(
        axes[0],
        "att",
        "Target attention mass $p_t$",
        "(a) Target attention mass",
        clip01=True,
    )
    plot_panel(
        axes[1],
        "logit",
        "Positive-example logit $z$",
        "(b) Positive-example logit",
        zero_line=True,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.supxlabel("Sequence length $n$", fontsize=8.5, y=0.145)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=6.3,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.095, right=0.99, top=0.88, bottom=0.31, wspace=0.34)

    stem = "final_report_attention_and_logit_by_length"
    pdf_path = OUT / f"{stem}.pdf"
    png_path = OUT / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


plot_combined()
