"""Regenerate the FINAL_REPORT figures from the multi-seed Stage 3 sweep.

Mean curve over seeds 0-4 with a +/- std shaded band, at the 7 decade lengths
(no 5e6). Writes vector PDF files for LaTeX and PNG files for Markdown preview.
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
OUT = BASE / "documents" / "figures"
SEEDS = [0, 1, 2, 3, 4]

# (run label, color, linestyle) -- keep run->style identical across both figures.
# learned_log_e200 is dashed so it stays visible where it coincides with log_e50
# at p_t = 1 in the target-attention figure.
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


def plot(key, ylabel, title, stem, *, clip01=False, zero_line=False, legend_loc="best"):
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    for run, color, linestyle in RUN_SPECS:
        lengths, means, stds = series(DATA[run], key)
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        if clip01:
            lo = [max(0.0, v) for v in lo]
            hi = [min(1.0, v) for v in hi]
        ax.plot(lengths, means, marker="o", color=color, label=run,
                linewidth=1.4, markersize=3.8, linestyle=linestyle)
        ax.fill_between(lengths, lo, hi, color=color, alpha=0.15, linewidth=0)
    if zero_line:
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Sequence length $n$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)
    ax.legend(ncol=2, fontsize=7.2, loc=legend_loc)

    pdf_path = OUT / f"{stem}.pdf"
    png_path = OUT / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


plot("att", "Target attention mass $p_t$", "Target attention by sequence length",
     "final_report_target_attention_by_length", clip01=True, legend_loc="lower left")
plot("logit", "Positive-example logit $z$", "Positive-example logit by sequence length",
     "final_report_positive_logit_by_length", zero_line=True, legend_loc="lower left")
