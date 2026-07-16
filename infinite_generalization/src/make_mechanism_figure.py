"""Vector-geometry figure for the Mechanism section.

Plots the learned query/key vectors of learned_log_e200 seed 1 exactly in the
2-D (d=2) plane: q_u, k_t, k_u, and the difference k_t - k_u. Shows that q_u is
nearly collinear with k_t - k_u, that k_t projects positively onto q_u (a > 0)
and k_u negatively (b < 0). Palette matches the report's other figures
(default matplotlib color cycle); identity is carried by legend + position,
not color alone.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

OUT = Path("D:/03_Coding/microLLM-generalization/infinite_generalization/documents/figures")
OUT2 = Path("D:/03_Coding/microLLM-generalization/infinite_generalization/documents/latex")

# learned_log_e200, seed 1 (matches the vectors printed in the Mechanism section)
q_u = np.array([-1.830, 1.646])
k_t = np.array([-2.232, 1.642])
k_u = np.array([1.990, -1.428])
diff = k_t - k_u  # (-4.222, 3.070)

# Colorblind-safe palette shared with the results figure.
VECS = [
    (r"$q_u$", q_u, "#0072B2"),
    (r"$k_t$", k_t, "#009E73"),
    (r"$k_u$", k_u, "#D55E00"),
    (r"$k_t-k_u$", diff, "#CC79A7"),
]

def plot_mechanism(*, preview=False):
    fig, ax = plt.subplots(figsize=(3.5, 2.85))

    # Query direction (the score axis): a dashed line through the origin.
    qhat = q_u / np.linalg.norm(q_u)
    line_length = 5.6
    ax.plot(
        [-line_length * qhat[0], line_length * qhat[0]],
        [-line_length * qhat[1], line_length * qhat[1]],
        color="#777777",
        ls=(0, (4, 2)),
        lw=0.85,
        alpha=0.75,
        zorder=1,
    )

    # Draw overlapping shafts from back to front, with compact arrowheads.
    arrows = [
        (diff, "#CC79A7", 2),
        (k_t, "#009E73", 3),
        (k_u, "#D55E00", 4),
        (q_u, "#0072B2", 5),
    ]
    for vector, color, zorder in arrows:
        ax.annotate(
            "",
            xy=(vector[0], vector[1]),
            xytext=(0.0, 0.0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.75,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=12,
            ),
            zorder=zorder,
        )

    # Direct labels sit in open regions and use light leader lines so that the
    # clustered arrowheads remain unobstructed.
    label_specs = [
        (r"$k_t-k_u$", diff, (-4.72, 3.43), "left", "center", "#CC79A7"),
        (r"$k_t$", k_t, (-2.85, 1.03), "center", "center", "#009E73"),
        (r"$q_u$", q_u, (-1.28, 2.08), "center", "center", "#0072B2"),
        (r"$k_u$", k_u, (2.20, -0.82), "center", "center", "#D55E00"),
    ]
    for label, vector, position, horizontal, vertical, color in label_specs:
        ax.annotate(
            label,
            xy=vector,
            xytext=position,
            textcoords="data",
            ha=horizontal,
            va=vertical,
            fontsize=8.2,
            color="#222222",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.15),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                alpha=0.7,
                lw=0.55,
                shrinkA=2,
                shrinkB=5,
            ),
            zorder=6,
        )

    direction_point = np.array([0.82, 0.82 * (qhat[1] / qhat[0])])
    ax.annotate(
        r"$q_u$ direction",
        xy=direction_point,
        xytext=(0.30, -1.42),
        textcoords="data",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#666666",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.15),
        arrowprops=dict(
            arrowstyle="-",
            color="#888888",
            lw=0.5,
            shrinkA=2,
            shrinkB=4,
        ),
        zorder=6,
    )

    # Axes through the origin and light structural grid.
    ax.axhline(0, color="#777777", lw=0.7, zorder=0)
    ax.axvline(0, color="#777777", lw=0.7, zorder=0)
    ax.scatter([0], [0], color="#202020", s=10, zorder=6)
    ax.set_axisbelow(True)
    ax.grid(True, color="#DEDEDE", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-5.0, 2.65)
    ax.set_ylim(-2.2, 3.75)
    ax.set_xlabel(r"$x_1$", fontsize=8.5)
    ax.set_ylabel(r"$x_2$", fontsize=8.5)
    ax.tick_params(axis="both", labelsize=7.2, width=0.65, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.975, bottom=0.16)

    suffix = "_preview" if preview else ""
    stem = f"final_report_mechanism_vectors{suffix}"
    pdf_path = OUT2 / f"{stem}.pdf"
    png_dir = OUT2 if preview else OUT
    png_path = png_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", pdf_path)
    print("wrote", png_path)


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
    plot_mechanism(preview=args.preview)
