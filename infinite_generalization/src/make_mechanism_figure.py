"""Vector-geometry figure for the Mechanism section.

Plots the learned query/key vectors of learned_log_e200 seed 1 exactly in the
2-D (d=2) plane: q_u, k_t, k_u, and the difference k_t - k_u. Shows that q_u is
nearly collinear with k_t - k_u, that k_t projects positively onto q_u (a > 0)
and k_u negatively (b < 0). Palette matches the report's other figures
(default matplotlib color cycle); identity is carried by legend + position,
not color alone.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

# (label, vector, color) -- fixed categorical order, report default palette
VECS = [
    (r"$q_u$", q_u, "C0"),
    (r"$k_t$", k_t, "C2"),
    (r"$k_u$", k_u, "C1"),
    (r"$k_t-k_u$", diff, "C4"),
]

fig, ax = plt.subplots(figsize=(4.5, 4.5))

# query direction ("score axis"): dashed line through the origin along q_u
qhat = q_u / np.linalg.norm(q_u)
L = 5.6
ax.plot([-L * qhat[0], L * qhat[0]], [-L * qhat[1], L * qhat[1]],
        color="gray", ls="--", lw=1.0, alpha=0.55, zorder=1)

# q_u, k_t and k_t-k_u are nearly collinear, so their shafts overlap. All four
# are drawn at equal width; z-order stacks them and their arrowheads sit at
# different distances (and q_u is a few degrees off the k_t axis), so each
# stays legible.
# (vector, color, zorder)
LW = 2.6
ARROWS = [
    (diff, "C4", 2),   # k_t - k_u (derived): behind
    (k_t,  "C2", 3),   # target key
    (k_u,  "C1", 4),   # non-target key
    (q_u,  "C0", 5),   # query: on top
]
for v, col, zo in ARROWS:
    ax.annotate("", xy=(v[0], v[1]), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=LW,
                                shrinkA=0, shrinkB=0, mutation_scale=16),
                zorder=zo)

# axes through the origin
ax.axhline(0, color="black", lw=0.8, alpha=0.45, zorder=0)
ax.axvline(0, color="black", lw=0.8, alpha=0.45, zorder=0)
ax.scatter([0], [0], color="black", s=14, zorder=5)

ax.set_aspect("equal")
ax.set_xlim(-5.0, 2.8)
ax.set_ylim(-2.3, 3.9)
ax.grid(True, alpha=0.22)
ax.set_xlabel("coordinate 1", fontsize=9)
ax.set_ylabel("coordinate 2", fontsize=9)
ax.set_title("Learned query/key geometry", fontsize=9.5)
ax.tick_params(labelsize=8)

# legend carries identity (arrows bunch, so direct labels would collide)
handles = [Line2D([0], [0], color=col, lw=2.4, label=lab) for lab, _v, col in VECS]
handles.append(Line2D([0], [0], color="gray", ls="--", lw=1.0, label="$q_u$ direction"))
ax.legend(handles=handles, loc="upper right", fontsize=7.8, framealpha=0.9)

pdf_path = OUT2 / "final_report_mechanism_vectors.pdf"
png_path = OUT / "final_report_mechanism_vectors.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("wrote", pdf_path)
print("wrote", png_path)
