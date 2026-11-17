"""Fig. 4: accuracy (mean over the nine cells) vs measured visual tokens per decoder layer.

Numbers are Table 3 of section6.tex (status file Nov 11 basis; tokens/layer = mean of the Lingshu and Qwen3-VL
columns). Run: python figures/make_fig4_budget.py  ->  fig4_budget.pdf / .png next to this file.
"""
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FULL = 71.6
# name: (tokens per layer %, All, group)
ROWS = {
    "VisionZip": (9.55, 64.4, "pre"), "PruneSID": (9.80, 63.7, "pre"), "CDPruner": (9.55, 63.4, "pre"),
    "VisPruner": (9.55, 61.3, "pre"), "HiPrune": (9.55, 59.8, "pre"), "DivPrune": (9.55, 59.0, "pre"),
    "Block pooling": (9.55, 63.5, "pre"),
    "Random": (15.30, 62.5, "ref"), "FastV": (15.30, 43.2, "llm"), "VScan": (15.65, 66.8, "llm"),
    "FEATHER": (41.95, 63.3, "large"), "TransPrune": (39.65, 63.3, "large"),
    "ScanPrune": (9.45, 67.5, "ours"), "ScanPrune-eq": (15.75, 68.6, "ours"),
}
STYLE = {
    "pre": dict(marker="o", s=22, c="#7a7a7a", zorder=3),
    "llm": dict(marker="s", s=22, c="#4d4d4d", zorder=3),
    "ref": dict(marker="D", s=20, c="white", edgecolors="#4d4d4d", linewidths=0.9, zorder=3),
    "large": dict(marker="^", s=24, c="#b5b5b5", zorder=3),
    "ours": dict(marker="*", s=120, c="#c0392b", edgecolors="black", linewidths=0.4, zorder=5),
}
LABELS = {  # text offset in points
    "VisionZip": (-6, 3, "right"), "VScan": (6, -9, "left"), "FastV": (6, 0, "left"), "Random": (6, -3, "left"),
    "FEATHER": (0, 6, "center"), "TransPrune": (0, -10, "center"),
    "ScanPrune": (-7, 3, "right"), "ScanPrune-eq": (7, 2, "left"),
}

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.3, 2.35))
ax.axhline(FULL, ls="--", lw=0.7, c="#2c3e50", zorder=1)
ax.text(45, FULL + 0.6, "Full (71.6)", ha="right", va="bottom", fontsize=6.5, color="#2c3e50")
for group in ("large", "pre", "llm", "ref", "ours"):
    pts = [(x, y) for n, (x, y, g) in ROWS.items() if g == group]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], **STYLE[group])
for name, (dx, dy, ha) in LABELS.items():
    x, y, g = ROWS[name]
    bold = g == "ours"
    ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                fontsize=7 if bold else 6.5, fontweight="bold" if bold else "normal",
                color="#c0392b" if bold else "#333333")
ax.annotate("other pre-LLM\npruners", (9.55, 60.4), xytext=(-8, 0), textcoords="offset points", ha="right",
            va="center", fontsize=6, color="#7a7a7a")
ax.set_xscale("log")
ax.set_xlim(6.5, 60)
ax.set_xticks([8, 10, 15, 20, 30, 50])
ax.set_xticklabels(["8", "10", "15", "20", "30", "50"])
ax.minorticks_off()
ax.set_ylim(41, 74)
ax.set_xlabel("Visual tokens per decoder layer (% of N)")
ax.set_ylabel("Accuracy, mean of nine cells (%)")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", lw=0.3, alpha=0.5)
fig.tight_layout(pad=0.3)
fig.savefig(HERE / "fig4_budget.pdf")
fig.savefig(HERE / "fig4_budget.png", dpi=300)
print("saved", HERE / "fig4_budget.pdf")
