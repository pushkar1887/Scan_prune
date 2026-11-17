"""Fig. 8: OmniMedVQA accuracy relative to Full by visual-token count."""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from _figure_common import DARK, FULL, MID, MODEL_NAMES, RED, clean_axis, plt, read_json, save, warn

LINES = {
    "ScanPrune": dict(color=RED, marker="*", ms=5.5, ls="-", lw=0.9, mec="black", mew=0.35, zorder=5),
    "ScanPrune-eq": dict(color=RED, marker="*", ms=5.5, ls="--", lw=0.85, mfc="white", mew=0.65, zorder=6),
    "VScan": dict(color=DARK, marker="D", ms=3, ls="-", lw=0.8, zorder=4),
    "VisionZip": dict(color=MID, marker="o", ms=3.2, ls=":", lw=0.85, zorder=3),
}


def bin_key(label):
    return int(label.lstrip("<>=").split("-")[0])


def main():
    data = read_json("sec7_analyses.json").get("omni_by_size", {})
    bins = sorted({label for row in data.values() for label in row}, key=bin_key)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(3.3, 2.05))
    fig.subplots_adjust(left=0.172, right=0.975, top=0.82, bottom=0.36, wspace=0.23)

    for ax, (model, model_name) in zip(axes, MODEL_NAMES.items()):
        row = data.get(model, {})
        if not row:
            warn(f"no OmniMedVQA size bins for {model}")
            ax.set_axis_off()
            continue
        for method, style in LINES.items():
            values = [row.get(label, {}).get(method) for label in bins]
            missing = [label for label, value in zip(bins, values) if value is None]
            if missing:
                warn(f"{model} {method}: no value for {', '.join(missing)}")
            ax.plot(range(len(bins)), [np.nan if v is None else v for v in values], **style)
        ax.axhline(0, c=FULL, ls="--", lw=0.65, zorder=1)
        ax.set_xticks(range(len(bins)), [b.replace("<=", "\u2264") for b in bins],
                      fontsize=6.3, rotation=28, ha="right", rotation_mode="anchor")
        ax.set_title(model_name, fontsize=7, pad=4)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        clean_axis(ax, "y")

    axes[0].set_ylabel("Accuracy vs Full (pp)")
    fig.supxlabel("Visual tokens per image", x=0.57, y=0.145, fontsize=7)
    handles = [Line2D([], [], label=method, **style) for method, style in LINES.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.014), ncol=4, frameon=False,
               handlelength=1.35, handletextpad=0.35, columnspacing=0.65, fontsize=6.2)
    save(fig, "fig8_omni_size")


if __name__ == "__main__":
    main()
