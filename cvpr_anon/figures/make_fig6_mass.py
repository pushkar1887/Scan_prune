"""Fig. 6: per-layer attention share with and without proportional attention."""

from itertools import product

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator

from _figure_common import FULL, MID, MODEL_NAMES, RED, clean_axis, plt, read_json, save, warn

CURVES = {
    "full": dict(label="Full", color=FULL, ls="--", lw=0.85),
    "without": dict(label="Merge", color=MID, ls="-", lw=0.8),
    "with": dict(label=r"Merge + $\log s$", color=RED, ls="-", lw=0.9, marker="*", ms=3.6, markevery=5),
}
PREDICTION = dict(label="Prop. 2 prediction", color="#555555", ls=(0, (1.1, 1.8)), lw=0.8)
DATASETS = {"slake_en": "SLAKE", "rin_anat_mc": "RIN anatomy"}


def main():
    data = read_json("sec7_analyses.json").get("mass_preservation", {})
    predictions = read_json("fig6_predictions.json").get("predicted_deep_share", {})
    fig, axes = plt.subplots(2, 2, figsize=(3.3, 2.37), sharey=True)
    fig.subplots_adjust(left=0.175, right=0.988, bottom=0.24, top=0.91, hspace=0.53, wspace=0.24)

    for (i, (model, model_name)), (j, (dataset, dataset_name)) in product(
        enumerate(MODEL_NAMES.items()), enumerate(DATASETS.items())
    ):
        ax = axes[i, j]
        key = f"{model}|{dataset}"
        row = data.get(key, {})
        profiles = {field: {int(layer): value for layer, value in row.get(field, {}).items()} for field in CURVES}
        available = {layer for profile in profiles.values() for layer in profile}
        if not available:
            warn(f"no mass preservation profiles for {key}")
            ax.set_axis_off()
            continue
        final_layer = max(available)
        layers = range(final_layer + 1)
        for field, style in CURVES.items():
            if not profiles[field]:
                warn(f"{key}: no {field} mass preservation curve")
                continue
            shares = [profiles[field].get(layer) for layer in layers]
            invalid = [layer for layer, value in zip(layers, shares) if value is None or not np.isfinite(value) or value <= 0]
            if invalid:
                warn(f"{key}: omitted {field} log-scale values at layers {invalid}")
            y = [100 * value if value is not None and np.isfinite(value) and value > 0 else np.nan for value in shares]
            ax.plot(layers, y, **style)
        predicted = predictions.get(key)
        if predicted is None:
            warn(f"{key}: no Prop. 2 prediction")
        else:
            ax.hlines(100 * predicted, (final_layer + 1) // 2, final_layer,
                      colors=PREDICTION["color"], linestyles=PREDICTION["ls"], lw=PREDICTION["lw"], zorder=2)
        ax.set_xlim(0, final_layer)
        ax.set_title(f"{model_name} · {dataset_name}", fontsize=6.5, pad=2)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        clean_axis(ax, "y")

    fig.supylabel("Image attention share (%)", x=0.035, y=0.575, fontsize=7)
    fig.supxlabel("Decoder layer (0-based)", x=0.58, y=0.115, fontsize=7)
    handles = [Line2D([], [], **style) for style in (*CURVES.values(), PREDICTION)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=4,
               frameon=False, handlelength=1.3, handletextpad=0.35, columnspacing=0.75, fontsize=6.1)
    save(fig, "fig6_mass")


if __name__ == "__main__":
    main()
