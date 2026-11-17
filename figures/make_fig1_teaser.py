"""Fig. 1: actual SLAKE keep-sets and the paired FastV minus Random gaps."""

from textwrap import fill

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from _figure_common import (
    DARK, DATASET_NAMES, LIGHT, MODEL_NAMES, RED, clean_axis, paired_accuracy, plt, save, seed_zero_rows, slake_ct,
    valid_keep, warn,
)
from medprune.adoption import ALIASES, CELLS, SINGLE_TOKEN, prompt_style
from medprune.masks import model_view


def overlay(ax, image, row, color, border):
    h, w = row["grid_hw"]
    ax.imshow(image, extent=(0, w, h, 0), interpolation="nearest", aspect="equal")
    cells = [Rectangle((k % w, k // w), 1, 1) for k in row["keep_idx"]]
    ax.add_collection(PatchCollection(cells, facecolor=color, edgecolor=border, linewidth=0.28, alpha=0.7))
    ax.set_axis_off()


def keep_sets(fig):
    fig.text(0.022, 0.952, "(a) Token keep-sets on one SLAKE CT", va="top", fontweight="bold")
    sample = slake_ct()
    if sample is None:
        return
    image, record = sample
    model, dataset = record["model"], "slake_en"
    view, _ = model_view(image, model)
    fastv = seed_zero_rows(model, dataset, "fastv")
    random = seed_zero_rows(model, dataset, "random_l2")
    _, branches = ALIASES["scanprune"].split(":", 1)
    single, other = branches.split("|")
    routed = {method: seed_zero_rows(model, dataset, method) for method in (single, other)}
    headers = (("FastV", DARK, "#ededed"), ("Random", LIGHT, "#f0f0f0"), ("ScanPrune", RED, "#f6dfda"))
    columns = (0.162, 0.278, 0.394)
    width = 0.103
    for x, (label, color, _) in zip(columns, headers):
        fig.text(x + width / 2, 0.841, label, ha="center", fontweight="bold", color=RED if color == RED else "#222222")

    displayed = []
    for y, qid in zip((0.467, 0.082), record["qids"]):
        question = fastv.get(qid) or random.get(qid)
        if question is None or not question.get("question") or not question.get("answer_type"):
            warn(f"{qid}: no matching FastV or Random row for the SLAKE teaser")
            continue
        route = single if prompt_style(dataset, question["answer_type"]) in SINGLE_TOKEN else other
        methods = (fastv.get(qid), random.get(qid), routed[route].get(qid))
        height = width * fig.get_figwidth() / fig.get_figheight()
        fig.text(0.022, y + height / 2, fill(question["question"], 19), va="center", fontsize=6.4)
        for x, row, (label, color, border) in zip(columns, methods, headers):
            if (row is None or row.get("image_id") != record["image_id"] or row.get("grid_hw") != record["grid_hw"]
                    or "radimagenet" in row["image_id"].lower() or not valid_keep(row)):
                warn(f"{qid}: no matching keep-set and grid for {label}")
                continue
            ax = fig.add_axes((x, y, width, height))
            overlay(ax, np.asarray(view), row, color, border)
            displayed.append((qid, label, len(row["keep_idx"]), row.get("n_vis")))
    print("SLAKE overlays (QID, method, kept, available):", displayed)


def gap_bars(fig):
    ax = fig.add_axes((0.697, 0.192, 0.282, 0.717))
    fig.text(0.547, 0.952, "(b) Accuracy across model–benchmark cells", va="top", fontweight="bold")
    y, ticks, labels, deltas = 0, [], [], []
    for model, name in MODEL_NAMES.items():
        ticks.append(y)
        labels.append(name)
        for key in (key for key in CELLS if key[0] == model):
            y += 1
            ticks.append(y)
            labels.append(DATASET_NAMES[key[1]])
            scores = paired_accuracy(key, "random_l2", "fastv")
            if scores is None:
                continue
            delta = scores[1] - scores[0]
            ax.barh(y, delta, height=0.57, color=DARK if model == "lingshu7b" else "#8f8f8f", zorder=2)
            ax.annotate(f"{delta:+.1f}", (delta, y), xytext=(-3 if delta < 0 else 3, 0), textcoords="offset points",
                        va="center", ha="right" if delta < 0 else "left", fontsize=6.3, color="#292929")
            deltas.append(delta)
        y += 1.32
    ax.set_ylim(ticks[-1] + 0.56, -0.54)
    ax.set_yticks(ticks, labels, fontsize=6.55)
    for label in ax.get_yticklabels():
        if label.get_text() in MODEL_NAMES.values():
            label.set_fontweight("bold")
    low, high = min([0, *deltas]), max([0, *deltas])
    pad = max(2, (high - low) * 0.19)
    ax.set_xlim(low - pad, high + (pad if high > 0 else pad / 4))
    ax.axvline(0, color=DARK, lw=0.65)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.set_xlabel("FastV − Random (pp)", labelpad=2)
    clean_axis(ax, "x")
    ax.tick_params(axis="y", length=0)
    print("FastV - Random by cell (pp):", [round(v, 2) for v in deltas])


def main():
    fig = plt.figure(figsize=(6.9, 2.17))
    keep_sets(fig)
    gap_bars(fig)
    save(fig, "fig1_teaser")


if __name__ == "__main__":
    main()
