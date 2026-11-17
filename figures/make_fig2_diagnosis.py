"""Fig. 2: FastV's keep locations, question overlap, layer ranks, and cut depth."""

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from _figure_common import DARK, FULL, LIGHT, MID, MODEL_NAMES, RESULTS, clean_axis, paired_accuracy, plt, save, seed_zero_rows, valid_keep, warn

SLAKE = "slake_en"
LINGSHU = "lingshu7b"
DEPTHS = (2, 8, 10, 12, 14, 16, 20)
RANK_FIELDS = {"position": "Position", "image_beyond_position": "Image", "question_beyond_image": "Question"}
RANK_COLORS = ("#4477aa", "#76bfaa", "#f0c575")
METHOD_STYLES = {
    "FastV": dict(color=DARK, marker="s", ms=3, lw=0.8),
    "Random": dict(color=MID, marker="D", ms=3, lw=0.8, mfc="white", mew=0.65, ls="--"),
}


def title(fig, x, text):
    fig.text(x, 0.953, text, va="top", fontweight="bold")


def small_panels(fig, x, width, labels):
    axes = [fig.add_axes((x, y, width, 0.247)) for y in (0.553, 0.230)]
    for ax, label in zip(axes, labels):
        ax.set_title(label, loc="left", fontsize=6.35, pad=2.3)
        ax.tick_params(labelsize=6.1)
        clean_axis(ax, "y")
    axes[0].tick_params(labelbottom=False)
    return axes


def keep_maps(fig, upright):
    title(fig, 0.016, "(a) FastV keep maps")
    flipped = seed_zero_rows(LINGSHU, "slake_en_vflip", "fastv")
    grids = defaultdict(lambda: defaultdict(list))
    rejected = 0
    for qid in upright.keys() & flipped.keys():
        a, b = upright[qid], flipped[qid]
        if (a["image_id"] != b["image_id"] or not valid_keep(a) or not valid_keep(b)
                or a["grid_hw"] != b["grid_hw"]):
            rejected += 1
            continue
        grids[tuple(a["grid_hw"])][a["image_id"]].append((a["keep_idx"], b["keep_idx"]))
    if rejected or upright.keys() != flipped.keys():
        warn(f"upright/flip: {rejected} shared QIDs had mismatched image or grid; QID totals {len(upright)}, {len(flipped)}")
    if not grids:
        warn("no matched FastV upright and flip grids")
        return

    # Choose the grid represented by the most matched images and weight each image equally.
    shape, images = max(grids.items(), key=lambda item: (len(item[1]), item[0]))
    probabilities = []
    for side in (0, 1):
        means = []
        for pairs in images.values():
            indicators = np.zeros((len(pairs), np.prod(shape)))
            for i, pair in enumerate(pairs):
                indicators[i, pair[side]] = 1
            means.append(indicators.mean(axis=0))
        probabilities.append(100 * np.mean(means, axis=0).reshape(shape))

    fig.text(0.122, 0.825, f"Lingshu · {len(images)} images · {shape[0]} × {shape[1]}", ha="center", fontsize=6.15)
    for x, values, label in zip((0.019, 0.126), probabilities, ("Upright", "Vertically flipped")):
        ax = fig.add_axes((x, 0.395, 0.101, 0.323))
        graphic = ax.imshow(values, vmin=0, vmax=100, cmap="YlGnBu", interpolation="nearest")
        ax.set_title(label, fontsize=6.25, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    cax = fig.add_axes((0.042, 0.309, 0.162, 0.030))
    bar = fig.colorbar(graphic, cax=cax, orientation="horizontal", ticks=(0, 50, 100))
    bar.ax.tick_params(labelsize=6, length=1.8, pad=1.2)
    bar.set_label("Keep probability (%)", fontsize=6.4, labelpad=1.5)
    print(f"upright/flip native-grid comparison: {len(images)} SLAKE images with grid {shape}")


def within_image_jaccard(rows, model, method):
    images = defaultdict(dict)
    rejected = 0
    for row in rows.values():
        if not valid_keep(row):
            rejected += 1
            continue
        images[(row["image_id"], tuple(row["grid_hw"]))][row["question"].strip().casefold()] = set(row["keep_idx"])
    if rejected:
        warn(f"{model} {method}: omitted {rejected} missing or invalid keep-sets")
    return {
        key: float(np.mean([len(a & b) / len(a | b) for a, b in combinations(sets.values(), 2)]))
        for key, sets in images.items() if len(sets) >= 2
    }


def overlap_panel(fig, upright):
    title(fig, 0.257, "(b) Same-image overlap")
    axes = small_panels(fig, 0.285, 0.172, MODEL_NAMES.values())
    edges = np.linspace(0, 1, 16)
    for ax, model in zip(axes, MODEL_NAMES):
        fastv = upright if model == LINGSHU else seed_zero_rows(model, SLAKE, "fastv")
        runs = {"FastV": fastv, "Random": seed_zero_rows(model, SLAKE, "random_l2")}
        overlaps = {name: within_image_jaccard(rows, model, name) for name, rows in runs.items()}
        keys = sorted(overlaps["FastV"].keys() & overlaps["Random"].keys())
        if not keys:
            warn(f"no shared SLAKE images with multiple questions for {model}")
            continue
        for name, color in (("Random", LIGHT), ("FastV", DARK)):
            values = [overlaps[name][key] for key in keys]
            counts = np.histogram(values, bins=edges)[0]
            ax.stairs(100 * counts / len(values), edges, fill=True, color=color, alpha=0.83, linewidth=0.5)
            print(f"same-image Jaccard {model} {name}: {np.mean(values):.4f}, {len(values)} images")
        ax.set(xlim=(0, 1), ylim=(0, 104), yticks=(0, 50, 100), xticks=(0, 0.5, 1))
    axes[-1].set_xlabel("Mean Jaccard per image", labelpad=1.5, fontsize=6.4)
    fig.text(0.242, 0.52, "Images (%)", rotation=90, ha="center", va="center", fontsize=6.3)
    fig.legend(handles=[Patch(color=DARK, label="FastV"), Patch(color=LIGHT, label="Random")],
               loc="lower center", bbox_to_anchor=(0.371, 0.002), ncol=2, frameon=False,
               handlelength=1.1, handletextpad=0.32, columnspacing=0.75, fontsize=6.2)


def variance_panel(fig):
    title(fig, 0.501, "(c) Rank variance on SLAKE")
    axes = small_panels(fig, 0.526, 0.166, MODEL_NAMES.values())
    path = RESULTS / "tables" / "layer_profile.csv"
    if not path.is_file():
        warn(f"missing {path}")
        return
    table = pd.read_csv(path)
    for ax, model in zip(axes, MODEL_NAMES):
        rows = table[(table["model"] == model) & (table["dataset"] == SLAKE) & (table["readout"] == "last")]
        if rows.empty:
            warn(f"no last-prompt-token SLAKE layer profile for {model}")
            continue
        rows = rows.set_index("layer").sort_index()
        rows = rows.reindex(range(int(rows.index.max()) + 1))
        values = rows[list(RANK_FIELDS)].to_numpy(float).T
        if not np.isfinite(values).all():
            warn(f"{model}: omitted missing variance shares at layers {rows.index[~np.isfinite(values).all(axis=0)].tolist()}")
        ax.stackplot(rows.index, 100 * values, colors=RANK_COLORS, linewidth=0)
        ax.set(xlim=(0, rows.index.max()), ylim=(0, 100), yticks=(0, 50, 100))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    axes[-1].set_xlabel("Decoder layer (0-based)", labelpad=1.5, fontsize=6.4)
    fig.text(0.484, 0.52, "Variance (%)", rotation=90, ha="center", va="center", fontsize=6.3)
    fig.legend(handles=[Patch(facecolor=color, edgecolor=MID, lw=0.3, label=label)
                        for color, label in zip(RANK_COLORS, RANK_FIELDS.values())],
               loc="lower center", bbox_to_anchor=(0.605, 0.002), ncol=3, frameon=False,
               handlelength=0.95, handletextpad=0.29, columnspacing=0.64, fontsize=6.0)


def depth_panel(fig):
    title(fig, 0.751, "(d) Cut depth on Lingshu")
    tasks = {"rin_anat_mc": "RIN anatomy", "rin_abn_closed": "RIN abnormality"}
    axes = small_panels(fig, 0.776, 0.205, tasks.values())
    for ax, dataset in zip(axes, tasks):
        deltas = []
        for name, style in METHOD_STYLES.items():
            values = []
            for k in DEPTHS:
                method = ("fastv" if k == DEPTHS[0] else f"fastv_k{k}") if name == "FastV" else f"random_l{k}"
                scores = paired_accuracy((LINGSHU, dataset), "random_l2", method)
                values.append(scores[1] - scores[0] if scores is not None else np.nan)
            ax.plot(DEPTHS, values, **style)
            deltas.extend(v for v in values if np.isfinite(v))
            print(f"depth {dataset} {name} (K, pp):", [(k, round(v, 2)) for k, v in zip(DEPTHS, values) if np.isfinite(v)])
        ax.axhline(0, c=FULL, ls="--", lw=0.65, zorder=1)
        ax.set(xlim=(DEPTHS[0] - 1, DEPTHS[-1] + 1), xticks=DEPTHS)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        low, high = min([0, *deltas]), max([0, *deltas])
        pad = max(1, (high - low) * 0.10)
        ax.set_ylim(low - pad, high + pad)
    axes[-1].set_xlabel("Cut layer $K$", labelpad=1.5, fontsize=6.4)
    fig.text(0.725, 0.52, "Accuracy vs Random at $K{=}2$ (pp)", rotation=90, ha="center", va="center", fontsize=6.2)
    fig.legend(handles=[Line2D([], [], label=name, **style) for name, style in METHOD_STYLES.items()],
               loc="lower center", bbox_to_anchor=(0.875, 0.002), ncol=2, frameon=False,
               handlelength=1.1, handletextpad=0.32, columnspacing=0.75, fontsize=6.2)


def main():
    fig = plt.figure(figsize=(6.9, 2.17))
    upright = seed_zero_rows(LINGSHU, SLAKE, "fastv")
    keep_maps(fig, upright)
    overlap_panel(fig, upright)
    variance_panel(fig)
    depth_panel(fig)
    save(fig, "fig2_diagnosis")


if __name__ == "__main__":
    main()
