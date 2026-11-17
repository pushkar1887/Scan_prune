"""Fig. 5: retained visual gain relative to Full and text-only."""

import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from _figure_common import (
    DARK, FULL, LIGHT, MID, MODEL_NAMES, RED, RESULTS, accuracy, cell, clean_axis, plt, save, verdicts, warn,
)
from medprune.adoption import BALANCED, CELLS, balanced
from medprune.analysis import parse_run_id, rescore_open
from medprune.judge import apply_judge

ROWS = {
    "FastV": "fastv",
    "Random": "random_l2",
    "VisionZip": "visionzip",
    "VScan": "vscan",
    "ScanPrune": "scanprune",
    "ScanPrune-eq": "scanprune_eq",
}


def text_only(model, dataset):
    # The reference rvg.py scores text-only at the row level and reads its r100 runs.
    rows = []
    for path in sorted((RESULTS / "runs").glob(f"{model}__{dataset}__text_only__*__t4x2.jsonl")):
        info = parse_run_id(path.stem)
        with path.open(encoding="utf-8") as handle:
            rows.extend({**json.loads(line), **info} for line in handle if line.strip())
    if not rows:
        raise FileNotFoundError(f"no text-only run for {model} {dataset}")
    frame = apply_judge(rescore_open(pd.DataFrame(rows)), verdicts())
    if dataset == "rin_open":
        frame = frame[frame["qid"].str.endswith("anatomy_open")]
    if (model, dataset) in BALANCED:
        gold = frame["answer"].str.strip().str.lower().str.startswith("yes").to_numpy(bool)
        yes = frame["response"].fillna("").str.strip().str.lower().str.match(r"\W*yes").to_numpy(float)
        return balanced(np.ones(len(frame)), gold, yes)
    return 100 * float(frame["correct"].mean())


def retained_gain():
    out = {}
    for key in CELLS:
        full = cell(*key, "full")
        if full is None:
            continue
        try:
            text_score = text_only(*key)
        except FileNotFoundError as error:
            warn(str(error))
            continue
        full_score = accuracy(full, key)
        if not np.isfinite([text_score, full_score]).all() or full_score == text_score:
            warn(f"{key}: retained visual gain has no finite, nonzero denominator")
            continue
        row = {"text": round(text_score, 1), "full": round(full_score, 1)}
        for label, method in ROWS.items():
            frame = cell(*key, method)
            if frame is None:
                continue
            score = accuracy(frame, key)
            if not np.isfinite(score):
                warn(f"{key}: {label} has no finite accuracy")
                continue
            row[label] = round(100 * (score - text_score) / (full_score - text_score), 1)
        out[key] = row
    return out


def mean_gain(data, label, keys=CELLS):
    values = [data.get(key, {}).get(label) for key in keys]
    if any(value is None for value in values):
        warn(f"{label}: incomplete retained visual gain over {len(keys)} cells")
        return np.nan
    return float(np.mean(values))


def main():
    data = retained_gain()
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    fig.subplots_adjust(left=0.17, right=0.982, bottom=0.26, top=0.87)
    positions = np.arange(len(ROWS))
    means = {label: mean_gain(data, label) for label in ROWS}
    models = {
        model: {label: mean_gain(data, label, [key for key in CELLS if key[0] == model]) for label in ROWS}
        for model in MODEL_NAMES
    }

    # In this normalized metric the Full model defines 100%.
    ax.axhline(100, c=FULL, ls="--", lw=0.7, zorder=1)
    for x, (label, color) in enumerate(zip(ROWS, (DARK, LIGHT, "#969696", MID, RED, RED))):
        mean = means[label]
        if np.isfinite(mean):
            ax.bar(x, mean, width=0.63, color=color, alpha=0.66 if color == RED else 0.86, zorder=2)
            if color == RED:
                ax.scatter(x, mean, s=29, marker="*", color=RED, edgecolors="black", linewidths=0.35, zorder=5)
            ax.text(x, 0.49 * mean, f"{mean:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if label == "FastV" else "#222222", fontweight="bold", zorder=3)
        for j, model in enumerate(MODEL_NAMES):
            value = models[model][label]
            if np.isfinite(value):
                ax.scatter(x + (-0.16, 0.16)[j], value, s=12, marker="o", zorder=4,
                           facecolors=("white", "#272727")[j], edgecolors="#272727", linewidths=0.55)

    values = [100, 0, *means.values(), *(v for row in models.values() for v in row.values())]
    finite = np.asarray(values)[np.isfinite(values)]
    ax.set_ylim(min(0, finite.min() - 4), finite.max() + 10)
    ax.set_xlim(-0.55, len(ROWS) - 0.45)
    ax.set_xticks(positions, list(ROWS), rotation=31, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Retained visual gain (%)")
    clean_axis(ax, "y")
    handles = [
        Line2D([], [], linestyle="none", marker="o", ms=3.4, mfc="white", mec="#272727", label="Lingshu"),
        Line2D([], [], linestyle="none", marker="o", ms=3.4, color="#272727", label="Qwen3-VL"),
        Line2D([], [], ls="--", lw=0.7, color=FULL, label="Full"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, handlelength=1.25, columnspacing=1.1)
    print("nine-cell means:", {key: round(value, 1) for key, value in means.items()})
    print("per-model means:", {MODEL_NAMES[key]: {k: round(v, 1) for k, v in row.items()} for key, row in models.items()})
    save(fig, "fig5_rvg")


if __name__ == "__main__":
    main()
