"""Fig. 7: scheduling by answer format and transfer to published pruners."""

import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from _figure_common import DARK, DATASET_NAMES, LIGHT, MODEL_NAMES, clean_axis, paired_accuracy, plt, read_json, save, warn
from medprune.adoption import CELLS

FORMATS = {
    "slake_en": "Mixed",
    "rin_anat_mc": "Single-token",
    "rin_abn_closed": "Single-token",
    "omni_anat": "Single-token",
    "rin_open": "Generated",
}
BLUE = "#4477aa"
TEAL = "#76bfaa"
GOLD = "#f0c575"
FORMAT_STYLE = {
    "Single-token": dict(facecolor=BLUE, edgecolor="#34567d"),
    "Generated": dict(facecolor=GOLD, edgecolor="#896324", hatch="///"),
    "Mixed": dict(facecolor=TEAL, edgecolor="#42796b"),
}
COMPONENT_COLORS = {"gate": BLUE, "merge": TEAL}
PAIRS = {
    "gate | FastV": ("fastv", "gate_fastv"),
    "gate | PDrop": ("pdrop", "gate_pdrop"),
    "gate | DivPrune": ("divprune", "gate_divprune"),
    "gate | CDPruner": ("cdpruner_neg", "gate_cdpruner_neg"),
    "gate | VisionZip": ("visionzip", "gate_visionzip"),
    "merge | FastV keep-set": ("fastv_prellm", "fastv_merge"),
    "merge | Random (pre-LLM)": ("random_l0", "random_merge"),
    "merge | VisionZip": ("visionzip_nomerge", "visionzip"),
    "merge | VScan": ("vscan_nomerge", "vscan"),
}
SHORT_DATASET = {**DATASET_NAMES, "rin_abn_closed": "RIN abn.", "omni_anat": "Omni anatomy"}
ERROR = dict(ecolor="#252525", elinewidth=0.65, capsize=1.7, capthick=0.65)


def schedule_panel(fig, data):
    ax = fig.add_axes((0.149, 0.22, 0.24, 0.59))
    positions = np.arange(len(CELLS), dtype=float)
    positions[[key[0] == "qwen3vl8b" for key in CELLS]] += 0.55
    for y, key in zip(positions, CELLS):
        name = "|".join(key)
        if data.get(name) is None:
            warn(f"no all-questions schedule delta for {name}")
            continue
        delta, se = data[name]
        if delta is None:
            warn(f"no all-questions schedule delta for {name}")
            continue
        if se is None:
            warn(f"no all-questions schedule SE for {name}")
        ax.barh(y, delta, xerr=se, height=0.67, linewidth=0.45, error_kw=ERROR, **FORMAT_STYLE[FORMATS[key[1]]])
    ax.axvline(0, color=DARK, lw=0.65, zorder=1)
    ax.set_yticks(positions, [SHORT_DATASET[ds] for _, ds in CELLS], fontsize=6.4)
    ax.set_ylim(positions[-1] + 0.58, -0.58)
    ax.set_xlabel("Accuracy change (pp)", labelpad=2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    clean_axis(ax, "x")
    ax.tick_params(axis="y", length=0)
    for model, title in MODEL_NAMES.items():
        group = [y for y, key in zip(positions, CELLS) if key[0] == model]
        ax.text(-0.49, np.mean(group), title, transform=ax.get_yaxis_transform(),
                ha="center", va="center", rotation=90, fontsize=6.7, fontweight="bold")
    handles = [Patch(label=label, linewidth=0.45, **style) for label, style in FORMAT_STYLE.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.222, 0.005), ncol=3,
               frameon=False, handlelength=1.1, handletextpad=0.35, columnspacing=0.85, fontsize=6.4)
    fig.text(0.022, 0.945, "(a) Front-loaded schedule for all questions", fontweight="bold", va="top")


def transfer_panel(fig, data):
    absolute = fig.add_axes((0.598, 0.22, 0.149, 0.59))
    effect = fig.add_axes((0.796, 0.22, 0.189, 0.59), sharey=absolute)
    positions = np.arange(len(PAIRS), dtype=float)
    positions[[name.startswith("merge |") for name in PAIRS]] += 0.55
    for y, (name, (base, added)) in zip(positions, PAIRS.items()):
        row = data.get(name, {})
        delta, se = row.get("mean"), row.get("se_approx")
        if delta is None:
            warn(f"no paired transfer delta for {name}")
            continue
        if se is None:
            warn(f"no paired transfer SE for {name}")
        component = name.split(" | ", 1)[0]
        color = COMPONENT_COLORS[component]
        effect.barh(y, delta, xerr=se, height=0.60, color=color, error_kw=ERROR, zorder=2)

        # Reproduce the source's exact cell set and per-cell QID alignment for the paired accuracy bars.
        keys = tuple(tuple(key.split("|")) for key in row.get("cells", {}))
        scores = [paired_accuracy(key, base, added) for key in keys]
        if not scores or any(pair is None for pair in scores):
            warn(f"{name}: incomplete paired accuracy bars")
            continue
        for key, pair in zip(keys, scores):
            recorded_delta = row["cells"]["|".join(key)][0]
            if not np.isclose(pair[1] - pair[0], recorded_delta, atol=0.0051, rtol=0):
                warn(f"{name} {key}: current runs disagree with the stored paired delta")
                break
        else:
            means = np.mean(scores, axis=0)
            absolute.barh(y - 0.16, means[0], height=0.29, color=LIGHT, zorder=2)
            absolute.barh(y + 0.16, means[1], height=0.29, color=color, zorder=2)

    labels = [name.replace(" | ", " \u00b7 ").replace("FastV keep-set", "FastV keep").replace(" (pre-LLM)", "")
              for name in PAIRS]
    absolute.set_yticks(positions, labels, fontsize=6.35)
    absolute.set_ylim(positions[-1] + 0.58, -0.58)
    absolute.set_xlim(0, 100)
    absolute.set_xticks((0, 50, 100))
    absolute.set_xlabel("Paired accuracy (%)", labelpad=2)
    effect.axvline(0, color=DARK, lw=0.65, zorder=1)
    effect.set_xlabel(r"$\Delta$ accuracy (pp)", labelpad=2)
    effect.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    for ax in (absolute, effect):
        clean_axis(ax, "x")
        ax.tick_params(axis="y", length=0)
    effect.tick_params(axis="y", labelleft=False)
    fig.text(0.462, 0.945, "(b) Component transfer across paired cells", fontweight="bold", va="top")
    handles = [Patch(color=LIGHT, label="Base")]
    handles.extend(Patch(color=color, label=f"Base + {name}") for name, color in COMPONENT_COLORS.items())
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.673, 0.005), ncol=3,
               frameon=False, handlelength=1.1, handletextpad=0.35, columnspacing=0.85, fontsize=6.4)


def main():
    fig = plt.figure(figsize=(6.9, 2.17))
    schedule_panel(fig, read_json("schedule_all_questions.json"))
    transfer_panel(fig, read_json("sec7_analyses.json").get("transfer", {}))
    save(fig, "fig7_schedule_transfer")


if __name__ == "__main__":
    main()
