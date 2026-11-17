"""Shared Fig. 4 typography and read-only access to the paper's results."""

from functools import lru_cache
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results" / "all"
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from medprune.adoption import BALANCED, balanced, load_cell
from medprune.judge import load_verdicts

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42})

RED = "#c0392b"
DARK = "#4d4d4d"
MID = "#7a7a7a"
LIGHT = "#b5b5b5"
FULL = "#2c3e50"
MODEL_NAMES = {"lingshu7b": "Lingshu", "qwen3vl8b": "Qwen3-VL"}
DATASET_NAMES = {
    "slake_en": "SLAKE",
    "rin_anat_mc": "RIN anatomy",
    "rin_abn_closed": "RIN abnormality",
    "omni_anat": "Omni anatomy",
    "rin_open": "RIN open",
}


def warn(message):
    print(f"WARNING: {message}", file=sys.stderr)


def read_json(name):
    path = HERE / name
    if not path.is_file():
        warn(f"missing {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def verdicts():
    path = RESULTS / "judge" / "verdicts.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing judge verdicts: {path}")
    return load_verdicts(path)


@lru_cache(maxsize=None)
def cell(model, dataset, method):
    try:
        frame = load_cell(RESULTS, model, dataset, method, verdicts())
    except (FileNotFoundError, ValueError) as error:
        warn(str(error))
        return None
    if frame.empty:
        warn(f"no scored questions for {model} {dataset} {method}")
        return None
    return frame


def accuracy(frame, key):
    if key in BALANCED:
        return balanced(np.ones(len(frame)), frame["gold"].to_numpy(bool), frame["yes"].to_numpy(float))
    return 100 * float(frame["c"].mean())


def paired_accuracy(key, base, other):
    a, b = (cell(*key, method) for method in (base, other))
    if a is None or b is None:
        return None
    qids = a.index.intersection(b.index)
    if len(qids) < len(a) or len(qids) < len(b):
        warn(f"{key}: {base} vs {other} has {len(qids)} shared questions (of {len(a)} and {len(b)})")
    if qids.empty or not a.loc[qids, "image"].equals(b.loc[qids, "image"]):
        warn(f"{key}: could not align image IDs for {base} vs {other}")
        return None
    scores = accuracy(a.loc[qids], key), accuracy(b.loc[qids], key)
    if not np.isfinite(scores).all():
        warn(f"{key}: no finite paired scores for {base} vs {other}")
        return None
    return scores


def seed_zero_rows(model, dataset, method):
    path = RESULTS / "runs" / f"{model}__{dataset}__{method}__r010__s0__t4x2.jsonl"
    if not path.is_file():
        warn(f"missing {path.name}")
        return {}
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len({row["qid"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate QIDs in {path.name}")
    return {row["qid"]: row for row in rows}


def valid_keep(row):
    grid = row.get("grid_hw")
    indices = row.get("keep_idx")
    return bool(
        grid and len(grid) == 2 and min(grid) > 0 and indices
        and len(indices) == len(set(indices)) and min(indices) >= 0 and max(indices) < np.prod(grid)
    )


def slake_ct():
    record = read_json("slake_ct.json")
    if not record:
        return None
    image_id = record["image_id"]
    path = (HERE / record["image_file"]).resolve()
    if (not image_id.startswith("slake/") or "radimagenet" in image_id.lower()
            or "radimagenet" in str(path).lower() or path.parent != HERE):
        warn(f"skipping disallowed CT image {image_id}")
        return None
    if not path.is_file():
        warn(f"missing SLAKE source image {path.name}")
        return None
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["image_sha256"]:
        warn(f"SLAKE source image checksum mismatch: {path.name}")
        return None
    with Image.open(path) as image:
        return image.convert("RGB"), record


def clean_axis(ax, grid=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=2)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis=grid, lw=0.3, alpha=0.5)


def save(fig, stem):
    for extension in ("pdf", "png"):
        path = HERE / f"{stem}.{extension}"
        fig.savefig(path, dpi=300)
        print(f"saved {path.relative_to(ROOT)}")
    plt.close(fig)
