"""Retained visual gain per cell, (acc - text_only) / (Full - text_only), on the registered basis
(judged; Lingshu RIN abnormality balanced; seeds on disk averaged). Run from the repository root."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from medprune.adoption import CELLS, BALANCED, balanced, load_cell  # noqa: E402
from medprune.analysis import parse_run_id, rescore_open  # noqa: E402
from medprune.judge import apply_judge, load_verdicts  # noqa: E402

R = Path("results/all")
V = load_verdicts(R / "judge" / "verdicts.jsonl")
ROWS = {"Random": "random_l2", "FastV": "fastv", "VisionZip": "visionzip", "VScan": "vscan",
        "ScanPrune-10": "route:gate_jitter_merge_sched10_pa|gate_jitter_merge_pa",
        "ScanPrune-eq": "route:gate_jitter_merge_sched_pa|gate_jitter_merge_pa_eq"}


def acc(df, key):
    if key in BALANCED:
        return balanced(np.ones(len(df)), df["gold"].to_numpy(bool), df["yes"].to_numpy(float))
    return 100 * df["c"].mean()


def text_only(model, ds):
    rows = []
    for p in sorted((R / "runs").glob(f"{model}__{ds}__text_only__*__t4x2.jsonl")):
        info = parse_run_id(p.stem)
        rows += [{**json.loads(l), **info} for l in p.open(encoding="utf-8") if l.strip()]
    df = apply_judge(rescore_open(pd.DataFrame(rows)), V)
    if ds == "rin_open":
        df = df[df["qid"].str.endswith("anatomy_open")]
    gold = df["answer"].str.strip().str.lower().str.startswith("yes").to_numpy(bool)
    yes = df["response"].fillna("").str.strip().str.lower().str.match(r"\W*yes").to_numpy(float)
    return balanced(np.ones(len(df)), gold, yes) if (model, ds) in BALANCED else 100 * df["correct"].mean()


out = {}
for model, ds in CELLS:
    key = (model, ds)
    t = text_only(model, ds)
    f = acc(load_cell(R, model, ds, "full", V), key)
    out[f"{model}|{ds}"] = {"text": round(t, 1), "full": round(f, 1)}
    for name, m in ROWS.items():
        a = acc(load_cell(R, model, ds, m, V), key)
        out[f"{model}|{ds}"][name] = round(100 * (a - t) / (f - t), 1)
print(json.dumps(out, indent=1))
means = {name: round(np.mean([v[name] for v in out.values()]), 1) for name in ROWS}
print("nine-cell mean RVG (%):", means)
