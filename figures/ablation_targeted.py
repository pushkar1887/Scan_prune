"""Ablation of ScanPrune: each component's effect on all questions and on the questions it acts on.

Nov 11 basis (judged; Lingshu RIN abnormality balanced; seeds averaged). Paired per question, image-clustered
bootstrap per cell; cells averaged with equal weight, SE combined as independent cells.
  - anatomy gate:  jitter_merge -> gate_jitter_merge (10%); acts on images with acquisition background (bg_frac > 0).
  - Hilbert tiling: random_merge -> jitter_merge (10%); acts most where k is small (<= 64 tokens per image, k = 6).
  - merging:        random_l0 -> random_merge (10%, same random selection, pre-LLM).
  - mass term:      gate_jitter_merge -> gate_jitter_merge_pa (10%); _eq versions (equal budget);
                    registered depth-bound cells: Qwen3-VL OmniMedVQA, both open-anatomy cells.
  - read-aware schedule: _pa -> route (10% and equal budget); acts on single-token answers (MC, yes/no).
Also: covering radius of the keep-sets (tiling vs random), from keep_idx.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from medprune.adoption import BALANCED, balanced, load_cell, prompt_style  # noqa: E402
from medprune.judge import load_verdicts  # noqa: E402

R = ROOT / "results" / "all"
V = load_verdicts(R / "judge" / "verdicts.jsonl")
CELLS = [("lingshu7b", "slake_en"), ("lingshu7b", "rin_anat_mc"), ("lingshu7b", "rin_abn_closed"),
         ("lingshu7b", "omni_anat"), ("lingshu7b", "rin_open"), ("qwen3vl8b", "slake_en"),
         ("qwen3vl8b", "rin_anat_mc"), ("qwen3vl8b", "omni_anat"), ("qwen3vl8b", "rin_open")]
DEPTH = {("qwen3vl8b", "omni_anat"), ("lingshu7b", "rin_open"), ("qwen3vl8b", "rin_open")}
SP10 = "route:gate_jitter_merge_sched10_pa|gate_jitter_merge_pa"
SPEQ = "route:gate_jitter_merge_sched_pa|gate_jitter_merge_pa_eq"


def rowinfo(model, ds, method):
    info = {}
    for p in glob.glob(str(R / "runs" / f"{model}__{ds}__{method}__r010__s0__t4x2.jsonl")):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                info[r["qid"]] = r
    return info


def score(df, key, w=None):
    w = np.ones(len(df)) if w is None else w
    if key in BALANCED:
        return balanced(w, df["gold"].to_numpy(bool), df["yes"].to_numpy(float))
    return 100 * float((w * df["c"].to_numpy(float)).sum() / w.sum())


def paired(model, ds, a, b, keep=None, draws=1000):
    try:
        A_ = load_cell(R, model, ds, a, V)
        B_ = load_cell(R, model, ds, b, V)
    except (FileNotFoundError, ValueError):
        return None
    q = A_.index.intersection(B_.index)
    if keep is not None:
        q = q[[k in keep for k in q]]
    if len(q) < 30:
        return None
    A_, B_ = A_.loc[q], B_.loc[q]
    key = (model, ds)
    d = score(B_, key) - score(A_, key)
    u, inv = np.unique(A_["image"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(0)
    bs = [score(B_, key, w) - score(A_, key, w) for w in
          (np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv].astype(float) for _ in range(draws))]
    return d, float(np.std(bs, ddof=1)), len(q)


def combine(rs):
    rs = [r for r in rs if r]
    if not rs:
        return None
    return {"delta": round(float(np.mean([r[0] for r in rs])), 2),
            "se": round(float(np.sqrt(sum(r[1] ** 2 for r in rs))) / len(rs), 2),
            "cells": len(rs), "questions": int(sum(r[2] for r in rs)),
            "positive": f"{sum(r[0] > 0 for r in rs)}/{len(rs)}"}


OUT = {}

# anatomy gate
allr, act, idle, nact, ntot = [], [], [], 0, 0
for m, d in CELLS:
    info = rowinfo(m, d, "gate_jitter_merge")
    bg = {k for k, r in info.items() if (r.get("bg_frac") or 0) > 0}
    nact += len(bg); ntot += len(info)
    allr.append(paired(m, d, "jitter_merge", "gate_jitter_merge"))
    act.append(paired(m, d, "jitter_merge", "gate_jitter_merge", bg))
    idle.append(paired(m, d, "jitter_merge", "gate_jitter_merge", set(info) - bg))
OUT["gate"] = {"all": combine(allr), "acts": combine(act), "idle": combine(idle),
               "share_acts": round(nact / max(ntot, 1), 3)}

# Hilbert tiling
allr, small = [], []
for m, d in CELLS:
    info = rowinfo(m, d, "jitter_merge")
    sm = {k for k, r in info.items() if r.get("n_vis", 999) <= 64}
    allr.append(paired(m, d, "random_merge", "jitter_merge"))
    small.append(paired(m, d, "random_merge", "jitter_merge", sm))
OUT["tiling"] = {"all": combine(allr), "small_images": combine(small)}

# merging (same random selection, before the LLM)
OUT["merge"] = {"all": combine([paired(m, d, "random_l0", "random_merge") for m, d in CELLS])}

# mass term
for tag, a, b in (("10", "gate_jitter_merge", "gate_jitter_merge_pa"), ("eq", "gate_jitter_merge_eq", "gate_jitter_merge_pa_eq")):
    OUT[f"mass_{tag}"] = {"all": combine([paired(m, d, a, b) for m, d in CELLS]),
                          "depth_bound": combine([paired(m, d, a, b) for m, d in CELLS if (m, d) in DEPTH])}

# read-aware schedule: acts on single-token answers only
for tag, a, b in (("10", "gate_jitter_merge_pa", SP10), ("eq", "gate_jitter_merge_pa_eq", SPEQ)):
    allr, act = [], []
    for m, d in CELLS:
        info = rowinfo(m, d, a)
        single = {k for k, r in info.items() if prompt_style(d, r["answer_type"]) in ("mc", "judgement")}
        allr.append(paired(m, d, a, b))
        act.append(paired(m, d, a, b, single) if single else None)
    OUT[f"schedule_{tag}"] = {"all": combine(allr), "acts": combine(act)}

# total over Random
OUT["total_10"] = {"all": combine([paired(m, d, "random_l2", SP10) for m, d in CELLS])}
OUT["total_eq"] = {"all": combine([paired(m, d, "random_l2", SPEQ) for m, d in CELLS])}


# covering radius of keep-sets: max over grid cells of the distance to the nearest kept token
def radius(rows):
    out = []
    for r in rows.values():
        h, w = r["grid_hw"]
        keep = np.array(r["keep_idx"])
        if len(keep) == 0:
            continue
        kr, kc = keep // w, keep % w
        gr, gc = np.mgrid[0:h, 0:w]
        dist = np.sqrt((gr[..., None] - kr) ** 2 + (gc[..., None] - kc) ** 2).min(-1)
        out.append((dist.max(), dist.mean()))
    return np.array(out)


cov = {}
for m, d in [("lingshu7b", "rin_anat_mc"), ("qwen3vl8b", "rin_anat_mc"), ("lingshu7b", "slake_en"), ("qwen3vl8b", "slake_en")]:
    res = {}
    for name, meth in (("random", "random_merge"), ("tiling", "jitter_merge"), ("fastv", "fastv")):
        a = radius(rowinfo(m, d, meth))
        if len(a):
            res[name] = {"max_radius_mean": round(float(a[:, 0].mean()), 2), "max_radius_p90": round(float(np.percentile(a[:, 0], 90)), 2),
                         "mean_dist": round(float(a[:, 1].mean()), 2)}
    cov[f"{m}|{d}"] = res
OUT["coverage"] = cov

(Path(__file__).resolve().parent / "ablation_targeted.json").write_text(json.dumps(OUT, indent=1))
print(json.dumps(OUT, indent=1))
