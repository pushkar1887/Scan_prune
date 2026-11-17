"""Section 7 analyses from results/all (Nov 11 basis: judged; Lingshu RIN abnormality balanced; seeds averaged).

(A) 7.2 mass preservation: per-layer image share of the last prompt token, unpruned vs merge without / with the
    log-size term (same questions and keep-sets), four profiled cells.
(B) 7.5 OmniMedVQA by visual tokens per image: accuracy minus Full per bin.
(C) 7.4 transfer: component added to (or removed from) published pruners, paired per cell.
Writes sec7_analyses.json next to this file. Run from the repository root.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from medprune.adoption import BALANCED, balanced, load_cell  # noqa: E402
from medprune.judge import load_verdicts  # noqa: E402

R = ROOT / "results" / "all"
V = load_verdicts(R / "judge" / "verdicts.jsonl")
OUT = {}


def ragged(f, key):
    vals, off = f[key], f[key + "__offsets"]
    return [vals[off[i]:off[i + 1]] for i in range(len(off) - 1)]


# ---------------------------------------------------------------- (A) per-layer image share
LAYERS = {"lingshu7b": 28, "qwen3vl8b": 36}
A = {}
for model in ("lingshu7b", "qwen3vl8b"):
    for ds in ("slake_en", "rin_anat_mc"):
        L = LAYERS[model]
        full = {l: [] for l in range(1, L)}
        for p in sorted(glob.glob(str(R / "diag" / f"{model}__{ds}__full_profile__r100__s0__t4x2" / "*.npz"))):
            f = np.load(p, allow_pickle=True)
            for l in range(1, L):
                k = f"lp_last_l{l}"
                if k in f:
                    full[l] += [float(np.exp(x).sum()) for x in ragged(f, k)]
        withb = {l: [] for l in range(L)}
        without = {l: [] for l in range(L)}
        for d in sorted(glob.glob(str(R / "diag" / f"{model}__{ds}__gate_jitter_merge_pa_eq__r010__s*__t4x2"))):
            for p in sorted(glob.glob(d + "/*.npz")):
                f = np.load(p, allow_pickle=True)
                for l in range(L):
                    k = f"ishare_l{l}"
                    if k in f:
                        for x in ragged(f, k):
                            if len(x) == 2:
                                withb[l].append(float(x[0]))
                                without[l].append(float(x[1]))
        med = lambda d: {l: (float(np.median(v)) if v else None) for l, v in d.items()}
        deep = lambda d: float(np.median([np.median(v) for l, v in d.items() if l >= L // 2 and v]))
        A[f"{model}|{ds}"] = {"full": med(full), "with": med(withb), "without": med(without),
                             "deep_full": deep(full), "deep_with": deep(withb), "deep_without": deep(without),
                             "n_full": len(full[1]), "n_pruned": len(withb[0])}
        print("A", model, ds, {k: round(v, 4) for k, v in A[f"{model}|{ds}"].items() if k.startswith("deep")})
OUT["mass_preservation"] = A

# ---------------------------------------------------------------- (B) OmniMedVQA by image size
SP10 = "route:gate_jitter_merge_sched10_pa|gate_jitter_merge_pa"
SPEQ = "route:gate_jitter_merge_sched_pa|gate_jitter_merge_pa_eq"
B_METHODS = {"Full": "full", "ScanPrune": SP10, "VisionZip": "visionzip", "VScan": "vscan", "ScanPrune-eq": SPEQ,
             "FastV": "fastv"}
BINS = [(0, 64, "<=64"), (65, 128, "65-128"), (129, 400, "129-400"), (401, 10 ** 6, ">400")]
B = {}
for model in ("lingshu7b", "qwen3vl8b"):
    nvis = {}
    for p in glob.glob(str(R / "runs" / f"{model}__omni_anat__full__r100__s0__t4x2.jsonl")):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                nvis[r["qid"]] = r["n_vis"]
    acc = {name: load_cell(R, model, "omni_anat", m, V)["c"] for name, m in B_METHODS.items()}
    res = {}
    for lo, hi, lab in BINS:
        q = [k for k, n in nvis.items() if lo <= n <= hi and all(k in acc[x].index for x in acc)]
        f = 100 * acc["Full"].loc[q].mean()
        res[lab] = {"n": len(q), "share": round(len(q) / len(nvis), 3), "Full": round(f, 1)}
        for name in B_METHODS:
            if name != "Full":
                res[lab][name] = round(100 * acc[name].loc[q].mean() - f, 1)
    B[model] = res
    print("B", model, res)
OUT["omni_by_size"] = B

# ---------------------------------------------------------------- (C) transfer pairs
CELLS = [("lingshu7b", "slake_en"), ("lingshu7b", "rin_anat_mc"), ("lingshu7b", "rin_abn_closed"),
         ("lingshu7b", "omni_anat"), ("lingshu7b", "rin_open"), ("qwen3vl8b", "slake_en"),
         ("qwen3vl8b", "rin_anat_mc"), ("qwen3vl8b", "omni_anat"), ("qwen3vl8b", "rin_open")]
PAIRS = {  # name: (base, with component)
    "gate | FastV": ("fastv", "gate_fastv"),
    "gate | PDrop": ("pdrop", "gate_pdrop"),
    "gate | DivPrune": ("divprune", "gate_divprune"),
    "gate | CDPruner": ("cdpruner_neg", "gate_cdpruner_neg"),
    "gate | VisionZip": ("visionzip", "gate_visionzip"),
    "merge | FastV keep-set": ("fastv_prellm", "fastv_merge"),
    "merge | VisionZip": ("visionzip_nomerge", "visionzip"),
    "merge | VScan": ("vscan_nomerge", "vscan"),
    "merge | Random (pre-LLM)": ("random_l0", "random_merge"),
}


def score(df, key, w=None):
    w = np.ones(len(df)) if w is None else w
    if key in BALANCED:
        return balanced(w, df["gold"].to_numpy(bool), df["yes"].to_numpy(float))
    return 100 * float((w * df["c"].to_numpy(float)).sum() / w.sum())


def paired(model, ds, a, b, draws=1000, seed=0):
    try:
        A_ = load_cell(R, model, ds, a, V)
        B_ = load_cell(R, model, ds, b, V)
    except (FileNotFoundError, ValueError):
        return None
    q = A_.index.intersection(B_.index)
    if len(q) < 0.95 * min(len(A_), len(B_)):
        return None
    A_, B_ = A_.loc[q], B_.loc[q]
    key = (model, ds)
    d = score(B_, key) - score(A_, key)
    u, inv = np.unique(A_["image"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(draws):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv].astype(float)
        bs.append(score(B_, key, w) - score(A_, key, w))
    return round(d, 2), round(float(np.std(bs, ddof=1)), 2)


C = {}
for name, (a, b) in PAIRS.items():
    cells = {}
    for model, ds in CELLS:
        r = paired(model, ds, a, b)
        if r:
            cells[f"{model}|{ds}"] = r
    if cells:
        ds_ = [v[0] for v in cells.values()]
        se = float(np.sqrt(sum(v[1] ** 2 for v in cells.values()))) / len(cells)
        C[name] = {"cells": cells, "mean": round(float(np.mean(ds_)), 2), "se_approx": round(se, 2),
                   "positive": f"{sum(x > 0 for x in ds_)}/{len(ds_)}"}
        print("C", name, C[name]["mean"], C[name]["se_approx"], C[name]["positive"])
OUT["transfer"] = C

(Path(__file__).resolve().parent / "sec7_analyses.json").write_text(json.dumps(OUT, indent=1))
print("saved")
