"""7.2 deep-layer image share with the registered P70 statistic: per question, the mean over layers >= L/2 of the
last prompt token's image share; then the median over questions. Seed 0 (the registered run) and all seeds."""
import glob
import sys
from pathlib import Path

import numpy as np

R = Path.cwd() / "results" / "all"
L = {"lingshu7b": 28, "qwen3vl8b": 36}


def ragged(f, key):
    v, o = f[key], f[key + "__offsets"]
    return [v[o[i]:o[i + 1]] for i in range(len(o) - 1)]


for model in ("lingshu7b", "qwen3vl8b"):
    for ds in ("slake_en", "rin_anat_mc"):
        n = L[model]
        for tag, pat in (("s0", "s0"), ("all", "s*")):
            per_q = {"with": [], "without": []}
            for d in sorted(glob.glob(str(R / "diag" / f"{model}__{ds}__gate_jitter_merge_pa_eq__r010__{pat}__t4x2"))):
                for p in sorted(glob.glob(d + "/*.npz")):
                    f = np.load(p, allow_pickle=True)
                    deep = [ragged(f, f"ishare_l{l}") for l in range(n // 2, n) if f"ishare_l{l}" in f]
                    for q in range(len(deep[0])):
                        vals = np.array([layer[q] for layer in deep if len(layer[q]) == 2])
                        per_q["with"].append(vals[:, 0].mean())
                        per_q["without"].append(vals[:, 1].mean())
            full_q = []
            for p in sorted(glob.glob(str(R / "diag" / f"{model}__{ds}__full_profile__r100__s0__t4x2" / "*.npz"))):
                f = np.load(p, allow_pickle=True)
                deep = [ragged(f, f"lp_last_l{l}") for l in range(n // 2, n) if f"lp_last_l{l}" in f]
                for q in range(len(deep[0])):
                    full_q.append(np.mean([np.exp(layer[q]).sum() for layer in deep]))
            w, wo, fu = np.median(per_q["with"]), np.median(per_q["without"]), np.median(full_q)
            print(f"{model:10s} {ds:12s} {tag:3s} with {w:.4f} without {wo:.4f} ratio {w / wo:.2f}x  full {fu:.4f}"
                  f"  n={len(per_q['with'])}")
