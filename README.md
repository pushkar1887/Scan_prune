# ScanPrune

Code for *What Stays After the Cut, and Where Does the Rest Go? Visual Token Pruning for Medical MLLMs*
(anonymous submission).

The repository contains:

- **The harness**: 13 published visual-token pruners and random selection, evaluated at matched per-layer budgets on Lingshu-7B,
  Qwen3-VL-8B and HuatuoGPT-Vision-7B.
- **The diagnosis**: the interventions behind hypotheses H1–H5.
- **ScanPrune itself**: a training-free pruner that reads no attention scores. It has four parts:
  - an anatomy gate,
  - Hilbert tiling,
  - mass-preserving merging with a log-size attention term,
  - a read-aware schedule.

## Layout

```
├── medprune/               Python package
│   ├── selection.py        every method (Random, FastV, ScanPrune = "scanprune" / "scanprune_eq", ablations, gated baselines)
│   ├── remedy.py           Hilbert tiling, merging, log-size bias, read-aware schedule
│   ├── masks.py            anatomy gate (foreground mask, per-token foreground fraction)
│   ├── attention_hook.py   in-LLM scoring and masking (masking is numerically identical to token removal)
│   ├── run_eval.py         evaluation driver (CLI)
│   ├── models.py, huatuo_llava.py, data.py, env.py
│   ├── scoring.py, judge.py           answer scoring and the LLM judge for open answers
│   ├── analysis.py, adoption.py, stats.py, checks.py, latency.py
│   └── baselines/          ports of published pruners; sources, commits and licences in baselines/__init__.py
├── configs/runs.yaml       the run matrix (every run in the paper)
├── notebooks/              00 data bundle (CPU) · 01 GPU runs · 02 analysis · 03 LLM judge
├── figures/                scripts for the paper's figures and ablation statistics (+ their small JSON inputs)
├── scripts/setup_env.sh    environment setup; prints the compute profile
└── tests/                  CPU tests (selection, gate, merging, size-term identity, baselines vs. official code)
```

## Setup

```bash
bash scripts/setup_env.sh          # installs requirements.txt; torch>=2.4 comes from the platform
python -m pytest -q                # CPU tests
```

Profiles: `t4x2` (fp16, two T4 GPUs, used for all paper runs) or `a100` (bf16, one GPU); override with `MEDPRUNE_PROFILE`.

## Data

All benchmarks are public. `notebooks/00_build_data_bundle.ipynb` (or the CLI below) downloads them once. It freezes the question
lists and images and builds the per-token background masks.

```bash
python -m medprune.data build --out "$MEDPRUNE_DATA"
python -m medprune.data masks --bundle "$MEDPRUNE_DATA"
```

## Running

```bash
python -m medprune.run_eval --list                                   # runs, progress, pending GPU hours
python -m medprune.run_eval --exp <EXP> --model lingshu7b            # one experiment block from configs/runs.yaml
python -m medprune.run_eval --exp <EXP> --model qwen3vl8b --method scanprune --dataset slake_en
python -m medprune.run_eval --dry-run --limit 5 --exp <EXP> --model lingshu7b   # CPU smoke test with a fake model
```

`notebooks/01_run_gpu.ipynb` wraps the same CLI for a two-T4 notebook. Open answers are scored with
`notebooks/03_judge.ipynb`, and `notebooks/02_analyze_results.ipynb` builds the tables.

## Figures

Run from the repository root after the runs are in `results/all/`:

```bash
python figures/ablation_targeted.py      # ablation statistics -> figures/ablation_targeted.json
python figures/sec7_analyses.py          # analysis statistics -> figures/sec7_analyses.json
python figures/make_fig1_teaser.py       # ... make_fig8_omni_size.py
```

## Third-party code

Each module in `medprune/baselines/` ports the token-selection code of one published method from a pinned commit of its official
repository and keeps that method's licence; `medprune/baselines/__init__.py` lists the sources. Methods without released code
(FEATHER) or without a licence (BTP) are independent implementations of the published algorithms.

## Licence

MIT (see `LICENSE`), except for the ported baselines, which keep their original licences.
