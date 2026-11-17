"""Fig. 3: ScanPrune on one SLAKE CT (slake/xmlab208, Lingshu-7B, 18 x 18 = 324 visual tokens).

What is real and what is not:
  (a)-(c) Real. The gate is `medprune.masks.image_gate` at tau = 0.5; its background fraction is checked against the
          `bg_frac` the ScanPrune run logged. The Hilbert strata come from `medprune.remedy.curve_order`, split into
          k1 = 40 strata. The kept tokens are the run's own layer-0 keep-set (seed 0, question slake_en/12079); the
          script checks that they sit one per stratum.
  (d)     The keep-set is the run's own. The merge assignment is `medprune.remedy.merge_plan`, but on patch-pixel
          features: Lingshu's vision embeddings are not stored, and the encoder cannot run on this machine. So this panel
          shows how merging works, not the exact clusters of the run.
  (e)     Real counts from the run: 40 tokens before layer c = 20, then 12 (single-token answers), and a flat 32 from
          the generated-answer branch (gate_jitter_merge_pa). L = 28.
"""

import json

import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from _figure_common import DARK, LIGHT, MID, RED, RESULTS, plt, save, slake_ct, warn
from medprune.masks import image_gate, model_view
from medprune.remedy import curve_order, merge_plan

RUN = "lingshu7b__slake_en__gate_jitter_merge_sched10_pa__r010__s0__t4x2"
FLAT_RUN = "lingshu7b__slake_en__gate_jitter_merge_pa__r010__s0__t4x2"
QID = "slake_en/12079"
TAU = 0.5
N_LAYERS = 28
# Muted qualitative palette for tiles and merge groups (red is reserved for kept tokens).
GROUP_COLOURS = ("#4f7cac", "#d08c2c", "#4f9a6a")
PALETTE = ("#a9c0d6", "#dcc296", "#afd0af", "#cdb0c7", "#cfcfa4", "#a6c8cc", "#deb4a6", "#bab4da")


def run_row(run, qid):
    path = RESULTS / "runs" / f"{run}.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if f'"{qid}"' in line:
                row = json.loads(line)
                if row["qid"] == qid:
                    return row
    raise KeyError(f"{qid} not in {run}")


def colour_groups(labels, side):
    """Greedy colouring so that 4-adjacent groups get different palette entries."""
    grid = labels.reshape(side, side)
    neighbours = {}
    for r in range(side):
        for c in range(side):
            a = grid[r, c]
            for rr, cc in ((r + 1, c), (r, c + 1)):
                if rr < side and cc < side:
                    b = grid[rr, cc]
                    if a >= 0 and b >= 0 and a != b:
                        neighbours.setdefault(a, set()).add(b)
                        neighbours.setdefault(b, set()).add(a)
    colour = {}
    for g in sorted(set(labels[labels >= 0])):
        used = {colour[n] for n in neighbours.get(g, ()) if n in colour}
        colour[g] = next(i for i in range(len(PALETTE)) if i not in used)
    return colour


def frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#a5a5a5")
        spine.set_linewidth(0.5)
    ax.set(xlim=(0, 18), ylim=(18, 0), aspect="equal")


def grid_lines(ax, side, colour, alpha, lw=0.25):
    for t in range(1, side):
        ax.axhline(t, color=colour, lw=lw, alpha=alpha)
        ax.axvline(t, color=colour, lw=lw, alpha=alpha)


def cells(ax, idx, side, colours, alpha=1.0, z=1):
    for i, fc in zip(idx, colours):
        r, c = divmod(int(i), side)
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=fc, edgecolor="none", alpha=alpha, zorder=z))


def borders(ax, labels, side, colour="white", lw=0.9):
    """Lines along every cell edge that separates two different (non-negative) groups."""
    grid = labels.reshape(side, side)
    for r in range(side):
        for c in range(side):
            a = grid[r, c]
            if a < 0:
                continue
            if c + 1 < side and grid[r, c + 1] >= 0 and grid[r, c + 1] != a:
                ax.plot((c + 1, c + 1), (r, r + 1), color=colour, lw=lw, zorder=2, solid_capstyle="butt")
            if r + 1 < side and grid[r + 1, c] >= 0 and grid[r + 1, c] != a:
                ax.plot((c, c + 1), (r + 1, r + 1), color=colour, lw=lw, zorder=2, solid_capstyle="butt")


def patch_features(view, side, cell):
    """Stand-in embeddings for the merge panel: each token's 28 x 28 patch, pooled to 7 x 7, centred."""
    grey = np.asarray(view.convert("L"), dtype=np.float64)[: side * cell, : side * cell] / 255
    pooled = grey.reshape(side, 7, cell // 7, side, 7, cell // 7).mean(axis=(2, 5))
    feats = pooled.transpose(0, 2, 1, 3).reshape(side * side, 49)
    return feats - feats.mean(axis=0)


def main():
    sample = slake_ct()
    if sample is None:
        raise SystemExit("verified SLAKE image unavailable")
    image, record = sample
    model = record["model"]
    view, _ = model_view(image, model)
    (h, w), frac = image_gate(image, model)
    if (h, w) != (18, 18):
        raise SystemExit(f"unexpected grid {(h, w)}")
    side, n = h, h * w
    allowed = frac.reshape(-1) >= TAU

    row = run_row(RUN, QID)
    if abs(row["bg_frac"] - (1 - allowed.mean())) > 1e-9:
        raise SystemExit("gate differs from the run's logged background fraction")
    (_, keep40), (cut, keep12) = row["keep_stages"]
    keep40, keep12 = np.array(sorted(keep40)), np.array(sorted(keep12))
    k_flat = run_row(FLAT_RUN, QID)["k_keep"]

    rc = np.indices((h, w)).reshape(2, -1).T
    order = curve_order(rc)
    along = order[allowed[order]]
    strata = np.array_split(along, len(keep40))
    tile = np.full(n, -1)
    for i, members in enumerate(strata):
        tile[members] = i
    if len(set(tile[keep40])) != len(keep40) or not allowed[keep40].all():
        raise SystemExit("the run's keep-set is not one token per Hilbert stratum")
    tile_colour = colour_groups(tile, side)

    feats = patch_features(view, side, 28)
    rows, weights = merge_plan(feats, keep40, allowed=allowed)
    target = np.full(n, -1)
    for j, kept in enumerate(rows):
        target[np.flatnonzero(weights[j] > 0)] = kept
    size = {int(k): int((target == k).sum()) for k in rows}

    fig_w, fig_h = 6.9, 2.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    xs = (0.008, 0.198, 0.388, 0.578)
    pw, py = 0.158, 0.205
    ph = pw * fig_w / fig_h
    axes = [fig.add_axes((x, py, pw, ph)) for x in xs]
    sx, sw = 0.818, 0.160
    sched = fig.add_axes((sx, py + 0.02, sw, ph - 0.02))
    pixels = np.asarray(view)[: side * 28, : side * 28]

    # (a) input
    ax = axes[0]
    ax.imshow(pixels, extent=(0, 18, 18, 0), interpolation="bilinear")
    grid_lines(ax, side, "white", 0.35)
    frame(ax)

    # (b) anatomy gate: background cells struck out
    ax = axes[1]
    ax.imshow(pixels, extent=(0, 18, 18, 0), interpolation="bilinear")
    cells(ax, np.flatnonzero(~allowed), side, ["#f4f4f4"] * n, z=2)
    grid_lines(ax, side, "#9a9a9a", 0.5)
    frame(ax)

    # (c) Hilbert tiles, the curve through the body, one kept token per tile
    ax = axes[2]
    idx = np.flatnonzero(allowed)
    cells(ax, idx, side, [PALETTE[tile_colour[tile[i]]] for i in idx], alpha=0.8)
    for a, b in zip(along, along[1:]):
        (ra, ca), (rb, cb) = rc[a], rc[b]
        if abs(ra - rb) + abs(ca - cb) == 1:
            ax.plot((ca + 0.5, cb + 0.5), (ra + 0.5, rb + 0.5), color=MID, lw=0.45, zorder=3,
                    solid_capstyle="round")
    ax.scatter(rc[keep40, 1] + 0.5, rc[keep40, 0] + 0.5, s=7, color=RED, edgecolors="white", linewidths=0.3,
               zorder=4)
    grid_lines(ax, side, "#d8d8d8", 0.8)
    borders(ax, tile, side)
    frame(ax)

    # (d) merging: each dropped token takes its kept token's colour; dot area grows with the merged count s
    ax = axes[3]
    # Three groups highlighted (the largest merged sizes, spatially apart); the other dropped body tokens in grey.
    cells(ax, idx, side, ["#e2e2e2"] * len(idx))
    highlight = []
    for k in sorted(keep40, key=lambda k: -size[int(k)]):
        if all(np.abs(rc[k] - rc[h]).sum() >= 6 for h in highlight):
            highlight.append(k)
        if len(highlight) == 3:
            break
    picks = []
    for k, colour in zip(highlight, GROUP_COLOURS):
        members = np.flatnonzero((target == k) & (np.arange(n) != k))
        cells(ax, members, side, [colour] * len(members), alpha=0.75, z=1.5)
        far = members[np.argsort(np.abs(rc[members] - rc[k]).sum(axis=1), kind="stable")]
        chosen = []
        for i in far:
            if 2 <= np.abs(rc[i] - rc[k]).sum() <= 4 and all(np.abs(rc[i] - rc[j]).sum() >= 3 for j in chosen):
                chosen.append(i)
            if len(chosen) == 2:
                break
        picks += [(i, k) for i in chosen]
    grid_lines(ax, side, "white", 0.9, lw=0.35)
    for i, t in picks:
        ax.add_patch(FancyArrowPatch((rc[i, 1] + 0.5, rc[i, 0] + 0.5), (rc[t, 1] + 0.5, rc[t, 0] + 0.5),
                                     arrowstyle="-|>", mutation_scale=4.5, color=DARK, lw=0.55, shrinkA=0,
                                     shrinkB=2.5, zorder=5))
    big = highlight[0]
    ax.annotate(f"$s={size[int(big)]}$", (rc[big, 1] + 0.5, rc[big, 0] + 0.5), xytext=(1.0, 16.6), fontsize=6,
                color=DARK, ha="left", va="center", zorder=6,
                arrowprops=dict(arrowstyle="-", color=MID, lw=0.45, shrinkA=1, shrinkB=4))
    ax.scatter(rc[keep40, 1] + 0.5, rc[keep40, 0] + 0.5, s=[2 + 1.6 * size[int(k)] for k in keep40], color=RED,
               edgecolors="white", linewidths=0.3, zorder=4)
    frame(ax)

    # (e) read-aware schedule
    ax = sched
    k1, k2 = len(keep40), len(keep12)
    ax.step((0, cut, N_LAYERS), (k1, k2, k2), where="post", color=RED, lw=1.1)
    ax.plot((0, N_LAYERS), (k_flat, k_flat), color=MID, lw=0.9, ls=(0, (3, 2)))
    ax.axvline(cut, color=LIGHT, lw=0.5, ls=":", zorder=0)
    ax.set(xlim=(0, N_LAYERS), ylim=(0, 50), xticks=(0, cut, N_LAYERS), yticks=(k2, k_flat, k1))
    ax.set_xticklabels(("1", "$c$", "$L$"))
    ax.set_xlabel("decoder layer", labelpad=0.5, fontsize=6)
    ax.set_ylabel("visual tokens", labelpad=1, fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=1.5, labelsize=6)
    ax.text(0.8, k1 + 1.5, "single-token answers", color=RED, fontsize=6, va="bottom")
    ax.text(0.8, k_flat - 2.0, "generated", color=MID, fontsize=6, ha="left", va="top")

    titles = ("(a) Input", "(b) Anatomy gate", "(c) Hilbert tiling", "(d) Mass-preserving merging",
              "(e) Read-aware scheduling")
    subs = (f"$N={n}$ visual tokens\n" r"(one per $28\times28$ patch)",
            f"{int((~allowed).sum())} background tokens\nremoved before the LLM",
            f"{k1} tiles along the curve,\none token kept per tile",
            "dropped token $\\to$ most similar\n" r"kept token; attention $+\log s$")
    centres = [x + pw / 2 for x in xs] + [sx + sw / 2]
    top = py + ph
    for x, t in zip(centres, titles):
        fig.text(x, top + 0.03, t, ha="center", va="bottom", fontsize=7)
    for x, s in zip(centres, subs):
        fig.text(x, py - 0.03, s, ha="center", va="top", fontsize=6, color=DARK, linespacing=1.15)

    for left, right in zip(xs, xs[1:] + (sx - 0.045,)):
        fig.add_artist(FancyArrowPatch((left + pw + 0.004, py + ph / 2), (right - 0.006, py + ph / 2),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=5,
                                       color=DARK, lw=0.6))

    def bracket(x0, x1, label):
        y = top + 0.155
        fig.add_artist(plt.Line2D((x0, x0, x1, x1), (y - 0.03, y, y, y - 0.03), transform=fig.transFigure,
                                  color=MID, lw=0.5))
        fig.text((x0 + x1) / 2, y + 0.015, label, ha="center", va="bottom", fontsize=7, style="italic")

    bracket(xs[1], xs[2] + pw, "what stays")
    bracket(xs[3], sx + sw, "where the rest goes")
    if size:
        warn_if = [k for k in keep40 if size[int(k)] < 1]
        if warn_if:
            warn(f"kept tokens with s < 1: {warn_if}")
    print(f"s: min {min(size.values())} max {max(size.values())} mean {np.mean(list(size.values())):.2f}; "
          f"merge arrows {len(picks)}")
    save(fig, "fig3_pipeline")


if __name__ == "__main__":
    main()
