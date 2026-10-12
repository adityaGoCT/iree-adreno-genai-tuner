"""Charts for the GEMM write-up (reads runs/<tag>/*.json, writes docs/img/gemm-*.png).

  plots.py --tag run2
"""

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from shapes import BY_ID, SHAPES  # noqa: E402

RUNS = pathlib.Path(__file__).parent / "runs"
IMG = pathlib.Path(__file__).resolve().parents[1] / "docs/img"
C = {"naive": "#9aa0a6", "iree": "#e8710a", "random": "#4285f4", "model": "#0f9d58", "ext": "#00796b",
     "roof": "#202124"}


def label(s):
    return f"{s.id} {s.name}\n{s.M}x{s.N}x{s.K} {s.layout}"


def curves(summ, out):
    fig, axs = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for ax, s in zip(axs.flat, SHAPES):
        row = summ.get(s.id)
        if not row:
            ax.set_visible(False)
            continue
        for m in ("random", "model"):
            if m in row:
                cv = row[m]["curve"]
                ax.plot(range(1, len(cv) + 1), cv, color=C[m], lw=2, label="model-guided" if m == "model" else "random")
        if "ext" in row:  # continuation with promoted fp16 accumulation, after the model run
            cv = row["ext"]["curve"]
            n0 = len(row.get("model", {}).get("curve", []))
            ax.plot(range(n0, len(cv) + 1), cv[n0 - 1:], color=C["ext"], lw=2, ls="--", label="+ f16p primitive")
        nv = row.get("naive", {}).get("best_gflops")
        if nv:
            ax.axhline(nv, color=C["naive"], ls=":", label="naive")
        ax.set_title(label(s), fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axs[1]:
        ax.set_xlabel("candidates measured on the GPU")
    for ax in axs[:, 0]:
        ax.set_ylabel("best GFLOP/s so far")
    axs.flat[0].legend(fontsize=8)
    fig.suptitle("Search efficiency: best-so-far GFLOP/s vs GPU measurements (same budget, same verifier)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def final(table, out):
    """Dot plot on a log axis: naive, IREE default, random-search best, tuned (best round, with its
    range over rounds as a whisker) and the roofline per shape (runs/<tag>/final_table.json)."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ys = np.arange(len(table))[::-1]
    for y, r in zip(ys, table):
        ax.plot([r["tlo"], r["tuned"]], [y, y], color=C["model"], lw=4, alpha=0.35, solid_capstyle="round")
        ax.scatter(r["naive"], y, s=60, facecolors="none", edgecolors=C["naive"], lw=1.5, zorder=3)
        ax.scatter(r["iree"], y, s=60, marker="D", color=C["iree"], zorder=3)
        ax.scatter(r["random"], y, s=60, color=C["random"], zorder=3)
        ax.scatter(r["tuned"], y, s=90, color=C["model"], zorder=4)
        ax.plot([r["roof"]] * 2, [y - 0.3, y + 0.3], color=C["roof"], lw=2.5)
        ax.text(1500, y, f"{r['tuned']:.0f} GFLOP/s   {r['tuned'] / r['naive']:.1f}x naive   {r['tuned'] / r['iree']:.0f}x IREE",
                va="center", fontsize=9)
    ax.set_yticks(ys, [f"{r['s']} {r['name']}" for r in table])
    ax.set_xscale("log")
    ax.set_xlim(1, 1200)
    ax.set_xlabel("GFLOP/s (log scale); best of 5 interleaved rounds after a 3 s GPU warm-up")
    for lbl, kw in (("naive", dict(facecolors="none", edgecolors=C["naive"])), ("IREE 3.12 default", dict(marker="D", color=C["iree"])),
                    ("random search, 96", dict(color=C["random"])), ("tuned (whisker: range over rounds)", dict(color=C["model"]))):
        ax.scatter([], [], s=60, label=lbl, **kw)
    ax.plot([], [], color=C["roof"], lw=2.5, label="roofline")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
    ax.grid(axis="x", alpha=0.3)
    gm = lambda xs: float(np.exp(np.mean(np.log(xs))))  # noqa: E731
    ax.set_title(f"Tuned kernels: {gm([r['tuned'] / r['naive'] for r in table]):.1f}x the naive kernel and "
                 f"{gm([r['tuned'] / r['iree'] for r in table]):.0f}x IREE's default (geomean)", pad=36)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")


def ablation(abl, out):
    import numpy as np
    names = []
    for d in abl.values():
        for n in d:
            if n != "best" and n not in names:
                names.append(n)
    ids = list(abl)
    M = np.full((len(names), len(ids)), np.nan)
    for j, sid in enumerate(ids):
        b = abl[sid]["best"]["us"]
        for i, n in enumerate(names):
            v = abl[sid].get(n)
            if v and v.get("us") and b:
                M[i, j] = v["us"] / b
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(names) + 2))
    im = ax.imshow(np.log2(M), cmap="RdYlGn_r", vmin=-1, vmax=4, aspect="auto")
    for i in range(len(names)):
        for j in range(len(ids)):
            v = M[i, j]
            txt = "n/a" if np.isnan(v) else f"{v:.2f}x"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(ids)), [label(BY_ID[i]) for i in ids], fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_title("Ablation: slowdown when one primitive is removed from the tuned schedule (n/a = not used / illegal)")
    fig.colorbar(im, ax=ax, label="log2 slowdown")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def ranking(rank, out):
    import numpy as np
    ids = list(rank)
    x = np.arange(len(ids))
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    for k, (m, col) in enumerate((("analytic", "#a142f4"), ("learned", C["model"]))):
        axs[0].bar(x + (k - 0.5) * 0.35, [rank[i][m]["rho"] for i in ids], 0.35, color=col, label=m)
        axs[1].bar(x + (k - 0.5) * 0.35, [100 * rank[i][m]["topk_frac"] for i in ids], 0.35, color=col, label=m)
    axs[0].set_ylabel("Spearman rho (held-out shape)")
    axs[1].set_ylabel("% of pool-best reached by the top-16 picks")
    for ax in axs:
        ax.set_xticks(x, ids)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    axs[0].set_title("Cost-model ranking on unseen shapes (leave-one-shape-out)")
    axs[1].set_title("If only the model's top-16 were measured")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def verifier(ver, out):
    rej = ver["rejected"]
    items = sorted(rej.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(items) + 1.5))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#5f6368")
    ax.set_xlabel("proposals rejected before reaching the GPU (all shapes, both searches)")
    ax.set_title(f"Verifier: {ver['valid']} of {ver['proposed']} proposals legal; "
                 f"{ver['measured'] - ver['gpu_failures']}/{ver['measured']} verified candidates correct on the GPU")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run2")
    a = ap.parse_args()
    d = RUNS / a.tag
    IMG.mkdir(parents=True, exist_ok=True)
    rd = lambda f: json.loads((d / f).read_text()) if (d / f).exists() else None  # noqa: E731
    summ = rd("summary.json")
    iree = json.loads((RUNS / "iree/kernel_results.json").read_text()) if (RUNS / "iree/kernel_results.json").exists() else {}
    if summ:
        curves(summ, IMG / "gemm-search-curves.png")
        if rd("final_table.json"):
            final(rd("final_table.json"), IMG / "gemm-final.png")
    if rd("ablation.json"):
        ablation(rd("ablation.json"), IMG / "gemm-ablation.png")
    if rd("rank.json"):
        ranking(rd("rank.json"), IMG / "gemm-cost-model.png")
    if rd("verifier.json"):
        verifier(rd("verifier.json"), IMG / "gemm-verifier.png")
    print("wrote", sorted(p.name for p in IMG.glob("gemm-*.png")))


if __name__ == "__main__":
    main()
