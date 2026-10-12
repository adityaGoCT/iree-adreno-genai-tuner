"""Turn a tuning run into the numbers for the report.

  analyze.py summary --tag run1          -> runs/run1/summary.json (+ printed tables)
  analyze.py ablate  --tag run1          -> runs/run1/ablation.json (GPU: knock out one primitive at a time)
  analyze.py final   --tag run1          -> runs/run1/final.json + final_table.json (GPU: interleaved re-measure)
  analyze.py rank    --tag run1          -> runs/run1/rank.json (cost model, leave-one-shape-out ranking quality)
  analyze.py verifier --tag run1         -> runs/run1/verifier.json (precision, rejection reasons, GPU failures)
"""

import argparse
import collections
import dataclasses
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cost_model import ROOF  # noqa: E402
from schedule import Schedule, schedule_text  # noqa: E402
from shapes import BY_ID, SHAPES  # noqa: E402
from verifier import verify  # noqa: E402

RUNS = pathlib.Path(__file__).parent / "runs"


def load(tag):
    recs = [json.loads(l) for l in (RUNS / tag / "log.jsonl").read_text().splitlines()]
    return [r for r in recs if r.get("kind") != "round"], [r for r in recs if r.get("kind") == "round"]


def attainable(s, acc="f16"):
    """Roofline bound for the shape: min(peak FMA, arithmetic intensity x DRAM bandwidth)."""
    peak = ROOF["f16_gflops"] if acc == "f16" else ROOF["f32_gflops"]
    ai = s.flops / s.min_bytes
    return min(peak, ai * ROOF["read_gbs"])


def summary(tag, ext=None):
    """ext: a continuation run (search.py --seed-from tag); its model-guided measurements become the
    method "ext" whose curve continues the model curve."""
    cands, rounds = load(tag)
    if ext:
        ec, _ = load(ext)
        cands = cands + [dict(r, method="ext") for r in ec if r.get("method") == "model"]
    # IREE's kernel timed in our harness (kernel_results.json); IREE runtime timing (results.json) as a cross-check
    rd = lambda f: json.loads((RUNS / "iree" / f).read_text()) if (RUNS / "iree" / f).exists() else {}  # noqa: E731
    iree, iree_rt = rd("kernel_results.json"), rd("results.json")
    out = {}
    for s in SHAPES:
        by = collections.defaultdict(list)
        for r in cands:
            if r["shape"] == s.id and r.get("stage") == "gpu":
                by[r["method"]].append(r)
        if not by:
            continue
        row = {"shape": dataclasses.asdict(s), "attainable_f16": attainable(s, "f16"), "attainable_f32": attainable(s, "f32")}
        for m, rs in by.items():
            if m == "ext" and "model" in by:  # continuation: best-so-far includes the seed population
                rs = by["model"] + rs
            ok = [r for r in rs if r["ok"]]
            best = min(ok, key=lambda r: r["us"]) if ok else None
            curve, b = [], None
            for r in rs:  # in measurement order
                if r["ok"] and (b is None or r["us"] < b):
                    b = r["us"]
                curve.append(s.flops / (b * 1e3) if b else 0)
            row[m] = dict(n=len(rs), n_ok=len(ok), best_us=best and best["us"],
                          best_gflops=best and s.flops / (best["us"] * 1e3), best_sched=best and best["sched"],
                          curve=curve, gpu_failures=[r.get("error") or f"err={r.get('err')}" for r in rs if not r["ok"]])
        if s.id in iree:
            row["iree"] = {k: iree[s.id].get(k) for k in ("us", "gflops", "ok", "err", "grid")}
        if s.id in iree_rt:
            row["iree_runtime"] = iree_rt[s.id]
        rr = [r for r in rounds if r["shape"] == s.id and r["method"] == "model"]
        row["rounds"] = [{k: r[k] for k in ("round", "n", "rho", "model", "valid", "proposed")} for r in rr]
        rej = rr[-1]["rejected"] if rr else {}
        row["rejected"] = rej
        out[s.id] = row
        nv = row.get("naive", {}).get("best_gflops") or 0
        print(f"{s.id} {s.name:11s} naive {nv:6.1f}  random {row.get('random', {}).get('best_gflops') or 0:6.1f}  "
              f"model {row.get('model', {}).get('best_gflops') or 0:6.1f}  "
              f"+f16p {row.get('ext', {}).get('best_gflops') or 0:6.1f}  IREE {row.get('iree', {}).get('gflops') or 0:6.1f}  "
              f"attainable(f16) {row['attainable_f16']:6.1f} GFLOP/s")
    (RUNS / tag / "summary.json").write_text(json.dumps(out, indent=1))
    return out


def rank(tag, k=16):
    """Cost-model ranking quality on unbiased data: for each shape, train the learned model on every
    *other* shape's measurements (leave-one-shape-out) and rank this shape's random-search
    measurements (uniform samples of the verified space). Reports Spearman rho for the analytic and
    learned models, and top-k quality: the best GFLOP/s among the k candidates the model would have
    sent to the GPU, as a fraction of the best in the whole pool."""
    from scipy.stats import spearmanr
    from cost_model import Learned, analytic_us
    cands, _ = load(tag)
    ok = [r for r in cands if r.get("stage") == "gpu" and r["ok"] and r["method"] in ("model", "random")]
    out = {}
    for s in SHAPES:
        test = [r for r in ok if r["shape"] == s.id and r["method"] == "random"]
        if len(test) < 8:
            continue
        m = Learned(min_samples=1)
        for r in ok:
            if r["shape"] != s.id:
                m.add(BY_ID[r["shape"]], Schedule(**r["sched"]), r["us"])
        m.fit()
        scs = [Schedule(**r["sched"]) for r in test]
        meas = [r["us"] for r in test]
        row = {"n": len(test)}
        for name, pred in (("analytic", [analytic_us(s, sc) for sc in scs]), ("learned", list(m.predict_us(s, scs)))):
            top = sorted(range(len(test)), key=lambda i: pred[i])[:k]
            row[name] = dict(rho=float(spearmanr(pred, meas).correlation),
                             topk_frac=min(meas) / min(meas[i] for i in top))
        out[s.id] = row
        print(f"{s.id}: n={row['n']:3d}  rho analytic {row['analytic']['rho']:+.2f} learned {row['learned']['rho']:+.2f}  "
              f"top-{k} reaches {row['analytic']['topk_frac'] * 100:5.1f}% / {row['learned']['topk_frac'] * 100:5.1f}% of pool best")
    (RUNS / tag / "rank.json").write_text(json.dumps(out, indent=1))
    return out


def verifier_stats(tag):
    """Verifier precision (verified candidates that were also correct on the GPU) and rejections."""
    cands, rounds = load(tag)
    gpu = [r for r in cands if r.get("stage") == "gpu" and r["method"] in ("model", "random", "naive")]
    comp = [r for r in cands if r.get("stage") == "compile" and r["method"] in ("model", "random", "naive")]
    wrong = [r for r in gpu if not r["ok"]]
    rej = collections.Counter()
    for sid in {r["shape"] for r in rounds}:
        for meth in ("model", "random"):
            rr = [r for r in rounds if r["shape"] == sid and r["method"] == meth]
            if rr:
                rej.update(rr[-1]["rejected"])
    prop = sum(r["proposed"] for r in rounds)
    valid = sum(r["valid"] for r in rounds)
    out = dict(measured=len(gpu), gpu_failures=len(wrong), compile_failures=len(comp),
               precision=(len(gpu) - len(wrong)) / max(1, len(gpu) + len(comp)),
               proposed=prop, valid=valid, rejected=dict(rej.most_common()),
               failures=[(r["shape"], Schedule(**r["sched"]).short(), r.get("err"), r.get("error")) for r in wrong])
    print(json.dumps({k: v for k, v in out.items() if k != "failures"}, indent=1))
    for f in out["failures"]:
        print("  GPU-FAIL", f)
    (RUNS / tag / "verifier.json").write_text(json.dumps(out, indent=1))
    return out


def best_row(row):
    """The best tuned result for a shape: the continuation (ext) if present, else model-guided."""
    return row["ext"] if row.get("ext") and row["ext"].get("best_us") else row["model"]


def ablations(s, sc: Schedule):
    """One primitive removed (or switched) at a time from the best schedule."""
    k = dataclasses.asdict(sc)
    v = {}
    if sc.KT > 1:
        v["no cross-thread reduction (KT=1)"] = dict(k, KT=1)
    if sc.TM * sc.TN > 1:
        # single-level tiling: one output per thread, same workgroup tile when legal, else shrink it
        bm, bn = sc.BM, sc.BN
        while (bm * bn * sc.KT) > 1024 and bm * bn > 1:
            if bn >= bm and bn > 1:
                bn //= 2
            else:
                bm //= 2
        v["no register tiling (TM=TN=1)"] = dict(k, TM=1, TN=1, VTM=1, VTN=1, BM=bm, BN=bn)
    if sc.VEC > 1:
        v["no vectorization (VEC=1)"] = dict(k, VEC=1)
    if s.la == "T" or s.lb == "T":
        v["flip inlining of transposes"] = dict(k, INLINE_A=1 - sc.INLINE_A if s.la == "T" else 1,
                                                 INLINE_B=1 - sc.INLINE_B if s.lb == "T" else 1)
    if sc.VTM * sc.VTN > 1:
        v["no vthreads (VT=1x1)"] = dict(k, VTM=1, VTN=1)
    else:
        if sc.TN >= 2:
            v["add vthreads (VTN=2)"] = dict(k, VTN=2)
    v["flip lane binding (BIND)"] = dict(k, BIND="m" if sc.BIND == "n" else "n")
    v["other wave size (SG)"] = dict(k, SG=128 if sc.SG == 64 else 64)
    if sc.SMEM_A or sc.SMEM_B:
        v["no shared-memory staging"] = dict(k, SMEM_A=0, SMEM_B=0, PAD=0)
    if sc.ACC in ("f16", "f16p"):
        v["accumulate in f32"] = dict(k, ACC="f32")
    return {name: Schedule(**kw) for name, kw in v.items()}


def ablate(tag, driver="turnip", reps=3):
    """Knock out one primitive at a time from the best schedule. Every variant is measured in `reps`
    interleaved rounds (order shuffled each round, fastest repetition per round, median over rounds)
    so GPU clock changes hit all variants alike."""
    import random
    from search import measure, Log
    summ = json.loads((RUNS / tag / "summary.json").read_text())
    log = Log(RUNS / tag / "ablation_log.jsonl")
    rng = random.Random(0)
    out = {}
    for sid, row in summ.items():
        s = BY_ID[sid]
        best = Schedule(**best_row(row)["best_sched"])
        variants = {"best": best, **ablations(s, best)}
        legal = {n: sc for n, sc in variants.items() if verify(s, sc, driver).ok}
        times = collections.defaultdict(list)
        for rep in range(reps):
            order = list(legal.items())
            rng.shuffle(order)
            res = measure(s, [sc for _, sc in order], RUNS / tag, f"abl{rep}", driver, log,
                          dict(method="ablation", round=rep, model="none"), metric="us_min")
            for n, sc in order:
                if res.get(sc.key()):
                    times[n].append(res[sc.key()])
        out[sid] = {}
        for n, sc in variants.items():
            v = sorted(times.get(n, []))
            us = v[len(v) // 2] if v else None
            out[sid][n] = dict(us=us, samples=v, gflops=s.flops / (us * 1e3) if us else None,
                               legal=n in legal, reasons=None if n in legal else verify(s, sc, driver).reasons)
        b = out[sid]["best"]["us"]
        print(sid, " | ".join(f"{n}: {(v['us'] / b if v['us'] and b else float('nan')):.2f}x" for n, v in out[sid].items()
                              if n != "best"), flush=True)
    (RUNS / tag / "ablation.json").write_text(json.dumps(out, indent=1))


def final(tag, driver="turnip", reps=5):
    """Interleaved re-measurement of naive / IREE default / random-best / model-best (/ +f16p):
    `reps` rounds, candidate order shuffled each round, fastest repetition per round, median over
    rounds -- so GPU clock changes (DVFS; clocks cannot be pinned without root) hit all alike."""
    import random
    import device
    from iree_baseline import iree_candidate
    from search import measure, Log, naive
    summ = json.loads((RUNS / tag / "summary.json").read_text())
    log = Log(RUNS / tag / "final_log.jsonl")
    rng = random.Random(1)
    out = {sid: collections.defaultdict(list) for sid in summ}
    for rep in range(reps):
        for sid, row in summ.items():
            s = BY_ID[sid]
            items = {"naive": naive(s), "random": Schedule(**row["random"]["best_sched"]),
                     "model": Schedule(**row["model"]["best_sched"])}
            if row.get("ext") and row["ext"]["best_sched"] != row["model"]["best_sched"]:
                items["model+f16p"] = Schedule(**row["ext"]["best_sched"])
            order = list(items.items())
            rng.shuffle(order)
            iree_first = rng.random() < 0.5
            cand, _ = iree_candidate(s)

            def run_iree():
                r = device.run_batch([cand], RUNS / tag, f"{sid}_final_iree{rep}", driver=driver).get(cand["name"], {})
                if r.get("ok"):
                    out[sid]["iree"].append(r["us_min"])
            if iree_first:
                run_iree()
            res = measure(s, [sc for _, sc in order], RUNS / tag, f"final{rep}", driver, log,
                          dict(method="final", round=rep, model="none"), metric="us_min")
            for n, sc in order:
                if res.get(sc.key()):
                    out[sid][n].append(res[sc.key()])
            if not iree_first:
                run_iree()
        print(f"final round {rep} done", flush=True)
    fin = {}
    for sid, d in out.items():
        s = BY_ID[sid]
        fin[sid] = {n: dict(us=sorted(v)[len(v) // 2], gflops=s.flops / (sorted(v)[len(v) // 2] * 1e3), samples=v)
                    for n, v in d.items() if v}
        print(sid, {n: round(x["gflops"], 1) for n, x in fin[sid].items()})
    (RUNS / tag / "final.json").write_text(json.dumps(fin, indent=1))


def final_table(tag):
    """Per shape: each contender's best round (the highest GPU clock level it reached; slow kernels
    repeat to within ~1%, fast ones land on discrete DVFS levels), the tuned kernel's range and
    median over rounds, the roofline, and speedups. -> runs/<tag>/final_table.json"""
    import statistics
    fin = json.loads((RUNS / tag / "final.json").read_text())
    summ = json.loads((RUNS / tag / "summary.json").read_text())
    rows = []
    for sid, d in fin.items():
        s = BY_ID[sid]
        g = {k: [s.flops / (u * 1e3) for u in v["samples"]] for k, v in d.items()}
        tk = max((k for k in ("model", "model+f16p") if k in g), key=lambda k: max(g[k]))
        t = g[tk]
        rows.append(dict(s=sid, name=s.name, naive=round(max(g["naive"]), 1), iree=round(max(g["iree"]), 1),
                         random=round(max(g["random"]), 1), tuned=round(max(t), 1), tlo=round(min(t), 1),
                         tmed=round(statistics.median(t), 1), tuned_from=tk, roof=round(summ[sid]["attainable_f16"], 1)))
    gm = lambda xs: statistics.geometric_mean(xs)  # noqa: E731
    for k in ("naive", "iree", "random"):
        xs = [r["tuned"] / r[k] for r in rows]
        print(f"tuned vs {k}: {min(xs):.2f}x to {max(xs):.2f}x, geomean {gm(xs):.2f}x")
    for r in rows:
        print(f"{r['s']} tuned {r['tuned']:6.1f} GFLOP/s [{r['tlo']}..{r['tuned']}, median {r['tmed']}] "
              f"= {100 * r['tuned'] / r['roof']:.1f}% of roofline")
    (RUNS / tag / "final_table.json").write_text(json.dumps(rows, indent=1))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summary", "ablate", "final", "final_table", "show", "rank", "verifier"])
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--ext", default=None, help="continuation run to fold in as method 'ext'")
    a = ap.parse_args()
    if a.cmd == "summary":
        summary(a.tag, a.ext)
    elif a.cmd == "ablate":
        ablate(a.tag)
    elif a.cmd == "final":
        final(a.tag)
        final_table(a.tag)
    elif a.cmd == "final_table":
        final_table(a.tag)
    elif a.cmd == "rank":
        rank(a.tag)
    elif a.cmd == "verifier":
        verifier_stats(a.tag)
    else:
        summ = json.loads((RUNS / a.tag / "summary.json").read_text())
        for sid, row in summ.items():
            print(schedule_text(BY_ID[sid], Schedule(**best_row(row)["best_sched"])), "\n")


if __name__ == "__main__":
    main()
