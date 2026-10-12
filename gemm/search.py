"""Search driver: propose -> verify -> rank (cost model) -> measure top-k on the GPU -> learn.

  search.py --shapes S1 S2 ... --budget 96 --batch 16 --method model random --tag run1

method=model   Ansor-style evolutionary search: each round builds a pool from random samples plus
               mutations/crossovers of the best measured schedules, keeps only verifier-approved
               candidates, ranks them with the cost model, measures the top `batch` (with 25%
               exploration picks) in one harness call, retrains the model.
method=random  same GPU budget, verifier-approved candidates in random order (baseline).
Every candidate's verdict, prediction and measurement goes to runs/<tag>/log.jsonl.
"""

import argparse
import collections
import dataclasses
import json
import math
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import device  # noqa: E402
from build import build_candidate  # noqa: E402
from cost_model import Learned, analytic_us  # noqa: E402
from schedule import Schedule, kernels  # noqa: E402
from shapes import BY_ID, SHAPES  # noqa: E402
from verifier import verify  # noqa: E402

RUNS = pathlib.Path(__file__).parent / "runs"


def p2(lo, hi):
    out, v = [], lo
    while v <= hi:
        out.append(v)
        v *= 2
    return out


ACC_DOMAIN = ["f32", "f16"]  # --acc extends it (e.g. with f16p, promoted fp16 accumulation)


def domains(s):
    np2 = lambda x: 1 << max(0, math.ceil(math.log2(x)))  # noqa: E731
    return {
        "BM": p2(1, min(128, np2(s.M))),
        "BN": p2(4, min(256, np2(s.N))),
        "BK": [b for b in p2(4, 256) if s.K % b == 0],
        "TM": [1, 2, 4, 8], "TN": [1, 2, 4, 8],
        "VTM": [1, 2], "VTN": [1, 2, 4],
        "KT": [1, 2, 4, 8, 16, 32, 64],
        "RED": ["subgroup", "smem"], "KMAP": ["block", "cyclic"],
        "SMEM_A": [0, 1], "SMEM_B": [0, 1], "DB": [0, 1],
        "VEC": [1, 2, 4, 8], "PAD": [0, 8], "UNROLL": [1, 2, 4],
        "ACC": ACC_DOMAIN,
        "INLINE_A": [1, 0] if s.la == "T" else [1],
        "INLINE_B": [1, 0] if s.lb == "T" else [1],
        "BIND": ["n", "m"], "SG": [64, 128],
    }


def random_sched(s, rng):
    d = domains(s)
    k = {name: rng.choice(vals) for name, vals in d.items()}
    k["TM"] = min(k["TM"], k["BM"])
    k["TN"] = min(k["TN"], k["BN"])
    k["VTM"] = min(k["VTM"], k["TM"])
    k["VTN"] = min(k["VTN"], k["TN"])
    if not (k["SMEM_A"] or k["SMEM_B"]):
        k["PAD"] = 0
        k["DB"] = 0
    return Schedule(**k)


def mutate(s, sc, rng, n=None):
    d = domains(s)
    k = dataclasses.asdict(sc)
    for name in rng.sample(list(d), n or rng.choice([1, 1, 2, 3])):
        k[name] = rng.choice(d[name])
    k["TM"] = min(k["TM"], k["BM"])
    k["TN"] = min(k["TN"], k["BN"])
    k["VTM"] = min(k["VTM"], k["TM"])
    k["VTN"] = min(k["VTN"], k["TN"])
    if not (k["SMEM_A"] or k["SMEM_B"]):
        k["PAD"] = 0
        k["DB"] = 0
    return Schedule(**k)


def _clamp(k):
    k["TM"] = min(k["TM"], k["BM"])
    k["TN"] = min(k["TN"], k["BN"])
    k["VTM"] = min(k["VTM"], k["TM"])
    k["VTN"] = min(k["VTN"], k["TN"])
    if not (k["SMEM_A"] or k["SMEM_B"]):
        k["PAD"] = 0
        k["DB"] = 0
    return Schedule(**k)


def crossover(a, b, rng):
    ka, kb = dataclasses.asdict(a), dataclasses.asdict(b)
    return _clamp({n: (ka if rng.random() < 0.5 else kb)[n] for n in ka})


def naive(s):
    bm = 1 if s.M < 8 else 8
    return Schedule(BM=bm, BN=64 // bm, BK=min(32, s.K), TM=1, TN=1)


class Log:
    def __init__(self, path):
        self.f = open(path, "a")

    def write(self, **kw):
        self.f.write(json.dumps(kw) + "\n")
        self.f.flush()


def measure(s, scheds, out_dir, tag, driver, log, meta, metric="us"):
    """Compile (verifier stage 3) and run a batch on the phone. Returns {key: time or None};
    metric "us" = median of the timed repetitions, "us_min" = fastest one (robust to DVFS dips)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:  # glslc runs in parallel (unrolled kernels compile slowly)
        built = list(ex.map(lambda sc: build_candidate(s, sc, kernels), scheds))
    cands, by_name = [], {}
    for i, (sc, (disp, err)) in enumerate(zip(scheds, built)):
        name = f"{s.id}_{tag}_{i}"
        if err:
            log.write(**meta, shape=s.id, sched=dataclasses.asdict(sc), stage="compile", ok=False, error=err)
            continue
        cands.append({"name": name, "shape": s.id, "sg": sc.SG, "dispatches": disp})
        by_name[name] = sc
    res = device.run_batch(cands, out_dir, f"{s.id}_{tag}", driver=driver) if cands else {}
    out = {}
    for name, sc in by_name.items():
        r = res.get(name, {"ok": False, "error": "missing"})
        good = bool(r.get("ok")) and r.get("us") is not None
        out[sc.key()] = r[metric] if good else None
        log.write(**meta, shape=s.id, sched=dataclasses.asdict(sc), stage="gpu", ok=good, us=r.get("us"),
                  us_min=r.get("us_min"), err=r.get("err"), error=r.get("error"),
                  gflops=(s.flops / (r["us"] * 1e3)) if good else None)
    return out


def tune(s, method, budget, batch, model: Learned, driver, out_dir, log, seed=0, pool=1500, init=None):
    rng = random.Random(seed * 1000 + int(s.id[1:]) + (0 if method == "model" else 7))
    measured = dict(init or {})  # key -> (sched, us or None); init = population carried over from a run
    budget += len(measured)
    rejected = collections.Counter()
    round_i = 0
    best_curve = []
    while len(measured) < budget:
        # ---- propose ----
        props = [random_sched(s, rng) for _ in range(pool)]
        if method == "model":
            good = sorted(((us, sc) for sc, us in measured.values() if us), key=lambda t: t[0])[:12]
            for _ in range(pool if good else 0):
                p = rng.choice(good)[1]
                props.append(mutate(s, p, rng) if rng.random() < 0.75 else crossover(p, rng.choice(good)[1], rng))
        # ---- verify (stages 1-2) ----
        seen, valid = set(measured), []
        for sc in props:
            k = sc.key()
            if k in seen:
                continue
            seen.add(k)
            v = verify(s, sc, driver)
            if v.ok:
                valid.append(sc)
            else:
                rejected.update(v.reasons)
        # ---- rank ----
        take = min(batch, budget - len(measured))
        meta = dict(method=method, round=round_i, model=model.kind if method == "model" else "none")
        if method == "random":
            rng.shuffle(valid)
            pick, preds = valid[:take], [None] * take
        else:
            preds_all = model.predict_us(s, valid)
            order = sorted(range(len(valid)), key=lambda i: preds_all[i])
            n_exploit = max(1, int(take * 0.75))
            exploit = [valid[i] for i in order[:n_exploit]]
            rest = [valid[i] for i in order[n_exploit:]]
            explore = rng.sample(rest, min(len(rest), take - n_exploit))
            pick = exploit + explore
            idx = {sc.key(): i for i, sc in enumerate(valid)}
            preds = [float(preds_all[idx[sc.key()]]) for sc in pick]
        # ---- measure on GPU ----
        t0 = time.time()
        res = measure(s, pick, out_dir, f"{method}{round_i}", driver, log, meta)
        for sc, pr in zip(pick, preds):
            us = res.get(sc.key())
            measured[sc.key()] = (sc, us)
            if us:
                model.add(s, sc, us)
        # ranking quality of the predictions made *before* measuring
        pairs = [(pr, res.get(sc.key())) for sc, pr in zip(pick, preds) if pr and res.get(sc.key())]
        rho = None
        if len(pairs) >= 5:
            from scipy.stats import spearmanr
            rho = float(spearmanr([p for p, _ in pairs], [m for _, m in pairs]).correlation)
        model.fit()
        best = min((us for _, us in measured.values() if us), default=None)
        best_curve.append((len(measured), best))
        log.write(kind="round", shape=s.id, method=method, round=round_i, n=len(measured), best_us=best,
                  rho=rho, valid=len(valid), proposed=len(props), gpu_s=round(time.time() - t0, 1),
                  model=meta["model"], rejected=dict(rejected))
        print(f"  {s.id} {method:6s} round {round_i}: measured {len(measured):3d}  best "
              f"{best and s.flops / (best * 1e3):7.1f} GFLOP/s  rank-rho {rho if rho is None else round(rho, 2)}  "
              f"[{meta['model']}] valid {len(valid)}/{len(props)}", flush=True)
        round_i += 1
    return measured, best_curve, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=[s.id for s in SHAPES])
    ap.add_argument("--method", nargs="+", default=["model", "random"])
    ap.add_argument("--budget", type=int, default=96)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--driver", default="turnip")
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--acc", nargs="+", default=None, help="accumulate dtypes to search (default f32 f16)")
    ap.add_argument("--seed-from", default=None,
                    help="continue from another run: warm-start the cost model with all its measurements "
                         "and start each shape's population from its measured schedules")
    a = ap.parse_args()
    if a.acc:
        ACC_DOMAIN[:] = a.acc
    out = RUNS / a.tag
    out.mkdir(parents=True, exist_ok=True)
    model = Learned()  # shared across shapes: knowledge transfers from earlier shapes
    lp = out / "log.jsonl"
    if lp.exists():  # resume: drop partial records of the shapes being (re)run, warm-start the model
        recs = [json.loads(l) for l in lp.read_text().splitlines()]
        keep = [r for r in recs if r["shape"] not in a.shapes]
        if len(keep) < len(recs):
            (out / "log.jsonl.bak").write_text(lp.read_text())
            lp.write_text("".join(json.dumps(r) + "\n" for r in keep))
        for r in keep:
            if r.get("stage") == "gpu" and r.get("ok") and r.get("method") in ("model", "random"):
                model.add(BY_ID[r["shape"]], Schedule(**r["sched"]), r["us"])
        model.fit()
        print(f"resume: {len(recs) - len(keep)} records of {a.shapes} dropped, model warm-started on "
              f"{len(model.y)} measurements ({model.kind})", flush=True)
    init = collections.defaultdict(dict)
    if a.seed_from:
        for r in (json.loads(l) for l in (RUNS / a.seed_from / "log.jsonl").read_text().splitlines()):
            if r.get("stage") == "gpu" and r.get("method") in ("model", "random"):
                sc = Schedule(**r["sched"])
                if r.get("ok"):
                    model.add(BY_ID[r["shape"]], sc, r["us"])
                if r["method"] == "model":
                    init[r["shape"]][sc.key()] = (sc, r["us"] if r.get("ok") else None)
        model.fit()
        print(f"seeded from {a.seed_from}: model on {len(model.y)} measurements; populations "
              + ", ".join(f"{k}={len(v)}" for k, v in sorted(init.items())), flush=True)
    log = Log(lp)
    device.setup(SHAPES, RUNS / "data")
    for sid in a.shapes:
        s = BY_ID[sid]
        print(f"== {s.id} {s.name} M={s.M} N={s.N} K={s.K} {s.layout}", flush=True)
        if not a.seed_from:
            base = measure(s, [naive(s)], out, "naive", a.driver, log, dict(method="naive", round=0, model="none"))
            nb = next(iter(base.values()))
            print(f"  naive: {nb and s.flops / (nb * 1e3):.1f} GFLOP/s", flush=True)
        for m in a.method:
            tune(s, m, a.budget, a.batch, model, a.driver, out, log, init=init.get(s.id) if m == "model" else None)


if __name__ == "__main__":
    main()
