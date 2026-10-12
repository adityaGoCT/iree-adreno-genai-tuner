"""Correctness fuzz for the code generator: per shape 8 random + 8 shared-memory-forced verified
schedules + 2 fixed ones; every candidate must match the fp32 reference on the GPU.

  fuzz.py --tag fuzz3 --seed 2026          -> runs/<tag>/log.jsonl
Its measurements (random verified schedules, disjoint from tuning runs) also calibrate the
analytic cost model's three constants (see cost_model.py).
"""

import argparse
import collections
import dataclasses
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from schedule import Schedule  # noqa: E402
from search import Log, measure, random_sched  # noqa: E402
from shapes import SHAPES  # noqa: E402
from verifier import verify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fuzz")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--driver", default="turnip")
    a = ap.parse_args()
    out = pathlib.Path(__file__).parent / "runs" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    log, rng, tot = Log(out / "log.jsonl"), random.Random(a.seed), collections.Counter()
    for s in SHAPES:
        picks, smem = [], []
        while len(picks) < 8:
            sc = random_sched(s, rng)
            if verify(s, sc, a.driver).ok:
                picks.append(sc)
        while len(smem) < 8:  # bias towards the riskiest path: shared-memory staging (+ DB)
            k = dataclasses.asdict(random_sched(s, rng))
            k["SMEM_A"] = rng.choice([0, 1])
            k["SMEM_B"] = 1 - k["SMEM_A"] if rng.random() < 0.5 else 1
            k["DB"] = rng.choice([0, 1])
            sc = Schedule(**k)
            if verify(s, sc, a.driver).ok:
                smem.append(sc)
        fixed = [Schedule(BM=32, BN=64, BK=16, TM=4, TN=8, SMEM_A=1, SMEM_B=1, DB=1, VEC=8, PAD=8),
                 Schedule(BM=16, BN=32, BK=32, TM=4, TN=4, VTN=2, KT=2, SMEM_A=1, DB=1, VEC=4, BIND="m")]
        picks += smem + [x for x in fixed if verify(s, x, a.driver).ok]
        res = measure(s, picks, out, a.tag, a.driver, log, dict(method="fuzz", round=0, model="none"))
        ok = sum(1 for sc in picks if res.get(sc.key()))
        tot["ok"] += ok
        tot["n"] += len(picks)
        bad = [sc.short() for sc in picks if not res.get(sc.key())]
        print(f"{s.id}: {ok}/{len(picks)} correct", ("FAILED: " + " | ".join(bad)) if bad else "", flush=True)
    print("TOTAL", dict(tot))


if __name__ == "__main__":
    main()
