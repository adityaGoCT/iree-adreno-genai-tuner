"""Flag-level tuner: greedy coordinate search over IREE compile / scheduling flags, scored by
real tok/s on the phone (Turnip + IREE 3.12 + llm-runner).

  flag_tuner.py --model-dir models/llama32-3b-q4 [--ctx 1024] [--new 32] [--metric decode]

Each candidate = base flags + a set of toggles. For each candidate: compile (turnip710), push the
.vmfb, run the runner once (warm-up excluded by the runner), record prefill/decode tok/s.
Greedy: try each remaining toggle on top of the current best; keep it if the metric improves by
> --min-gain; repeat until nothing helps. Every result is appended to <model-dir>/tune/flags.jsonl.
"""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/llm"))
DEV = "/data/local/tmp/iree/llama32-3b-q4"
RUNNER = "/data/local/tmp/iree312/llm-runner"

BASE = ["--iree-opt-const-expr-hoisting=false"]  # keep int4 weights packed in memory
TOGGLES = {
    "memoize": ["--iree-hal-memoization=true"],
    "indirect_cb": ["--iree-hal-indirect-command-buffers=true"],
    "hfuse": ["--iree-dispatch-creation-enable-fuse-horizontal-contractions=true"],
    "aggr_fuse": ["--iree-dispatch-creation-enable-aggressive-fusion=true"],
    "O3": ["--iree-opt-level=O3"],
    "concurrency": ["--iree-stream-partitioning-favor=max-concurrency"],
    "opt_bindings": ["--iree-scheduling-optimize-bindings=true"],
}


def flags_for(toggles):
    return BASE + [f for t in sorted(toggles) for f in TOGGLES[t]]


def evaluate(a, meta, toggles, prompt_name):
    fl = flags_for(toggles)
    key = hashlib.sha1(" ".join(fl).encode()).hexdigest()[:8]
    vmfb = a.model_dir / "tune" / f"cand_{key}.vmfb"
    t0 = time.time()
    env = dict(os.environ, IREE_VERSION="3.12", OUT=str(vmfb))
    r = subprocess.run([str(ROOT / "scripts/compile.sh"), str(a.model_dir), "turnip710", *fl],
                       env=env, capture_output=True, text=True)
    if r.returncode:
        return dict(toggles=sorted(toggles), ok=False, error=r.stderr[-500:])
    compile_s = time.time() - t0
    subprocess.run(["adb", "push", str(vmfb), f"{DEV}/cand.vmfb"], check=True, capture_output=True)
    res = []
    for _ in range(a.repeats):
        cmd = (f"cd {DEV} && LD_LIBRARY_PATH=/data/local/tmp/turnip {RUNNER} --device=vulkan --module=cand.vmfb "
               f"--parameters=model=weights.irpa --prompt={prompt_name} --max_new_tokens={a.new} --chunk={meta['chunk']} "
               f"--buckets={','.join(map(str, meta['buckets']))} --n_layers={meta['n_layers']} "
               f"--n_kv_heads={meta['n_kv_heads']} --head_dim={meta['head_dim']} --max_ctx={meta['max_ctx']}")
        out = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True).stdout
        line = next((l for l in out.splitlines() if l.startswith("{")), None)
        if not line:
            return dict(toggles=sorted(toggles), ok=False, error=out[-500:])
        res.append(json.loads(line))
    best = max(res, key=lambda x: x[f"{a.metric}_tok_s"])
    return dict(toggles=sorted(toggles), ok=True, flags=fl, compile_s=round(compile_s),
                prefill_tok_s=best["prefill_tok_s"], decode_tok_s=best["decode_tok_s"],
                tokens=best["tokens"][:8], vmfb=vmfb.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=pathlib.Path, default=ROOT / "models/llama32-3b-q4")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--metric", choices=["decode", "prefill"], default="decode")
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--start", nargs="*", default=[])
    a = ap.parse_args()
    meta = json.loads((a.model_dir / "meta.json").read_text())
    (a.model_dir / "tune").mkdir(exist_ok=True)
    log = a.model_dir / "tune" / "flags.jsonl"

    from transformers import AutoTokenizer
    from bench import build_prompt
    tok = AutoTokenizer.from_pretrained(pathlib.Path(open(ROOT / "models/LLAMA_SNAPSHOT.txt").read().strip()))
    pf = a.model_dir / "tune" / f"prompt_tune_{a.ctx}.txt"
    pf.write_text(" ".join(map(str, build_prompt(tok, "prose", a.ctx - a.new))))
    subprocess.run(["adb", "push", str(pf), f"{DEV}/{pf.name}"], check=True, capture_output=True)

    def run(toggles):
        r = evaluate(a, meta, toggles, pf.name)
        r.update(metric=a.metric, ctx=a.ctx)
        with open(log, "a") as f:
            f.write(json.dumps(r) + "\n")
        s = (f"prefill {r['prefill_tok_s']:7.1f} tok/s  decode {r['decode_tok_s']:5.2f} tok/s  compile {r['compile_s']}s"
             if r["ok"] else f"FAILED {r['error'][-200:]!r}")
        print(f"  {'+'.join(r['toggles']) or 'baseline':55s} {s}", flush=True)
        return r

    best = run(set(a.start))
    assert best["ok"], best
    ref_tokens = best["tokens"]
    remaining = set(TOGGLES) - set(a.start)
    while remaining:
        trials = []
        for t in sorted(remaining):
            r = run(set(best["toggles"]) | {t})
            if r["ok"] and r["tokens"] != ref_tokens:
                print(f"    ! {t}: output tokens changed {r['tokens']} vs {ref_tokens} -> rejected")
                continue
            if r["ok"]:
                trials.append((r[f"{a.metric}_tok_s"], t, r))
        if not trials:
            break
        score, t, r = max(trials)
        if score < best[f"{a.metric}_tok_s"] * (1 + a.min_gain):
            break
        print(f"  ==> keep {t}: {best[f'{a.metric}_tok_s']:.2f} -> {score:.2f} tok/s")
        best = r
        remaining.discard(t)
    print(f"BEST: {'+'.join(best['toggles']) or 'baseline'}  flags={best['flags']}  "
          f"prefill {best['prefill_tok_s']:.1f} tok/s  decode {best['decode_tok_s']:.2f} tok/s")
    (a.model_dir / "tune" / f"best_flags_{a.metric}.json").write_text(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
