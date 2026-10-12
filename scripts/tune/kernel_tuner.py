"""L2 kernel-config tuner for IREE SPIR-V on Adreno (Turnip), scored on the phone.

  kernel_tuner.py profile --model-dir models/llama32-3b-q4-tune --function decode_l1024 [flags...]
  kernel_tuner.py tune    --model-dir ... --function decode_l1024 [--top 6] [flags...]
  kernel_tuner.py apply   --model-dir ...   (recompile with tuned configs substituted, check tokens)

profile: compile with --iree-hal-dump-executable-{configurations,benchmarks}-to, count how often
         each executable is dispatched per call of --function (from stream IR), compile + time each
         dispatch benchmark on the GPU with iree-benchmark-module -> ranked hot list.
tune:    for the hottest executables, generate candidate (translation_info, lowering_config)
         pairs, rewrite the benchmark file, compile, time on device; keep the fastest. Writes the
         winning configured executables to <model-dir>/tune/tuned_cfg/ (same file names as the
         dump), consumed by --iree-hal-substitute-executable-configurations-from.
apply:   recompile with the substitution dir, run the runner, require identical tokens.
All measurements -> <model-dir>/tune/kernels.jsonl
"""

import argparse
import itertools
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
VENV = ROOT / ".venv-iree312"
DEV_DIR = "/data/local/tmp/iree/ktune"
BENCH_TOOL = "/data/local/tmp/iree312/iree-benchmark-module"  # IREE 3.12 build
ENV = "LD_LIBRARY_PATH=/data/local/tmp/turnip"


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def compile_model(a, extra, out_vmfb, dumps=True):
    env = dict(os.environ, IREE_VERSION="3.12", OUT=str(out_vmfb))
    t = a.model_dir / "tune"
    fl = list(a.flags) + list(extra)
    if dumps:
        for d in ("cfg", "bench"):
            shutil.rmtree(t / d, ignore_errors=True)
        fl += [f"--iree-hal-dump-executable-configurations-to={t / 'cfg'}",
               f"--iree-hal-dump-executable-benchmarks-to={t / 'bench'}"]
    r = sh([str(ROOT / "scripts/compile.sh"), str(a.model_dir), "turnip710", *fl], env=env)
    if r.returncode:
        raise SystemExit(f"compile failed:\n{r.stderr[-2000:]}")


def dispatch_counts(a, function):
    """Executable symbol -> number of dispatches per call of `function` (from stream IR)."""
    stream = a.model_dir / "tune" / "stream.mlir"
    tgt = (ROOT / "targets/turnip710.target").read_text().replace("\n", "")
    src = (a.model_dir / "model.mlir").read_text()
    src = re.sub(r"^module @module \{", f"module @module attributes {{hal.device.targets = [{tgt}]}} {{", src, count=1)
    r = sh([str(VENV / "bin/iree-compile"), "-", "--iree-input-type=torch", "--iree-input-demote-i64-to-i32",
            *a.flags, "--compile-to=stream", "-o", str(stream)], input=src)
    if r.returncode:
        raise SystemExit(r.stderr[-2000:])
    text = stream.read_text()
    m = re.search(rf"util\.func public @{re.escape(function)}\$async\(.*?\n  \}}\n", text, re.S)
    body = m.group(0)
    counts = {}
    for exe in re.findall(r"stream\.cmd\.dispatch @([^:\s]+)::", body):
        counts[exe] = counts.get(exe, 0) + 1
    return counts


def bench_file_for(a, exe):
    files = list((a.model_dir / "tune" / "bench").glob(f"*{exe}_vulkan_spirv_fb_benchmark.mlir"))
    return files[0] if files else None


def time_benchmark(a, mlir_text, tag):
    """Compile a dispatch benchmark module, run it on the GPU, return mean µs per dispatch."""
    t = a.model_dir / "tune" / "work"
    t.mkdir(parents=True, exist_ok=True)
    src, vmfb = t / f"{tag}.mlir", t / f"{tag}.vmfb"
    src.write_text(mlir_text)
    r = sh([str(VENV / "bin/iree-compile"), str(src), "-o", str(vmfb)])
    if r.returncode:
        return None, "compile: " + (re.findall(r"error: (.*)", r.stderr) or [r.stderr[-200:]])[0][:200]
    sh(["adb", "push", str(vmfb), f"{DEV_DIR}/{tag}.vmfb"])
    # --batch_size repeats the dispatch inside one submission; the reported time is per item
    # (per dispatch), so submission latency (~1-3 ms on this phone) is amortised away.
    out = sh(["adb", "shell", f"cd {DEV_DIR} && {ENV} {BENCH_TOOL} --device=vulkan --module={tag}.vmfb "
                              f"--batch_size={a.batch} --benchmark_repetitions={a.reps} "
                              f"--benchmark_min_warmup_time=0.1 --benchmark_report_aggregates_only=true"]).stdout
    m = re.search(r"_median\s+([\d.]+) (us|ms|ns)", out)
    if not m:
        return None, "run: " + out[-300:]
    return float(m.group(1)) * {"us": 1, "ms": 1e3, "ns": 1e-3}[m.group(2)], ""


def profile(a):
    sh(["adb", "shell", f"mkdir -p {DEV_DIR}"])
    compile_model(a, [], a.model_dir / "tune" / "profile.vmfb")
    counts = dispatch_counts(a, a.function)
    rows = []
    for exe, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        bf = bench_file_for(a, exe)
        if not bf:
            continue
        us, err = time_benchmark(a, bf.read_text(), "prof")
        rows.append(dict(exe=exe, calls=n, us=us, total_us=(us or 0) * n, err=err, file=bf.name))
        print(f"  {exe[-60:]:60s} x{n:3d}  {us if us is not None else float('nan'):9.1f} us  {err}", flush=True)
    rows.sort(key=lambda r: -r["total_us"])
    total = sum(r["total_us"] for r in rows)
    (a.model_dir / "tune" / f"profile_{a.function}.json").write_text(json.dumps(rows, indent=1))
    print(f"\n{a.function}: sum of dispatch time {total / 1e3:.1f} ms per call")
    acc = 0
    for r in rows[:15]:
        acc += r["total_us"]
        print(f"  {r['total_us'] / 1e3:7.2f} ms {100 * r['total_us'] / total:5.1f}% (cum {100 * acc / total:5.1f}%)  "
              f"x{r['calls']:3d} {r['exe'][-70:]}")
    return rows


# ---------------------------------------------------------------------------------------------
# Candidate generation. Configs are rewritten textually on the dispatch root op.
# ---------------------------------------------------------------------------------------------
TI_RE = re.compile(r"translation_info = #iree_codegen\.translation_info<[^\n]*?>(?=[,}])")
LC_RE = re.compile(r"lowering_config = #iree_codegen\.lowering_config<tile_sizes = \[[^\n]*?\]>")


def current_config(text):
    ti = TI_RE.search(text)
    lc = LC_RE.search(text)
    return (ti.group(0) if ti else None), (lc.group(0) if lc else None)


def loop_bounds(text):
    m = re.search(r"_(matvec_like|matmul_like|matmul|batch_matmul|reduction|contract)_([0-9x]+)_", text)
    return [int(x) for x in m.group(2).split("x")] if m else None


def divisors(n, cands):
    return [c for c in cands if c <= n and n % c == 0]


def candidates(text):
    """Yield (name, translation_info, lowering_config) candidates for the dispatch root op.

    SubgroupReduce (decode GEMV / reductions): tiles = [[parallel...], [0.., reduction...]].
      * outputs per workgroup: tile the last parallel dim (rows) and, for the fused int4 GEMV,
        the nibble dim (2) -> the activation vector is reused across outputs and each packed byte
        is read once.
      * outer reduction tile (how much K a workgroup consumes per iteration) and threads per
        workgroup (multiples of the 64-wide subgroup).
    BaseVectorize (prefill GEMM / attention): [[wg tile], [thread tile], [reduction tile]].
    Pruned by: tile divisibility, <= 1024 threads, <= 16 reduction elements per thread per step.
    """
    ti, lc = current_config(text)
    if not ti or not lc:
        return
    pipe = re.search(r"spirv_pipeline<(\w+)>", ti) or re.search(r"pipeline = (\w+)", ti)
    pipe = pipe.group(1) if pipe else ""
    tiles = json.loads(re.search(r"tile_sizes = (\[.*\])", lc).group(1))
    bounds = loop_bounds(text)
    if "SubgroupReduce" in pipe and bounds and len(tiles) == 2:
        par, red = tiles
        npar = len(par)
        red_dims = [i for i, t in enumerate(red) if t]
        if not red_dims:
            return
        outer = red_dims[0]
        inner = 1
        for i in red_dims[1:]:
            inner *= red[i]
        rows_opts = divisors(bounds[npar - 1], [1, 2, 4, 8])
        h_opts = [1, bounds[0]] if npar >= 2 and bounds[0] <= 4 else [par[0]]
        for h, rows, kt, wg in itertools.product(h_opts, rows_opts, divisors(bounds[outer], [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]),
                                                 [64, 128, 256, 512]):
            per_step = kt * inner
            if per_step % wg or per_step // wg > 16:
                continue
            new_par = list(par)
            new_par[npar - 1] = rows
            if npar >= 2:
                new_par[0] = h
            new_red = list(red)
            new_red[outer] = kt
            new_lc = f"lowering_config = #iree_codegen.lowering_config<tile_sizes = {json.dumps([new_par, new_red])}>"
            new_ti = re.sub(r"workgroup_size = \[\d+, \d+, \d+\]", f"workgroup_size = [{wg}, 1, 1]", ti)
            yield f"sgr_h{h}_rows{rows}_k{kt}_wg{wg}", new_ti, new_lc
        # Cross-pipeline: BaseVectorize for the same GEMV. Each thread owns `nt` outputs and walks
        # the reduction with vector loads, so the activation is reused across a thread's outputs
        # (SubgroupReduce is limited to one output per workgroup).
        base_ti = re.sub(r"spirv_pipeline<\w+>", "spirv_pipeline<BaseVectorize>", ti)
        for h, nt, wx, gt, kt in itertools.product(h_opts, [1, 2, 4, 8], [32, 64, 128], [1, 2, 4],
                                                   divisors(inner, [4, 8, 16, 32])):
            nw = wx * nt
            if bounds[npar - 1] % nw or bounds[outer] % gt:
                continue
            wg_t = [0] * len(red)
            th_t = [0] * len(red)
            rd_t = [0] * len(red)
            wg_t[npar - 1], th_t[npar - 1] = nw, nt
            if npar >= 2:
                wg_t[0], th_t[0] = h, h
            rd_t[outer] = gt
            for i in red_dims[1:]:
                rd_t[i] = kt if i == red_dims[-1] else red[i]
            new_lc = f"lowering_config = #iree_codegen.lowering_config<tile_sizes = {json.dumps([wg_t, th_t, rd_t])}>"
            new_ti = re.sub(r"workgroup_size = \[\d+, \d+, \d+\]", f"workgroup_size = [{wx}, 1, 1]", base_ti)
            yield f"bv_h{h}_nt{nt}_wx{wx}_g{gt}_k{kt}", new_ti, new_lc
    elif "BaseVectorize" in pipe and len(tiles) == 3:
        wgt, tt, rt = tiles
        nd = len(wgt)
        for tm, tn, tk, wx, wy in itertools.product([1, 2, 4], [4, 8], [4, 8, 16, 32], [16, 32, 64], [1, 2, 4]):
            if wx * wy > 256:
                continue
            new_wg = list(wgt)
            new_tt = list(tt)
            new_rt = list(rt)
            new_wg[nd - 2:] = [wy * tm, wx * tn]
            new_tt[nd - 2:] = [tm, tn]
            red_i = max(i for i, t in enumerate(rt) if t) if any(rt) else None
            if red_i is not None:
                new_rt[red_i] = tk
            lcs = json.dumps([new_wg, new_tt, new_rt])
            new_lc = f"lowering_config = #iree_codegen.lowering_config<tile_sizes = {lcs}>"
            new_ti = re.sub(r"workgroup_size = \[\d+, \d+, \d+\]", f"workgroup_size = [{wx}, {wy}, 1]", ti)
            yield f"bv_m{tm}n{tn}k{tk}_wg{wx}x{wy}", new_ti, new_lc


def rewrite(text, new_ti, new_lc):
    ti, lc = current_config(text)
    return text.replace(ti, new_ti).replace(lc, new_lc)


def tune(a):
    t = a.model_dir / "tune"
    rows = json.loads((t / f"profile_{a.function}.json").read_text())
    tuned = t / "tuned_cfg"
    tuned.mkdir(exist_ok=True)
    log = open(t / "kernels.jsonl", "a")
    for r in [x for x in rows if x["us"]][: a.top]:
        bf = t / "bench" / r["file"]
        base_text = bf.read_text()
        base_us = r["us"]
        best = ("baseline", None, None, base_us)
        tried = 0
        print(f"\n{r['exe']}  x{r['calls']}  baseline {base_us:.1f} us  {current_config(base_text)[0][:110]}")
        for name, ti, lc in candidates(base_text):
            if tried >= a.max_candidates:
                break
            tried += 1
            us, err = time_benchmark(a, rewrite(base_text, ti, lc), "cand")
            log.write(json.dumps(dict(exe=r["exe"], cand=name, us=us, err=err, base_us=base_us)) + "\n")
            log.flush()
            mark = ""
            if us and us < best[3]:
                best, mark = (name, ti, lc, us), "  <== best"
            print(f"    {name:28s} {us if us else float('nan'):9.1f} us  {err[:80]}{mark}", flush=True)
        if best[1] and best[3] < base_us * (1 - a.min_gain):
            # Write the configured executable (from the configuration dump) with the new config.
            cfgs = list((t / "cfg").glob(f"*{r['exe']}*.mlir"))
            for cf in cfgs:
                ctext = cf.read_text()
                if current_config(ctext)[0]:
                    (tuned / cf.name).write_text(rewrite(ctext, best[1], best[2]))
            print(f"  -> {best[0]}: {base_us:.1f} -> {best[3]:.1f} us ({base_us / best[3]:.2f}x), "
                  f"saves {(base_us - best[3]) * r['calls'] / 1e3:.2f} ms per {a.function}")


def tune_files(a):
    """Tune standalone dispatch benchmark files (no model compile needed)."""
    log = open(a.model_dir / "tune" / "kernels.jsonl", "a")
    results = {}
    for bf in a.files:
        text = pathlib.Path(bf).read_text()
        base_us, err = time_benchmark(a, text, "base")
        print(f"\n{pathlib.Path(bf).name}\n  baseline {base_us} us {err}  {current_config(text)}", flush=True)
        best = ("baseline", None, None, base_us)
        n = 0
        for name, ti, lc in candidates(text):
            if a.only and not name.startswith(a.only):
                continue
            if n >= a.max_candidates:
                break
            n += 1
            us, err = time_benchmark(a, rewrite(text, ti, lc), "cand")
            log.write(json.dumps(dict(file=str(bf), cand=name, us=us, err=err, base_us=base_us)) + "\n")
            log.flush()
            mark = ""
            if us and (best[3] is None or us < best[3]):
                best, mark = (name, ti, lc, us), "  <== best"
            if mark or a.verbose:
                print(f"    {name:32s} {us if us else float('nan'):9.1f} us {err[:90]}{mark}", flush=True)
        results[str(bf)] = dict(best=best[0], us=best[3], base_us=base_us, ti=best[1], lc=best[2])
        if base_us and best[3]:
            print(f"  BEST {best[0]}: {base_us:.1f} -> {best[3]:.1f} us ({base_us / best[3]:.2f}x) after {n} candidates")
    (a.model_dir / "tune" / "tune_files_result.json").write_text(json.dumps(results, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["profile", "tune", "tune-files"])
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--only", default="", help="only candidates whose name starts with this")
    ap.add_argument("--model-dir", type=pathlib.Path, default=ROOT / "models/llama32-3b-q4-tune")
    ap.add_argument("--function", default="decode_l1024")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--max-candidates", type=int, default=40)
    ap.add_argument("--min-gain", type=float, default=0.03)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--batch", type=int, default=50, help="dispatches per submission when timing")
    ap.add_argument("flags", nargs="*", default=["--iree-opt-const-expr-hoisting=false"])
    a = ap.parse_args()
    (a.model_dir / "tune").mkdir(exist_ok=True)
    {"profile": profile, "tune": tune, "tune-files": tune_files}[a.cmd](a)


if __name__ == "__main__":
    main()
