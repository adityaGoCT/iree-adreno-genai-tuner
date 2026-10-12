"""IREE baseline for the same 8 GEMMs: linalg.generic (fp16 in, fp32 accumulate, fp16 out, layout as
indexing maps) compiled by IREE 3.12 for the Turnip Adreno 710 target; every dispatch IREE emits is
timed on the phone with iree-benchmark-module (batched submissions, GPU time per dispatch), summed.

  iree_baseline.py compile      # on the Mac
  iree_baseline.py run          # on the phone (Turnip + IREE 3.12 runtime): end-to-end dispatch time
  iree_baseline.py kernel       # IREE's SPIR-V kernel in *our* harness: same data, GPU timestamps,
                                # same reference check as the tuner's candidates (kernel vs kernel)
"""

import json
import os
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import device  # noqa: E402
from shapes import SHAPES  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(__file__).parent / "runs/iree"
COMPILE = ROOT / ".venv-iree312/bin/iree-compile"
BENCH = "/data/local/tmp/iree312/iree-benchmark-module"


def mlir(s):
    a_t = f"tensor<{s.M}x{s.K}xf16>" if s.la == "N" else f"tensor<{s.K}x{s.M}xf16>"
    b_t = f"tensor<{s.K}x{s.N}xf16>" if s.lb == "N" else f"tensor<{s.N}x{s.K}xf16>"
    ma = "(d0, d2)" if s.la == "N" else "(d2, d0)"
    mb = "(d2, d1)" if s.lb == "N" else "(d1, d2)"
    return f"""func.func @main(%a: {a_t}, %b: {b_t}) -> tensor<{s.M}x{s.N}xf16> {{
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<{s.M}x{s.N}xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e : tensor<{s.M}x{s.N}xf32>) -> tensor<{s.M}x{s.N}xf32>
  %acc = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2) -> {ma}>, affine_map<(d0, d1, d2) -> {mb}>,
                                          affine_map<(d0, d1, d2) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel", "reduction"]}}
      ins(%a, %b : {a_t}, {b_t}) outs(%init : tensor<{s.M}x{s.N}xf32>) {{
    ^bb0(%x: f16, %y: f16, %c: f32):
      %xf = arith.extf %x : f16 to f32
      %yf = arith.extf %y : f16 to f32
      %p = arith.mulf %xf, %yf : f32
      %s = arith.addf %c, %p : f32
      linalg.yield %s : f32
  }} -> tensor<{s.M}x{s.N}xf32>
  %o = tensor.empty() : tensor<{s.M}x{s.N}xf16>
  %r = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"]}} ins(%acc : tensor<{s.M}x{s.N}xf32>)
      outs(%o : tensor<{s.M}x{s.N}xf16>) {{
    ^bb0(%x: f32, %y: f16):
      %t = arith.truncf %x : f32 to f16
      linalg.yield %t : f16
  }} -> tensor<{s.M}x{s.N}xf16>
  return %r : tensor<{s.M}x{s.N}xf16>
}}
"""


def compile_all():
    tgt = (ROOT / "targets/turnip710.target").read_text().replace("\n", "")
    for s in SHAPES:
        d = OUT / s.id
        d.mkdir(parents=True, exist_ok=True)
        src = f"module attributes {{hal.device.targets = [{tgt}]}} {{\n{mlir(s)}}}\n"
        (d / "gemm.mlir").write_text(src)
        bdir = d / "bench"
        subprocess.run(["rm", "-rf", str(bdir)])
        r = subprocess.run([str(COMPILE), str(d / "gemm.mlir"), f"--iree-hal-dump-executable-benchmarks-to={bdir}",
                            f"--iree-hal-dump-executable-configurations-to={d / 'cfg'}", "-o", str(d / "gemm.vmfb")],
                           capture_output=True, text=True)
        if r.returncode:
            print(s.id, "COMPILE FAILED", r.stderr[-400:])
            continue
        benches = sorted(bdir.glob("*.mlir"))
        for b in benches:
            subprocess.run([str(COMPILE), str(b), "-o", str(b.with_suffix(".vmfb"))], check=True)
        cfgs = []
        for c in sorted((d / "cfg").glob("*.mlir")):
            t = c.read_text()
            m = re.search(r"func.func @(\S+?)\(\).*?translation_info = (#iree_codegen.translation_info<[^\n]*?>)[,}]", t, re.S)
            lc = re.search(r"lowering_config = (#iree_codegen.lowering_config<[^>]*>)", t)
            if m:
                cfgs.append(dict(dispatch=m.group(1), translation_info=m.group(2), lowering_config=lc.group(1) if lc else None))
        (d / "configs.json").write_text(json.dumps(cfgs, indent=1))
        print(f"{s.id}: {len(benches)} dispatch(es):", ", ".join(c["dispatch"].split("$async_")[-1] for c in cfgs))


def run_all(batch=20, reps=5):
    env = "LD_LIBRARY_PATH=/data/local/tmp/turnip "
    out = {}
    for s in SHAPES:
        d = OUT / s.id
        total, parts = 0.0, []
        for v in sorted((d / "bench").glob("*.vmfb")):
            remote = f"{device.DEV}/kern/iree_{v.name}"
            device.adb("push", str(v), remote)
            r = device.adb("shell", f"{env}{BENCH} --device=vulkan --module={remote} --batch_size={batch} "
                                    f"--benchmark_repetitions={reps} --benchmark_report_aggregates_only=true", check=False)
            m = re.search(r"_median\s+([\d.]+) (us|ms|ns|s)\b", r.stdout)
            us = float(m.group(1)) * {"us": 1, "ms": 1e3, "ns": 1e-3, "s": 1e6}[m.group(2)] if m else None
            if us is None:
                print(f"{s.id}: no timing ({r.returncode}): {(r.stdout + r.stderr)[-300:]}")
            parts.append((v.stem, us))
            total += us or 0
        out[s.id] = dict(us=total, gflops=s.flops / (total * 1e3) if total else None, dispatches=parts)
        print(f"{s.id}: {total:10.1f} us  {out[s.id]['gflops'] or 0:.1f} GFLOP/s  ({len(parts)} dispatches)")
    (OUT / "results.json").write_text(json.dumps(out, indent=1))


def iree_grid(s, lowering_config):
    """Workgroup count from IREE's first-level (workgroup) tile sizes over the loop dims
    (matmul loops (m, n, k); matvec loops (n, k)). IREE 3.12 linearizes the workgroup grid: the
    kernel reads only WorkgroupId.x and delinearizes it, so the count is the product along x."""
    t1 = [int(x) for x in re.search(r"tile_sizes = \[\[([^\]]*)\]", lowering_config).group(1).split(",")]
    dims = [s.N] if s.M == 1 else [s.M, s.N]
    n = 1
    for d, t in zip(dims, t1):
        if t:
            n *= -(-d // t)
    return (n, 1, 1)


def iree_candidate(s, sg=64):
    """Harness candidate running IREE's own SPIR-V for shape s (compiled with dumped binaries)."""
    d = OUT / s.id
    bdir = d / "bin"
    if not bdir.exists():
        subprocess.run([str(COMPILE), str(d / "gemm.mlir"), f"--iree-hal-dump-executable-binaries-to={bdir}",
                        "-o", "/dev/null"], check=True)
    cfgs = json.loads((d / "configs.json").read_text())
    disp = []
    for i, c in enumerate(cfgs):
        src = next(bdir.glob(f"*{c['dispatch']}.spv"))
        spv = d / f"iree_{s.id}_{i}.spv"  # short unique name on the phone
        spv.write_bytes(src.read_bytes())
        disp.append((f"{spv}@{c['dispatch']}", iree_grid(s, c["lowering_config"])))
    return {"name": f"{s.id}_iree", "shape": s.id, "sg": sg, "dispatches": disp}, cfgs


def kernel_all(driver="turnip", sg=64):
    out = {}
    for s in SHAPES:
        cand, cfgs = iree_candidate(s, sg)
        r = device.run_batch([cand], OUT, f"{s.id}_iree", driver=driver).get(cand["name"], {})
        us = r.get("us") if r.get("ok") else None
        out[s.id] = dict(us=us, gflops=s.flops / (us * 1e3) if us else None, ok=r.get("ok"), err=r.get("err"),
                         error=r.get("error"), grid=[g for _, g in cand["dispatches"]], configs=cfgs)
        print(f"{s.id}: ok={r.get('ok')} err={r.get('err')}  {us} us  "
              f"{out[s.id]['gflops'] and round(out[s.id]['gflops'], 1)} GFLOP/s  grid {cand['dispatches'][0][1]}  "
              f"{r.get('error') or ''}")
    (OUT / "kernel_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"compile": compile_all, "run": run_all, "kernel": kernel_all}[sys.argv[1]]()
