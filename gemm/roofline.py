"""Measured roofline for the phone's GPU: peak FMA throughput (fp32, fp16) and memory bandwidth.

  roofline.py            -> runs/roofline.json  {"f32_gflops", "f16_gflops", "read_gbs", "copy_gbs"}
Microbenchmarks run through the same harness as the GEMM candidates (no correctness check).
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import device  # noqa: E402
from build import compile_glsl  # noqa: E402

RUNS = pathlib.Path(__file__).parent / "runs"
# microbenchmark "shape": A = S0 = 4096 x 4096 fp16 = 32 MiB each
MB_M, MB_N, MB_K = 4096, 16, 4096
WG, GROUPS = 256, 512
FMA_VARIANTS = [(4, 4), (8, 4), (8, 1), (16, 1)]  # (independent chains, unroll): ILP vs occupancy

HDR = """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
"""


def fma_kernel(dtype, chains, unroll):
    """FMA throughput probe. Needs tuning itself: on Adreno 710, 4 chains x unroll 4 reaches
    921 GFLOP/s fp16 while 8+ chains lose occupancy (register pressure) and run slower."""
    t = "vec4" if dtype == "f32" else "f16vec4"
    s = "float" if dtype == "f32" else "float16_t"
    iters = 8192 // chains // unroll
    init = "\n".join(f"  {t} v{i} = {t}({s}(gl_LocalInvocationID.x) * {s}(0.001) + {s}({i}));" for i in range(chains))
    body = "\n".join([f"    v{i} = fma(v{i}, a, b);" for i in range(chains)] * unroll)
    total = " + ".join(f"v{i}" for i in range(chains))
    src = HDR + f"""layout(local_size_x = {WG}) in;
layout(std430, binding = 2) writeonly buffer C1 {{ float16_t C[]; }};
void main() {{
  {t} a = {t}({s}(0.999)), b = {t}({s}(0.0001));
{init}
  for (uint i = 0u; i < {iters}u; ++i) {{
{body}
  }}
  {t} r = {total};
  if (r.x == {s}(-12345.0)) C[gl_GlobalInvocationID.x] = float16_t(r.y);  // defeat DCE
}}
"""
    return src, WG * GROUPS * iters * unroll * chains * 4 * 2


def bw_kernel(copy):
    n16 = MB_M * MB_K * 2 // 16  # uvec4 elements in 32 MiB
    total_threads = WG * GROUPS
    per = n16 // total_threads
    store = "S[i] = v;" if copy else "acc ^= v;"
    tail = "" if copy else "if (acc.x == 0x12345678u) S[gl_GlobalInvocationID.x] = acc;"
    decl_s = "layout(std430, binding = 3) writeonly buffer O { uvec4 S[]; };"
    return HDR + f"""layout(local_size_x = {WG}) in;
layout(std430, binding = 0) readonly buffer I {{ uvec4 A[]; }};
{decl_s}
void main() {{
  uvec4 acc = uvec4(0);
  for (uint j = 0u; j < {per}u; ++j) {{
    uint i = gl_GlobalInvocationID.x + j * {total_threads}u;   // grid-stride, coalesced
    uvec4 v = A[i];
    {store}
  }}
  {tail}
}}
""", per * total_threads * 16


def main():
    d = RUNS / "data" / "RF"
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.txt").write_text(f"{MB_M} {MB_N} {MB_K} N N -1\n")
    device.adb("shell", f"mkdir -p {device.DEV}/shapes/RF")
    device.adb("push", str(d / "meta.txt"), f"{device.DEV}/shapes/RF/meta.txt")
    cands, work = [], {}
    for dtype in ("f32", "f16"):
        for chains, unroll in FMA_VARIANTS:
            src, flops = fma_kernel(dtype, chains, unroll)
            spv, err = compile_glsl(src)
            assert not err, err
            name = f"{dtype}_fma_c{chains}u{unroll}"
            cands.append({"name": name, "shape": "RF", "sg": 128, "dispatches": [(spv, (GROUPS, 1, 1))]})
            work[name] = flops
    for name, copy in [("read_bw", False), ("copy_bw", True)]:
        src, nbytes = bw_kernel(copy)
        spv, err = compile_glsl(src)
        assert not err, err
        cands.append({"name": name, "shape": "RF", "sg": 0, "dispatches": [(spv, (GROUPS, 1, 1))]})
        work[name] = nbytes * (2 if copy else 1)
    res = device.run_batch(cands, RUNS, "roofline", target_ms=60, reps=7)
    out = {}
    for name, units in work.items():
        val = units / (res[name]["us"] * 1e3)
        print(f"{name:16s} {res[name]['us']:10.1f} us  ->  {val:8.1f} {'GFLOP/s' if 'fma' in name else 'GB/s'}")
        key = ("f32_gflops" if name.startswith("f32") else "f16_gflops") if "fma" in name else \
              ("read_gbs" if name == "read_bw" else "copy_gbs")
        out[key] = max(out.get(key, 0), round(val, 1))
    print(out)
    (RUNS / "roofline.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
