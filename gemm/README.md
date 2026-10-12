# gemm/: fp16 GEMM schedule search for the Adreno 710

Companion code for [Part 2: Tuning fp16 GEMMs on a Phone GPU](../BLOG-GEMM.md). A TVM/Ansor-style
schedule space lowered to GLSL, compiled to SPIR-V with `glslc`, and checked and timed on the phone
(Mesa Turnip, Vulkan 1.4) by a small Vulkan harness over adb on Wi-Fi.

```
propose ──> verifier ──> cost model ──> glslc ──> phone harness ──> learn
(search.py)  (verifier.py)  (cost_model.py)  (build.py)  (device.py + harness/)
```

| file | what it does |
|---|---|
| `shapes.py` | the 8 GEMMs (NN/NT/TN/TT, flat, skinny, large/small K, GEMV), data + fp32 reference |
| `schedule.py` | `Schedule` (tiling, vthreads, binding, rfactor + cross-thread reduction, shared staging, double buffering, vectorize, unroll, inline/materialize transposes, f32 / f16 / promoted-f16 accumulate) and its GLSL code generator |
| `verifier.py` | static legality rules (device limits, Mesa ir3 residency, numerics measured on the GPU) + index-algebra replay (coverage, guards, whole-vector shared writes) |
| `cost_model.py` | analytic roofline with ir3 occupancy, lane-order coalescing and bank conflicts; online gradient-boosted model on 48 features |
| `search.py` | evolutionary search (model) vs random search, same verifier and GPU budget |
| `harness/vk_gemm_bench.c` | Vulkan harness: poisons C, checks against the reference, times with GPU timestamps |
| `device.py` | adb (Wi-Fi only, `ADB_WIFI=ip:port`), batch push/run, recovery from GPU hangs and transport drops |
| `fuzz.py` | correctness fuzz: random verified schedules on every shape must match the reference (also calibrates the analytic model) |
| `roofline.py` | measured fp16/fp32 FMA peaks and DRAM bandwidth |
| `iree_baseline.py` | IREE 3.12's default kernels for the same GEMMs, in our harness and in IREE's runtime |
| `analyze.py` | summary, ablations, interleaved final re-measurement, cost-model and verifier evaluation |
| `plots.py` | charts in `docs/img/gemm-*.png` |

## Run it

```bash
TC=$ANDROID_NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin
mkdir -p build/gemm && $TC/aarch64-linux-android34-clang -O2 -Wall -Wno-unused-function \
    -Ithird_party/iree/third_party/vulkan_headers/include -o build/gemm/vk_gemm_bench \
    gemm/harness/vk_gemm_bench.c -ldl -lm
cd gemm
export ADB_WIFI=192.168.1.7:5555                      # your phone's adb-over-Wi-Fi address
../.venv-iree312/bin/python roofline.py               # measured peaks -> runs/roofline.json
../.venv-iree312/bin/python fuzz.py --tag fuzz3       # codegen correctness on the phone (must be ~100%)
../.venv-iree312/bin/python search.py --tag run2      # 8 shapes x {model, random} x 96 measurements
../.venv-iree312/bin/python search.py --tag run2x --seed-from run2 --acc f32 f16 f16p --method model --budget 32
../.venv-iree312/bin/python iree_baseline.py compile  # IREE default kernels (Mac)
../.venv-iree312/bin/python iree_baseline.py kernel   # ... timed in our harness
../.venv-iree312/bin/python analyze.py summary --tag run2 --ext run2x
for c in ablate final rank verifier; do GEMM_WARMUP_MS=3000 ../.venv-iree312/bin/python analyze.py $c --tag run2; done
../.venv-iree312/bin/python plots.py --tag run2
```

Requires Mesa Turnip in `/data/local/tmp/turnip` on the phone (`scripts/build_turnip.sh --push`).

Results of the run behind the write-up are committed in `runs/` (logs + JSON summaries; the test
data and compiled kernels are regenerated): tuned kernels 7.0x naive and 28x IREE 3.12's default
(geomean), model-guided search 1.9x random search at equal GPU budget, GEMV at 97.5% of roofline.
