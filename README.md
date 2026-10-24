# iree-adreno-genai-tuner

Lowering Hugging Face PyTorch models to a mid-range **Adreno 710** phone GPU through
**MLIR / IREE → SPIR-V → Vulkan**, plus the start of a **genAI tuner** for the IREE stack.
Companion code for the blog posts **[Bringing PyTorch LLMs to a Mid-Range Adreno GPU with IREE](BLOG.md)**,
**[Part 2: Tuning fp16 GEMMs on a Phone GPU: Schedules, a Verifier and a Cost Model](BLOG-GEMM.md)** and
**[Part 3: Running Mojo GPU Kernels on an Adreno Phone GPU](BLOG-MOJO.md)**.

Target: moto g stylus 5G (2024) · Snapdragon 7s Gen 2 · Adreno 710 · Android 14, driven from a
Mac over adb (Wi-Fi). Everything runs without root.

## What's here

| Result | Status |
|---|---|
| DistilBERT SST-2 fp32 on the phone GPU | matches PyTorch (max logit error 2.4e-6 to 3.7e-6) |
| Mesa **Turnip** driver (Vulkan 1.4) loaded per-process via a `libvulkan.so` shim | works; unblocks IREE 3.12 |
| IREE runtime barrier bug (stale reads after buffer copies; broke LLM decode) | patched for 3.11 and 3.12 (`runner/patches/`) |
| Qualcomm shader-compiler bug (3/5/6-subgroup reductions) | worked around with a target description (`targets/adreno710.target`) |
| Tiny Llama (prefill + decode, chunked, KV buckets) on the GPU | exact vs Hugging Face |
| Llama 3.2 3B int4 export (int4 K-packed int32, 5 KV buckets, in-place KV cache) | built + validated on host; on-device run is next |
| Kernel tuner on the decode int4 GEMV (5120 x 3072) | 20.6 ms → **2.63 ms** (7.8x) |
| fp16 GEMM schedule search ([`gemm/`](gemm/)): 8 shapes, verifier + cost model + evolutionary search | **7.0x** naive, **28x** IREE default (geomean); 1.9x random search at equal budget; GEMV at 97.5% of roofline |
| Mojo GPU kernels on the Adreno 710 ([`mojo/`](mojo/), standalone: [mojo-adreno](https://github.com/adityaGoCT/mojo-adreno)): Mojo's kernel LLVM IR (AMD `gfx90a` build, 64-lane warps) retargeted to Vulkan SPIR-V and OpenCL C | PASS, 0 mismatches on Turnip, stock Vulkan 1.1, Qualcomm OpenCL 3.0 and the phone CPU, warp shuffles included |

Every bug and its fix: [`docs/BUGS.md`](docs/BUGS.md); bisection notes: [`docs/NOTES.md`](docs/NOTES.md). Tuner design:
[`docs/TUNER_PLAN.md`](docs/TUNER_PLAN.md). Device facts: [`docs/DEVICE.md`](docs/DEVICE.md).

## Layout

```
scripts/
  fetch_deps.sh            pinned IREE v3.11.0 / v3.12.0 (+ barrier patch) and Mesa main@745709a
  setup_venvs.sh           .venv (IREE 3.11), .venv-iree312 (IREE 3.12), .venv-mojo (Mojo 1.2 + MAX)
  build_android_runtime.sh IREE runtime tools for Android arm64 (IREE_VERSION=3.11|3.12)
  build_runner.sh          runner/llm_runner.c for Android (IREE_VERSION=3.11|3.12)
  build_turnip.sh          Mesa Turnip + libvulkan.so shim + vkinfo (--push to install on phone)
  export_hf.py             HF sequence-classification model -> torch-dialect MLIR + inputs + ref
  compile.sh               iree-compile for host-cpu | android-cpu | adreno710 | turnip710 | adreno-generic
  check.py                 run on Mac or phone (adb) and compare with PyTorch
  llm/                     LlamaQ4 (int4 Llama for IREE), export, tiny HF check, host run, bench, summary
  tune/                    kernel_tuner.py (L2 kernel configs), flag_tuner.py (L1 compile flags)
  debug/                   per-op GPU tests, layer bisect, dispatch-trace diff, probes
runner/
  llm_runner.c             on-device LLM runner = scheduler (chunked prefill, smallest KV bucket)
  turnip/vulkan_shim.c     loads Turnip's HAL module, pins subgroup size to 64
  turnip/vkinfo.c          compute capability dump (stock vs Turnip)
  patches/                 IREE HAL execution-barrier fix (3.11, 3.12)
gemm/                      fp16 GEMM tuner: schedule space -> GLSL, verifier, cost model, search, Vulkan harness
mojo/                      Mojo -> Adreno: kernel IR -> Vulkan SPIR-V (mojo2vk) / OpenCL C (mojo2cl), phone runners
targets/                   GPU target descriptions injected as hal.device.targets
tests/probes/              subgroup-reduction probe, smoke matmul
```

## Quickstart

Prereqs (macOS host; Linux works with path changes): Homebrew `cmake ninja meson bison`,
`android-ndk` (r30 used), `uv`, `adb` with the phone connected (`adb devices`). See
[`docs/SETUP.md`](docs/SETUP.md).

```bash
scripts/fetch_deps.sh            # third_party/ (IREE 3.11, IREE 3.12, Mesa)
scripts/setup_venvs.sh           # .venv, .venv-iree312, .venv-mojo
IREE_VERSION=3.12 scripts/build_android_runtime.sh
IREE_VERSION=3.12 scripts/build_runner.sh
scripts/build_turnip.sh --push   # Turnip into /data/local/tmp/turnip
adb shell mkdir -p /data/local/tmp/iree312
adb push build/iree312-android/tools/iree-run-module build/iree312-android/tools/iree-benchmark-module \
         build/runner312-android/llm-runner /data/local/tmp/iree312/
```

DistilBERT end to end on the GPU (Turnip + IREE 3.12):

```bash
.venv-iree312/bin/python scripts/export_hf.py --out models/distilbert-sst2
IREE_VERSION=3.12 scripts/compile.sh models/distilbert-sst2 turnip710
IREE_VERSION=3.12 ADRENO_DRIVER=turnip .venv/bin/python scripts/check.py models/distilbert-sst2 turnip710 --device
```

Llama 3.2 3B int4 (weights come from your own Hugging Face download; nothing is redistributed):

```bash
.venv-iree312/bin/python scripts/llm/test_tiny.py            # LlamaQ4 vs HF on a tiny config
hf download meta-llama/Llama-3.2-3B-Instruct                 # accept the Llama 3.2 license on HF first
ls -d ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/* > models/LLAMA_SNAPSHOT.txt
.venv-iree312/bin/python scripts/llm/export_llama.py --model-dir "$(cat models/LLAMA_SNAPSHOT.txt)" \
    --out models/llama32-3b-q4 --buckets 256 512 1024 2048 4096
IREE_VERSION=3.12 scripts/compile.sh models/llama32-3b-q4 turnip710 --iree-opt-const-expr-hoisting=false
adb shell mkdir -p /data/local/tmp/iree/llama32-3b-q4
adb push models/llama32-3b-q4/weights.irpa models/llama32-3b-q4/turnip710.iree312.vmfb /data/local/tmp/iree/llama32-3b-q4/
.venv-iree312/bin/python scripts/llm/bench.py --vmfb turnip710.iree312.vmfb --schedule bucketed max
```

Tuning a kernel (standalone dispatch benchmarks, timed on the phone):

```bash
IREE_VERSION=3.12 scripts/compile.sh <model_dir> turnip710   # dumps <target>.iree312.benchmarks/
.venv-iree312/bin/python scripts/tune/kernel_tuner.py tune-files --model-dir <model_dir> \
    --files <model_dir>/turnip710.iree312.benchmarks/<dispatch>.mlir --max-candidates 200
```

fp16 GEMM tuning on the phone (Turnip; adb over Wi-Fi, `ADB_WIFI=ip:port`): see [`gemm/README.md`](gemm/README.md).

A Mojo GPU kernel on the phone's GPU (Turnip, stock Vulkan, OpenCL) and CPU, checked element by element;
see [`mojo/README.md`](mojo/README.md):

```bash
python3 mojo/mojo_adreno.py mojo/src/vecadd_gpu.mojo   # summary: turnip=PASS  stock=PASS  opencl=PASS  cpu=PASS
```

## Status

Work in progress. Next: on-device Llama 3B validation and baseline tok/s, tuned configs applied
to every decode GEMV shape, flag tuning, and the 256 to 4K context benchmark (bucketed vs fixed 4K).

## Licenses and credits

Code in this repo: see [`LICENSE`](LICENSE). Builds on [IREE](https://github.com/iree-org/iree)
(Apache-2.0 with LLVM exceptions) and [Mesa](https://gitlab.freedesktop.org/mesa/mesa) (MIT);
patches in `runner/patches/` modify IREE and keep its license. Llama 3.2 weights are governed by
the Llama 3.2 Community License and are not included.
