# Adreno port — working notes

Target: moto g stylus 5G (2024), Snapdragon 7s Gen 2, **Adreno 710**, Vulkan 1.1.128 driver
(see [DEVICE.md](DEVICE.md)). Toolchain: IREE 3.11.0 (compiler pip + runtime
cross-built), torch 2.14.1, transformers 5.19.0. Host setup: [SETUP.md](SETUP.md).

## GPU driver + compiler upgrade (2026-10-12)

The stock Qualcomm driver (Vulkan 1.1.128, shader compiler EV031.36.08.28, 2024-01) is the
root of most problems. The phone can't get a newer vendor driver: bootloader locked, no
updatable-driver support (`ro.gfx.driver.*` empty), last OTA = Android 14 / 2024-12 patch.
**Fix without root: Mesa Turnip** (open-source Vulkan driver for Adreno, built from Mesa `main` —
Adreno 710 support is not in any stable Mesa release yet) loaded only by our processes through
a `libvulkan.so` shim (`runner/turnip/vulkan_shim.c`) via `LD_LIBRARY_PATH=/data/local/tmp/turnip`.

| | Qualcomm stock | Turnip (Mesa 26.3.0-devel 745709a) |
|---|---|---|
| Vulkan | 1.1.128 | 1.4.363 |
| sync2 / timeline / scalarBlockLayout | no | yes → **IREE 3.12 works** |
| int dot product / shaderInt64 | no / no | yes / yes |
| subgroup reduce, 3 subgroups (probe) | wrong (1024) | right (768) once subgroup size pinned |
| DistilBERT fp32 (generic adreno target) | wrong (Δ 1.32) | **2.4e-6** |
| tiny Llama (prefill + decode) | wrong | **exact** |
| DistilBERT fp16 | NaN | VK_ERROR_DEVICE_LOST (open) |

Turnip-specific: it runs compute 128-wide unless a subgroup size is required; IREE assumes 64 and
only pins it for some pipelines → the shim adds `requiredSubgroupSize=64` to every compute
pipeline that doesn't set one (`TURNIP_SUBGROUP_SIZE`, 0 = off).

**IREE runtime bug (3.11 and 3.12):** `iree_hal_module_command_buffer_execution_barrier` always
uses a global barrier scoped DISPATCH_WRITE→DISPATCH_READ, so dispatch→copy and copy→dispatch
hazards get no cache flush. LLM decode (M=1) lowers the int4 halves' concat/slices to
`stream.cmd.copy` → stale reads on Adreno with *either* driver. Patched in both trees
(`runner/patches/`). Worth reporting upstream.

How to run with the upgraded stack:
```
IREE_VERSION=3.12 scripts/compile.sh <dir> turnip710
IREE_VERSION=3.12 ADRENO_DRIVER=turnip scripts/check.py <dir> turnip710 --device
adb shell 'LD_LIBRARY_PATH=/data/local/tmp/turnip /data/local/tmp/iree312/llm-runner ...'
```

## Layout

```
scripts/build_android_runtime.sh   cross-compile IREE runtime tools for the phone
scripts/export_hf.py               HF seq-classification model -> torch-dialect MLIR (+ inputs, fp32 ref)
scripts/compile.sh <dir> <target>  iree-compile for host-cpu | android-cpu | adreno710 | adreno-generic
scripts/check.py <dir> <target>    run (locally or on phone over adb) and compare with PyTorch
scripts/debug/op_test.py           per-op GPU numerics regression tests
scripts/debug/bisect_layers.py     layer-by-layer GPU bisect (DistilBERT)
targets/adreno710.target           our device target description (hal.device.targets attr)
```

## Status

| Model | Target | Result |
|---|---|---|
| DistilBERT SST-2, fp32, seq 128, batch 2 | host-cpu | max abs logit error 3.6e-6 ✅ |
| " | android-cpu (phone CPU) | 3.6e-6 ✅ |
| " | **adreno710 (phone GPU)** | **3.7e-6 ✅** |
| " | adreno-generic (IREE built-in target) | wrong logits (error up to 1.3) ❌ — subgroup bug below |
| DistilBERT fp16 (weights exported in fp16) | host-cpu | finite, error 0.31 (fp16 precision) |
| " | adreno710 | NaN ❌ — fp16 reductions overflow on GPU (LayerNorm variance) |

## Findings (each one cost a debugging loop — read before changing anything)

1. **IREE ≥ 3.12 cannot run on this phone.** 3.12 rewrote the Vulkan HAL and hard-requires
   Vulkan 1.3 + synchronization2 (`device_plan.c`, `physical_device_selection.c`). The Adreno 710
   driver is Vulkan 1.1.128. Pinned compiler + runtime to **3.11.0** (they must match).
2. **No shaderInt64** on the device. `--iree-input-demote-i64-to-i32` works for computed values but
   **cannot retype `dense_resource` constants** (fails with `arith.constant ... same type`). Fix: cast
   int64 buffers (e.g. `position_ids`) and inputs to int32 in PyTorch before export.
   Same limitation for `--iree-input-demote-f32-to-f16` → cast weights to fp16 in PyTorch instead.
3. **transformers 5 vmap mask builder** (`masking_utils`) produces a 0-d bool `new_ones` that IREE 3.11's
   torch lowering rejects. Fix: pass a precomputed 4D additive mask (HF uses 4D masks as-is).
4. torch-mlir's sdpa → `tm_tensor.attention` lowering **does not broadcast the mask** over the query
   dim → expand mask to `[B,1,S,S]`.
5. **SPIR-V backend has no codegen for `iree_linalg_ext.attention`** (fused attention from sdpa) — falls
   back to `linalg-ext-to-loops`, which fails. Decomposing it in preprocessing
   (`iree-linalg-ext-decompose-aggregated-ops{filter-ops=iree_linalg_ext.attention}`) compiles but
   gives **NaN even on CPU** (masked attention) → upstream bug to investigate. Workaround: export with
   eager attention (plain matmul/softmax/matmul). A real Adreno attention kernel is a TODO.
6. **Adreno shader-compiler bug: cross-subgroup reduction with a non-power-of-two subgroup count.**
   `SPIRVSubgroupReduce` with 3/5/6 subgroups per workgroup gives `partial × nextPow2(n)`; 1/2/4/8
   subgroups are exact. The SPIR-V IREE emits is correct (lanes ≥ n get the identity via `OpSelect`
   before the `OpGroupNonUniformShuffleXor` butterfly) — the driver (compiler EV031.36.08.28,
   2024-01-05) ignores the select. Pinning subgroup size to 64 does not help.
   IREE's generic `adreno` target picks 192 threads (3 subgroups) for K=768 reductions → wrong
   LayerNorm / classifier results. **Fix: our `targets/adreno710.target` caps
   `max_thread_count_per_workgroup = 128`**, so `setReductionConfig`'s GCD clamp always yields 1 or 2
   subgroups. Probe: `tests/probes/subgroup_probe.mlir` with all-ones inputs.
7. IREE's generic `adreno` target also claims `dot = dp4xi8toi32` and 64-bit storage; the 710 driver
   has neither → removed in our target. It also claims SPIR-V 1.6 (Vulkan 1.1 only guarantees 1.3);
   the driver accepts 1.6 modules in practice, so left as-is for now.
8. **fp16 reductions overflow on GPU** (LayerNorm sum of squares over 768). CPU backend stays finite.
   For LLMs: keep norms / softmax / attention accumulation in fp32, weights + matmuls in fp16.
9. `maxStorageBufferRange` = 128 MB: any single tensor bound to a kernel must be < 128 MB
   (matters for LLM embedding / lm_head tables — split them).

## Status at pause (2026-10-12 08:10) — resume here

Working stack: **Mesa Turnip + IREE 3.12 (+ barrier patch) + target `turnip710`**.

Done / verified on the phone GPU:
* DistilBERT fp32 exact; tiny Llama (prefill + decode, chunked, bucketed) exact via `runner/llm_runner.c`.
* Llama 3.2 3B int4 eager reference on Mac produces correct text (`models/llama32-3b-q4/ref_q4.json`).
* Decode int4 GEMV: dequant fusion fixed at export; **K-packed int32 layout** (`llama_q4.QLinear`)
  gives ONE fused kernel; kernel tuner: qkv-shaped GEMV 19-20 ms (orig default) -> 12-15 ms (new
  layout default) -> 3.54 ms (`sgr_h1_rows1_k32_wg64`) -> **2.63 ms** (`bv_h1_nt2_wx128_g2_k4`,
  BaseVectorize; search stopped at 49 of ~300 candidates, models/qgemv_k8/tune/tune.log).

Not done yet:
1. Push the NEW `models/llama32-3b-q4/weights.irpa` (K-packed; device copy is the old layout!).
2. Compile 3B (`IREE_VERSION=3.12 scripts/compile.sh models/llama32-3b-q4-tune turnip710
   --iree-opt-const-expr-hoisting=false`), check compile time (attention @L=4096 was slow), run
   runner, compare tokens with ref_q4.json, measure baseline tok/s.
3. Apply tuned GEMV configs to all decode GEMV shapes (kernel_tuner profile/tune + substitution),
   then flag tuner (scripts/tune/flag_tuner.py), then bench across ctx 256..4K (scripts/llm/bench.py,
   schedules bucketed vs max) and summarize (scripts/llm/summarize.py).
4. Open: fp16 DistilBERT VK_ERROR_DEVICE_LOST on Turnip; IREE 3.12 HAL ~2.3x per-dispatch overhead
   vs 3.11 on Turnip (compare end-to-end).
