# Bugs and fixes

Every problem hit while lowering Hugging Face models to the Adreno 710 through IREE, in the order
they appeared. Each entry: symptom → cause → fix → where the fix lives. Longer bisection notes:
[NOTES.md](NOTES.md).

## B1 IREE 3.12 refuses Vulkan 1.1 devices
- **Symptom:** `iree-run-module --dump_devices` marks the GPU `[below baseline]`; device creation fails.
- **Cause:** IREE 3.12 rewrote its Vulkan HAL and requires Vulkan 1.3 + synchronization2. The stock Adreno 710 driver is Vulkan 1.1.128.
- **Fix:** pin compiler + runtime to 3.11 for the stock driver; use Mesa Turnip (Vulkan 1.4) to run 3.12.
- **Where:** [`scripts/setup_venvs.sh`](../scripts/setup_venvs.sh), [`scripts/build_turnip.sh`](../scripts/build_turnip.sh).

## B2 No 64-bit integers, and input demotion cannot retype weights
- **Symptom:** `arith.constant op failed to verify that all of {value, result} have same type`.
- **Cause:** the GPU has no `shaderInt64`; `--iree-input-demote-i64-to-i32` (and `-f32-to-f16`) cannot retype `dense_resource` weight constants.
- **Fix:** cast int64 buffers (e.g. `position_ids`) and inputs to int32, and weights to the target dtype, in PyTorch before export.
- **Where:** [`scripts/export_hf.py`](../scripts/export_hf.py), [`scripts/llm/llama_q4.py`](../scripts/llm/llama_q4.py).

## B3 transformers 5 mask builder does not lower
- **Symptom:** `failed to legalize operation 'torch.vtensor.literal'` on a 0-d bool tensor.
- **Cause:** transformers 5 builds attention masks with vmap and a 0-d bool `new_ones`.
- **Fix:** pass a precomputed 4D additive mask; Hugging Face uses 4D masks as-is.
- **Where:** `LogitsOnly.forward` in [`scripts/export_hf.py`](../scripts/export_hf.py).

## B4 SDPA mask is not broadcast
- **Symptom:** `'tm_tensor.attention' op mask sequence length and query sequence length mismatch`.
- **Cause:** torch-mlir's SDPA lowering does not broadcast a `[B,1,1,S]` mask over the query dim.
- **Fix:** expand the mask to `[B,1,S,S]`.
- **Where:** [`scripts/export_hf.py`](../scripts/export_hf.py).

## B5 No SPIR-V codegen for fused attention
- **Symptom:** SPIR-V translation fails in `LinalgExtToLoopsPass` on `iree_linalg_ext.attention`; decomposing it with `iree-linalg-ext-decompose-aggregated-ops` compiles but returns NaN, even on CPU.
- **Fix:** export with eager attention (matmul, softmax, matmul). A real Adreno attention kernel is future work.
- **Where:** `--attn eager` in [`scripts/export_hf.py`](../scripts/export_hf.py); `LlamaQ4.step` in [`scripts/llm/llama_q4.py`](../scripts/llm/llama_q4.py).

## B6 Qualcomm shader compiler breaks non-power-of-two subgroup reductions
- **Symptom:** right labels, wrong logits (error up to 1.32) on the GPU only.
- **Cause:** `SPIRVSubgroupReduce` with 3, 5 or 6 subgroups returns `partial x nextPow2(n)`; the emitted SPIR-V is correct (an `OpSelect` before an `OpGroupNonUniformShuffleXor` butterfly), the driver (compiler EV031.36.08.28) ignores the select.
- **Fix:** a device target with `max_thread_count_per_workgroup = 128`, so reductions use 1 or 2 subgroups.
- **Where:** [`targets/adreno710.target`](../targets/adreno710.target); probe [`tests/probes/subgroup_probe.mlir`](../tests/probes/subgroup_probe.mlir).

## B7 fp16 reductions overflow on the GPU
- **Symptom:** fp16 DistilBERT returns NaN on the GPU, finite on CPU.
- **Cause:** LayerNorm variance (a sum of squares over 768 values) overflows fp16 in GPU arithmetic.
- **Fix:** keep norms, softmax and attention math in fp32; store the KV cache in fp16.
- **Where:** [`scripts/llm/llama_q4.py`](../scripts/llm/llama_q4.py).

## B8 Turnip runs compute 128-wide unless asked
- **Symptom:** the subgroup probe returns 1280 instead of 768 on Turnip.
- **Cause:** Turnip picks 64- or 128-wide waves per shader; IREE assumes 64 and pins it only for some pipelines.
- **Fix:** the `libvulkan.so` shim adds `requiredSubgroupSize = 64` to every compute pipeline that doesn't set one.
- **Where:** [`runner/turnip/vulkan_shim.c`](../runner/turnip/vulkan_shim.c).

## B9 IREE execution barriers miss transfer accesses
- **Symptom:** LLM decode wrong on both drivers; prefill right. A dispatch-trace diff shows correct matmul outputs followed by stale inputs after `stream.cmd.copy`.
- **Cause:** `iree_hal_module_command_buffer_execution_barrier` always uses a global barrier scoped `DISPATCH_WRITE` → `DISPATCH_READ`, so dispatch→copy and copy→dispatch get no cache flush.
- **Fix:** widen the scopes to include transfer reads and writes (IREE 3.11 and 3.12).
- **Where:** [`runner/patches/`](../runner/patches/).

## B10 `index_copy_` on a non-leading dim does not lower
- **Symptom:** `operand types should have the same type as the list contained type`.
- **Cause:** the `None` index torch emits for dims before the update dim.
- **Fix:** KV caches laid out position-first, `[max_ctx, kv_heads, head_dim]`, updated along dim 0.
- **Where:** `LlamaQ4.step` in [`scripts/llm/llama_q4.py`](../scripts/llm/llama_q4.py).

## B11 Bool `&` and tensor-by-tensor shifts do not lower
- **Symptom:** `failed to legalize operation 'torch.aten.__and__.Tensor'` / `torch.aten.__rshift__.Tensor`.
- **Fix:** range checks as `clamp(idx) == idx`; nibble extraction with scalar shifts.
- **Where:** `QEmbedding.lookup`, `QLinear._gemv_fused` in [`scripts/llm/llama_q4.py`](../scripts/llm/llama_q4.py).

## B12 int4 dequant not fused into the decode matmul
- **Symptom:** decode GEMVs read fully dequantized fp32 weights (`tensor<2560x3072xf32>`): about 8x the memory traffic of int4.
- **Cause:** a reshape between dequant (`[N, K/G, G]`) and `F.linear` (`[N, K]`) blocks fusion; later, a two-input multiply-and-sum is treated as a contraction, which also blocks producer fusion.
- **Fix:** decode GEMV written as one elementwise chain over K-packed int32 words with the scale folded into each term, then a single-input sum: one fused kernel.
- **Where:** `QLinear._gemv_fused` in [`scripts/llm/llama_q4.py`](../scripts/llm/llama_q4.py).

## B13 Tensor literals break dispatch benchmarks
- **Symptom:** the tuner cannot compile a dumped dispatch benchmark: `could not find resource blob`.
- **Cause:** `torch.tensor([...])` literals become `dense_resource` blobs captured inside the kernel; the benchmark dump drops them.
- **Fix:** build such constants from `torch.arange` (an iota).
- **Where:** [`scripts/llm/llama_q4.py`](../scripts/llm/llama_q4.py).

## B14 Const-expr hoisting doubles int4 weight memory
- **Symptom:** initializer dispatches unpack packed weights at load time.
- **Fix:** `--iree-opt-const-expr-hoisting=false` for LLM builds.
- **Where:** [`scripts/tune/flag_tuner.py`](../scripts/tune/flag_tuner.py) (`BASE`), README commands.

## B15 Mesa cross-build drops its own warning flags
- **Symptom:** NDK r30 clang fails on C++11 narrowing in Turnip headers although Mesa passes `-Wno-c++11-narrowing`.
- **Cause:** `-static-libstdc++` as a *compile* argument makes every meson compiler probe fail, so Mesa's flags are silently dropped.
- **Fix:** pass it as `cpp_link_args`.
- **Where:** [`scripts/build_turnip.sh`](../scripts/build_turnip.sh).

## Open
- **O1** fp16 DistilBERT hits `VK_ERROR_DEVICE_LOST` on Turnip (all fp16 ops pass individually: [`scripts/debug/f16_test.py`](../scripts/debug/f16_test.py)).
- **O2** IREE 3.12's new HAL costs about 2.3x more per dispatch than 3.11's on Turnip (1.23 vs 0.54 ms for a tiny kernel, 100 per submission).
- **O3** Some default configs compile very slowly (> 2 min per kernel: attention at L = 4096, GEMVs with dynamically offset bindings).
