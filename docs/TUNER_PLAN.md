# A genAI tuner for the IREE stack — plan

Target: LLM / genAI inference compiled with IREE (torch → MLIR → SPIR-V/Vulkan here; the design is
backend-agnostic). Goal: maximise **prefill tok/s** (compute-bound, TTFT) and **decode tok/s**
(memory/latency-bound) per *device × driver × compiler version*, without ever trading away
correctness. Reference implementation + measurements: Llama 3.2 3B int4 on Adreno 710
(see "Implemented" at the end).

## Why genAI needs its own tuner

* **Two workloads in one model.** Prefill = GEMM (M = chunk), decode = GEMV (M = 1) + attention
  over a growing KV cache. The best kernel config, fusion and even precision differ per phase.
* **Shapes are dynamic, compilers like static.** Context length moves 1 → 4K+ per request. We
  compile a small set of static *buckets* and schedule between them; the bucket set itself is a
  tuning knob.
* **Per-token host overhead matters.** A 3B decode step is ~300+ dispatches; on a phone CPU the
  submission cost is a real fraction of the step. Scheduling / command-buffer reuse are tunables.
* **The stack is buggy at the edges.** In this port alone: two Adreno driver bugs, one IREE runtime
  barrier bug, one IREE decomposition NaN. A tuner that measures speed without checking outputs
  will happily "optimise" into a wrong kernel. Correctness gating is mandatory, not optional.

## Architecture (layers, outermost first)

| Layer | Decides | Search space (examples) | Signal | Mechanism in IREE |
|---|---|---|---|---|
| **L0 Measure + guard** | nothing; gates the rest | none | median tok/s over repeats; thermal check; token/logit equality vs reference | on-device runner (`llm-runner`), `iree-benchmark-module`, reference outputs |
| **L1 Compile flags** | global pipeline choices | fusion (horizontal contractions, aggressive), memoization, indirect command buffers, partitioning favor, opt level, binding optimisation, const-expr hoisting | end-to-end prefill / decode tok/s | `iree-compile` flags (`scripts/tune/flag_tuner.py`) |
| **L2 Kernel configs** | per-dispatch codegen | pipeline, workgroup size, tile sizes, subgroup size, vector width, outputs per thread | per-dispatch µs × calls per step | dump configs → edit `translation_info` / `lowering_config` → `--iree-hal-substitute-executable-configurations-from` (or tuning specs, `--iree-codegen-tuning-spec-path`) (`scripts/tune/kernel_tuner.py`) |
| **L3 Export formulation** | how the model is written before IREE sees it | per-phase op forms (decode int4 GEMV as one fused reduction vs dequant + matmul), weight layout (int4 packed along K into int32), literals vs parameters | kernels per token, bytes per token, compile time | model rewrite before `torch.export` (`scripts/llm/llama_q4.py`) |
| **L4 Shape buckets** | which static variants exist | KV buckets, prefill chunk sizes, per-phase configs | tok/s across the context-length mix | exported entry points (`prefill_c{C}_l{L}`, `decode_l{L}`) |
| **L5 Runtime scheduler** | what runs when | bucket policy (smallest-fit vs fixed), chunk schedule, warm-up; later batching, prefill/decode interleave | TTFT, tok/s, p50/p95 latency | host runner (`runner/llm_runner.c`) |
| **L6 Numerics + memory** | precision and footprint | int4 group size, int4/int8 per layer, fp16 vs fp32 compute, KV dtype, KV capacity, hoisting | tok/s within an accuracy budget; peak RSS | export transforms, stream/HAL flags |

### Search strategy (per layer)
1. **Constraint pruning first** (free): hardware limits (shared memory, max threads, max binding
   range), known driver/compiler bugs (e.g. stock Adreno: power-of-two subgroup counts only),
   divisibility of tile sizes.
2. **Greedy / coordinate descent** for small discrete spaces (L1, L4, L5) — few compiles, robust.
3. **Exhaustive-within-budget or evolutionary** for L2 per hot kernel (spaces are ~10²–10³).
4. **Learned cost model** (later): features = op class, shapes, dtypes, config; target = measured
   µs; XGBoost/GBDT like TVM-Ansor; used to rank candidates so only the top-k are measured;
   transfers across sibling devices/driver versions.
5. **Profile-guided budget**: only tune dispatches covering ≥ 80 % of step time.

### Compile time is a constraint, not a side effect
Some IREE default configs take > 2 min *per kernel* to compile (attention batch-matmuls at
L = 4096; decode GEMVs with dynamically-offset bindings), and a full 11-entry-point 3B module took
> 40 min. The tuner therefore (a) tunes standalone dispatch benchmarks (seconds per candidate),
(b) caps per-candidate compile time and records timeouts as failures, (c) screens global flags on
a single-bucket proxy model before the full build.

### Correctness gate (every candidate)
* Kernel level: compare the candidate's outputs with the baseline kernel on random inputs.
* Model level: greedy tokens must equal the reference (eager PyTorch int4 model) for a fixed
  prompt; for precision changes (L6) use a logit-error / perplexity budget instead.
* Rejected candidates are recorded so they are never retried (they are bug reports, too).

### Result database
Key = (model hash, entry point, dispatch signature [op, shapes, dtypes], target, driver build,
compiler version). Value = best config + measured µs + all tried candidates. Artifacts emitted:
flag set, substituted-configuration directory / tuning spec, scheduler policy JSON. Re-tune only
what changed when the compiler/driver/model version moves.

### Measurement hygiene (phones)
Warm-up per entry point (pipeline creation), median of N, alternate A/B runs to cancel thermal
drift, cool-down between candidates, record SoC temperature/clocks with each sample.

## Roadmap
1. L0 + L5 (measurement, guard, bucketed scheduler) and L1 (flag tuner): implemented; flag search not yet run on 3B.
2. L3 + L2 for decode GEMVs (done for the qkv shape), then lm_head, prefill GEMMs, attention.
3. L4 bucket/chunk-set optimisation from a real context-length distribution.
4. L6 accuracy-guarded precision search (fp16 compute for prefill, int8 KV).
5. Cost model + cross-device transfer; CI that re-tunes on compiler/driver bumps.
6. Upstream: Adreno-aware defaults in IREE's SPIR-V `KernelConfig`; report the HAL barrier bug.

## Implemented so far (this repo)

| Component | File | Status |
|---|---|---|
| Kernel-config tuner (profile, tune standalone dispatch benchmarks, cross-pipeline candidates) | `scripts/tune/kernel_tuner.py` | used: qkv int4 GEMV 20.6 ms → 2.63 ms |
| Export formulation: fused decode GEMV, int4 packed along K into int32 | `scripts/llm/llama_q4.py` | used: one kernel instead of three; enables wide loads |
| Flag tuner (greedy coordinate search, token-equality guard) | `scripts/tune/flag_tuner.py` | written; not yet run on 3B |
| Runtime scheduler (chunked prefill, smallest KV bucket vs fixed 4K) | `runner/llm_runner.c` | verified on tiny Llama |
| Benchmark across context 256 to 4K and prompt kinds; summary tables | `scripts/llm/bench.py`, `summarize.py` | written |

Int4 GEMV (5120 x 3072) on Adreno 710 + Turnip + IREE 3.12, ms per call (median of 3, 20 dispatches
per submission): IREE default on the byte-packed layout 20.6 → K-packed layout default 14.85 →
tuned `SubgroupReduce` (64 threads, 32-group K tile) 3.54 → tuned `BaseVectorize` (2 outputs per
thread, 128 threads) **2.63**. Config search on the byte layout alone stopped at 10.5.
