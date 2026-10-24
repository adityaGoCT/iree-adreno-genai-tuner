# Mojo → Adreno 710 (moto g stylus 5G 2024)

Mojo 1.2 (Modular) has GPU targets for NVIDIA, AMD and Apple only: no SPIR-V, Vulkan, OpenCL or
Adreno. This package gets ordinary Mojo GPU code (MAX `gpu` module, `global_idx`, `DeviceContext`)
running on the phone's Adreno 710 anyway, by taking the kernel from **Mojo's own LLVM IR** and
retargeting it. One command, four verified backends:

```
$ python3 mojo/mojo_adreno.py mojo/src/vecadd_gpu.mojo --n 1000003   # from the repo root
summary: turnip=PASS  stock=PASS  opencl=PASS  cpu=PASS
```

Write-up: [Part 3: Running Mojo GPU Kernels on an Adreno Phone GPU](../BLOG-MOJO.md).

| backend | path | result (n = 1,000,003) |
|---|---|---|
| `turnip` | Mojo IR → `mojo2vk.py` → LLVM 22 SPIR-V backend (SPIR-V 1.6) → Vulkan 1.3 on Mesa Turnip | PASS, 0 mismatches |
| `stock` | same, SPIR-V 1.4 + `VK_KHR_spirv_1_4`, int64 narrowed to int32 → Qualcomm Vulkan 1.1 driver | PASS, 0 mismatches |
| `opencl` | Mojo IR → `mojo2cl.py` → OpenCL C → Qualcomm OpenCL 3.0 compiler on the phone | PASS, 0 mismatches |
| `cpu` | `mojo build --emit object --target-triple aarch64-unknown-linux-android34` → NDK link | PASS, 0 mismatches |

A second kernel with a loop (4 elements per thread, phis), `src/vecadd4_gpu.mojo`, passes on all
three GPU backends too.

## How it works

1. `mojo build --emit llvm --target-accelerator sm_80 vecadd_gpu.mojo` writes the host IR plus one
   **LLVM IR sidecar per kernel** (NVPTX flavour): generic pointers, `llvm.nvvm.read.ptx.sreg.{tid,ctaid,ntid}`.
2. **`mojo2vk.py`** rewrites that kernel into a Vulkan compute shader for LLVM's SPIR-V backend:
   pointer arg *k* → storage buffer `(set 0, binding k)` via `llvm.spv.resource.handlefrombinding`;
   each GEP on it → `llvm.spv.resource.getpointer` (logical addressing, `addrspace(11)`);
   scalar args → int64 slots of a params buffer; NVPTX tid/ctaid → `llvm.spv.thread.id.in.group` /
   `llvm.spv.group.id`, ntid → the fixed workgroup size; `hlsl.shader="compute"` + `hlsl.numthreads`;
   then `llc` + `spirv-val`. `--i32 --vulkan 1.1` targets the stock driver (no shaderInt64, SPIR-V ≤ 1.4).
3. **`mojo2cl.py`** translates the same IR to OpenCL C (one variable per SSA value, labels + goto,
   phis resolved on edges; byte-addressed `__global uchar*` pointers; NVPTX registers →
   `get_local_id/get_group_id/get_local_size`).
4. `runner/vk_run.c` / `runner/cl_run.c` (phone side, `dlopen` of `libvulkan.so` / `libOpenCL.so`):
   A[i] = i, B[i] = 2i, C poisoned with NaN, one dispatch, every element checked.

Kernels can also skip MAX and read SPIR-V builtins directly from Mojo:
`llvm_intrinsic["llvm.spv.thread.id", Int32](Int32(0))` (`src/vecadd_spv.mojo`), compiled for the host
target; `mojo2vk.py` and `mojo2cl.py` take that IR as well (PASS on Turnip and OpenCL):

```
.venv-mojo/bin/mojo build --emit llvm -O3 -o build/mojo/spv/vecadd_spv.ll mojo/src/vecadd_spv.mojo
python3 mojo/mojo2vk.py build/mojo/spv/vecadd_spv.ll --kernel vecadd --local 64 -o build/mojo/spv/vecadd_spv.spv
python3 mojo/mojo2cl.py build/mojo/spv/vecadd_spv.ll --kernel vecadd -o build/mojo/spv/vecadd_spv.cl
```

## What did not work (and why)

| attempt | outcome |
|---|---|
| Mojo GPU target for Adreno / SPIR-V | none: accelerators are NVIDIA `sm_*`, AMD `gfx*`, Apple `apple-m*` only |
| OpenCL with SPIR-V IL (`clCreateProgramWithIL`) | `-59 CL_INVALID_OPERATION`: Qualcomm OpenCL 3.0 reports no IL version and no `cl_khr_il_program` (`probes/cl_probe.c`) |
| Running the kernel on the Mac's GPU for reference | needs Xcode's `metal` compiler (`xcrun metal` missing); not needed: the phone runners check every element |
| `ptr null` as the resource name in `handlefrombinding` | crashes LLVM 22's SPIR-V instruction selector; a name string per buffer works (adds the Int8 capability; both drivers have shaderInt8) |
| SPIR-V 1.3 for Vulkan 1.1 | LLVM 22's Vulkan output starts at SPIR-V 1.4; the stock driver accepts it via `VK_KHR_spirv_1_4` |
| Numbered entry block | Mojo's unlabeled entry block is `%<#args>`; phis name it, so retargeted bodies keep that label |
| `uv pip install mojo` / `max` | hung after downloading twice; a clean retry (`-v`) installed both |

## Files

```
mojo_adreno.py     one-command pipeline (build Mojo IR, retarget, push, run, verify)
mojo2vk.py         Mojo kernel IR -> Vulkan compute SPIR-V
mojo2cl.py         Mojo kernel IR -> OpenCL C
src/               vecadd_gpu.mojo (MAX gpu), vecadd4_gpu.mojo (loop), vecadd_spv.mojo (SPIR-V intrinsic), vecadd_cpu.mojo
runner/            vk_run.c, cl_run.c, cpu_main.c (phone side)
probes/cl_probe.c  OpenCL platform / IL capability probe
ref/               hand-written Vulkan IR, reference OpenCL kernel
```

Tools: Mojo 1.2.0 + MAX 26.7.0 in the repo's `.venv-mojo` (`scripts/setup_venvs.sh`); Homebrew
LLVM 22.1.2 (`llc` SPIR-V backend) and spirv-tools; Android NDK r30; Vulkan headers from
`third_party/iree` (`scripts/fetch_deps.sh`). Build outputs go to `build/mojo/`. The phone is reached
over adb Wi-Fi (`ANDROID_SERIAL`, default `192.168.1.7:5555`); the Turnip backend expects Mesa
Turnip in `/data/local/tmp/turnip` (`scripts/build_turnip.sh --push`).

## Limits

The retargeters cover the instruction subset of simple kernels (integer/float arithmetic, compares,
casts, GEP/load/store, branches, phis). Not yet: shared memory, barriers, atomics, subgroup ops,
`nctaid`/grid-stride loops, non-float element types in the runners, and timing beyond wall-clock
submit+wait.
