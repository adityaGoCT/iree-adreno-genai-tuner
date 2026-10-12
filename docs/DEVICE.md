# Target device

Queried 2026-10-11 over adb (Wi-Fi). Raw dump: `adb shell cmd gpu vkjson`; Turnip side: [vkinfo-turnip.txt](vkinfo-turnip.txt).

| | |
|---|---|
| Phone | Motorola moto g stylus 5G (2024), codename `boston` |
| SoC | Qualcomm SM6450 (Snapdragon 7s Gen 2), platform `parrot` |
| GPU | **Adreno 710** |
| OS | Android 14 (API 34), ABI arm64-v8a |
| GL driver | OpenGL ES 3.2 V@0615.80 (2024-01-05) |
| Vulkan | loader 1.3, **device API 1.1.128**, driver conformance 1.2.2 |
| OpenCL | `/vendor/lib64/libOpenCL.so` present |

## Compute-relevant Vulkan facts

| Property | Value | Implication |
|---|---|---|
| subgroupSize | 64 (`VK_EXT_subgroup_size_control` present) | tuner can try 64 vs 128 |
| subgroup ops mask | 191 (basic, vote, arith, ballot, shuffle, shuffle-rel, clustered, quad) | subgroup reductions OK |
| shaderFloat16 / shaderInt8 | yes / yes | fp16 compute is the default precision |
| storageBuffer16BitAccess | yes | fp16 tensors in SSBOs OK |
| uniformAndStorageBuffer16BitAccess | **no** | must not emit 16-bit uniform-buffer access |
| 8-bit storage | `VK_KHR_8bit_storage` present | int8/int4-packed weights OK |
| Integer dot product | **no** | no dot4 fast path for int8 |
| Cooperative matrix | **no** | matmul uses plain SIMT/vectorized pipeline |
| maxComputeSharedMemorySize | 32 KB | bounds workgroup-memory promotion |
| maxComputeWorkGroupInvocations | 1024 | |
| maxStorageBufferRange | **128 MB** | any single tensor/binding > 128 MB must be split |
| maxPushConstantsSize | 256 B | |
| maxBoundDescriptorSets | 7 | |
| Other | buffer_device_address, timeline_semaphore, synchronization2, push_descriptor, vulkan_memory_model, variable_pointers | |
