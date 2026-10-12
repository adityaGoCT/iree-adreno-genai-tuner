// libvulkan.so shim: lets an unmodified IREE runtime (or any program that dlopen()s
// "libvulkan.so" and calls vkGetInstanceProcAddr) use Mesa Turnip instead of the vendor driver,
// without root and without touching the system. Only processes started with
//   LD_LIBRARY_PATH=/data/local/tmp/turnip
// see it. Undo: delete /data/local/tmp/turnip.
//
// Turnip built for Android exports only the HAL module symbol `HMI` (hwvulkan). This shim does
// what Android's loader does: open the HAL device "vk0" and forward vkGetInstanceProcAddr to it.
// The real system loader is bypassed entirely (no layers), so all calls go straight to Turnip.
//
// Driver path: $TURNIP_DRIVER, default "libvulkan_freedreno.so" (searched via LD_LIBRARY_PATH).
//
// Subgroup size: Turnip picks 64- or 128-wide waves per shader unless the pipeline requests a
// size. IREE's SPIR-V codegen assumes the target subgroup size (64 for Adreno) but only pins it
// for some pipelines, so cross-subgroup reductions come out wrong (e.g. 1280 instead of 768).
// The shim wraps vkCreateComputePipelines and adds VkPipelineShaderStageRequiredSubgroupSize-
// CreateInfo (= $TURNIP_SUBGROUP_SIZE, default 64; 0 disables) when the caller didn't.

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hardware/hwvulkan.h>

static hwvulkan_device_t* g_dev;
static PFN_vkGetDeviceProcAddr g_gdpa;
static PFN_vkCreateComputePipelines g_create_compute;
static uint32_t g_subgroup_size = 64;

static VKAPI_ATTR VkResult VKAPI_CALL create_compute_pipelines(
    VkDevice device, VkPipelineCache cache, uint32_t count, const VkComputePipelineCreateInfo* infos,
    const VkAllocationCallbacks* alloc, VkPipeline* pipelines) {
  if (!g_subgroup_size || count > 64) {
    return g_create_compute(device, cache, count, infos, alloc, pipelines);
  }
  VkComputePipelineCreateInfo patched[64];
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req[64];
  for (uint32_t i = 0; i < count; ++i) {
    patched[i] = infos[i];
    int has = 0;
    for (const VkBaseInStructure* n = (const VkBaseInStructure*)infos[i].stage.pNext; n; n = n->pNext) {
      if (n->sType == VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO) has = 1;
    }
    if (!has) {
      req[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
      req[i].pNext = infos[i].stage.pNext;
      req[i].requiredSubgroupSize = g_subgroup_size;
      patched[i].stage.pNext = &req[i];
    }
  }
  return g_create_compute(device, cache, count, patched, alloc, pipelines);
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device_proc_addr(VkDevice device, const char* name) {
  PFN_vkVoidFunction fn = g_gdpa(device, name);
  if (fn && strcmp(name, "vkCreateComputePipelines") == 0) {
    g_create_compute = (PFN_vkCreateComputePipelines)fn;
    return (PFN_vkVoidFunction)create_compute_pipelines;
  }
  return fn;
}

static VKAPI_ATTR VkResult VKAPI_CALL no_layers(uint32_t* count, VkLayerProperties* props) {
  (void)props;
  *count = 0;
  return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL instance_version(uint32_t* version) {
  *version = VK_API_VERSION_1_3;
  return VK_SUCCESS;
}

static int load(void) {
  if (g_dev) return 1;
  const char* path = getenv("TURNIP_DRIVER");
  if (!path) path = "libvulkan_freedreno.so";
  void* lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    fprintf(stderr, "vulkan_shim: dlopen(%s) failed: %s\n", path, dlerror());
    return 0;
  }
  hwvulkan_module_t* hmi = (hwvulkan_module_t*)dlsym(lib, HAL_MODULE_INFO_SYM_AS_STR);
  if (!hmi) {
    fprintf(stderr, "vulkan_shim: %s has no HMI symbol\n", path);
    return 0;
  }
  hw_device_t* dev = NULL;
  if (hmi->common.methods->open(&hmi->common, HWVULKAN_DEVICE_0, &dev) != 0 || !dev) {
    fprintf(stderr, "vulkan_shim: HAL open(vk0) failed\n");
    return 0;
  }
  g_dev = (hwvulkan_device_t*)dev;
  const char* sg = getenv("TURNIP_SUBGROUP_SIZE");
  if (sg) g_subgroup_size = (uint32_t)atoi(sg);
  if (getenv("TURNIP_SHIM_VERBOSE")) {
    fprintf(stderr, "vulkan_shim: using %s, required subgroup size %u\n", path, g_subgroup_size);
  }
  return 1;
}

__attribute__((visibility("default"))) VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance instance, const char* name) {
  if (!load()) return NULL;
  PFN_vkVoidFunction fn = g_dev->GetInstanceProcAddr(instance, name);
  if (fn && strcmp(name, "vkGetDeviceProcAddr") == 0) {
    g_gdpa = (PFN_vkGetDeviceProcAddr)fn;
    return (PFN_vkVoidFunction)get_device_proc_addr;
  }
  if (fn && strcmp(name, "vkCreateComputePipelines") == 0) {
    g_create_compute = (PFN_vkCreateComputePipelines)fn;
    return (PFN_vkVoidFunction)create_compute_pipelines;
  }
  if (!fn && instance == VK_NULL_HANDLE) {
    // Loader-provided globals an ICD may not implement.
    if (strcmp(name, "vkEnumerateInstanceLayerProperties") == 0) return (PFN_vkVoidFunction)no_layers;
    if (strcmp(name, "vkEnumerateInstanceVersion") == 0) return (PFN_vkVoidFunction)instance_version;
  }
  return fn;
}
