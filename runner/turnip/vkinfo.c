// Minimal Vulkan capability dump for compute (no loader layers, no WSI).
// Stock driver: ./vkinfo     Turnip: LD_LIBRARY_PATH=/data/local/tmp/turnip ./vkinfo
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define GIPA(inst, name) PFN_##name name = (PFN_##name)gipa(inst, #name)

int main(void) {
  void* lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
  if (!lib) { fprintf(stderr, "dlopen libvulkan.so: %s\n", dlerror()); return 1; }
  PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
  if (!gipa) gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "_ZN11qglinternal21vkGetInstanceProcAddrEP12VkInstance_TPKc");
  GIPA(NULL, vkCreateInstance);
  VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "vkinfo", 1, NULL, 0, VK_API_VERSION_1_3};
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL};
  VkInstance inst;
  if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { fprintf(stderr, "vkCreateInstance failed\n"); return 1; }
  GIPA(inst, vkEnumeratePhysicalDevices);
  GIPA(inst, vkGetPhysicalDeviceProperties2);
  GIPA(inst, vkGetPhysicalDeviceFeatures2);
  GIPA(inst, vkEnumerateDeviceExtensionProperties);
  uint32_t n = 1;
  VkPhysicalDevice pd;
  vkEnumeratePhysicalDevices(inst, &n, &pd);

  VkPhysicalDeviceSubgroupSizeControlProperties ssc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
  VkPhysicalDeviceSubgroupProperties sg = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, &ssc};
  VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, &sg};
  VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv};
  vkGetPhysicalDeviceProperties2(pd, &p2);
  VkPhysicalDeviceProperties* p = &p2.properties;
  printf("device: %s\ndriver: %s %s\napi: %u.%u.%u\n", p->deviceName, drv.driverName, drv.driverInfo,
         VK_API_VERSION_MAJOR(p->apiVersion), VK_API_VERSION_MINOR(p->apiVersion), VK_API_VERSION_PATCH(p->apiVersion));
  printf("subgroup: size=%u ops=0x%x min=%u max=%u requiredStages=0x%x\n", sg.subgroupSize,
         sg.supportedOperations, ssc.minSubgroupSize, ssc.maxSubgroupSize, ssc.requiredSubgroupSizeStages);
  printf("maxStorageBufferRange=%u MB  maxComputeSharedMemorySize=%u KB  maxComputeWorkGroupInvocations=%u\n",
         p->limits.maxStorageBufferRange >> 20, p->limits.maxComputeSharedMemorySize >> 10,
         p->limits.maxComputeWorkGroupInvocations);

  VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
  VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
  if (VK_API_VERSION_MINOR(p->apiVersion) >= 3) {
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    printf("1.2: timelineSemaphore=%u scalarBlockLayout=%u bufferDeviceAddress=%u shaderFloat16=%u shaderInt8=%u "
           "storageBuffer8BitAccess=%u vulkanMemoryModel=%u\n", f12.timelineSemaphore, f12.scalarBlockLayout,
           f12.bufferDeviceAddress, f12.shaderFloat16, f12.shaderInt8, f12.storageBuffer8BitAccess,
           f12.vulkanMemoryModel);
    printf("1.3: synchronization2=%u subgroupSizeControl=%u computeFullSubgroups=%u shaderIntegerDotProduct=%u "
           "maintenance4=%u\n", f13.synchronization2, f13.subgroupSizeControl, f13.computeFullSubgroups,
           f13.shaderIntegerDotProduct, f13.maintenance4);
  }
  printf("shaderInt64=%u shaderInt16=%u\n", f2.features.shaderInt64, f2.features.shaderInt16);
  uint32_t ne = 0;
  vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL);
  VkExtensionProperties ext[512];
  if (ne > 512) ne = 512;
  vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, ext);
  printf("extensions (%u), compute-relevant:", ne);
  const char* keys[] = {"cooperative_matrix", "integer_dot", "shader_float16", "8bit", "16bit", "subgroup",
                        "bfloat16", "shader_float8", "push_descriptor", "buffer_device", "maintenance5"};
  for (uint32_t i = 0; i < ne; ++i)
    for (unsigned k = 0; k < sizeof(keys) / sizeof(keys[0]); ++k)
      if (strstr(ext[i].extensionName, keys[k])) { printf(" %s", ext[i].extensionName); break; }
  printf("\n");
  return 0;
}
