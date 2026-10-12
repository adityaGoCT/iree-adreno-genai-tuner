# On-device LLM runner, built against the IREE runtime source (same pinned version, v3.11.0).
#   scripts/build_runner.sh
cmake_minimum_required(VERSION 3.21)
project(adreno_llm_runner C CXX)

set(IREE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/iree" CACHE PATH "IREE source tree")
add_subdirectory("${IREE_ROOT}" iree EXCLUDE_FROM_ALL)

add_executable(llm-runner llm_runner.c)
target_link_libraries(llm-runner
  iree_base_base
  iree_base_tooling_flags
  iree_hal_hal
  iree_modules_hal_types
  iree_tooling_context_util
  iree_vm_vm
)
if(EXISTS "${IREE_ROOT}/runtime/src/iree/hal/replay/recorder.h")
  target_link_libraries(llm-runner iree_hal_replay_recorder)
endif()
