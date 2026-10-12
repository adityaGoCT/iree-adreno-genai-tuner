#!/usr/bin/env bash
# Cross-compile runner/llm_runner.c (+ IREE runtime v3.11.0) for the phone.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# IREE_VERSION=3.11 (default; pinned for the stock Qualcomm Vulkan 1.1 driver) or 3.12 (needs a
# Vulkan 1.3 driver, i.e. Mesa Turnip). Each version has its own source tree, venv and build dir.
IREE_VERSION="${IREE_VERSION:-3.11}"
if [[ "$IREE_VERSION" == 3.11 ]]; then SRC="$ROOT/third_party/iree"; VENV="$ROOT/.venv"; SUFFIX=""
else SRC="$ROOT/third_party/iree-$IREE_VERSION"; VENV="$ROOT/.venv-iree${IREE_VERSION/./}"; SUFFIX="${IREE_VERSION/./}"; fi
ANDROID_NDK="${ANDROID_NDK:-/opt/homebrew/share/android-ndk}"
PY="$VENV/bin/python"
cmake -G Ninja -S "$ROOT/runner" -B "$ROOT/build/runner${SUFFIX}-android" -DIREE_ROOT="$SRC" \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 -DCMAKE_BUILD_TYPE=Release \
  -DIREE_HOST_BIN_DIR="$VENV/lib/python3.12/site-packages/iree/_runtime_libs" \
  -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF -DIREE_BUILD_PYTHON_BINDINGS=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_VULKAN=ON \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_ENABLE_ASSERTIONS=OFF -DPython_EXECUTABLE="$PY" -DPython3_EXECUTABLE="$PY"
cmake --build "$ROOT/build/runner${SUFFIX}-android" --target llm-runner
