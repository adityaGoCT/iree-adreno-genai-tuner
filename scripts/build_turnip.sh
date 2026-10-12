#!/usr/bin/env bash
# Cross-build Mesa Turnip (Vulkan driver for Adreno) for Android arm64 + the libvulkan.so shim
# and the vkinfo tool, then optionally push them to /data/local/tmp/turnip on the phone.
#   scripts/build_turnip.sh [--push]
# Needs: Android NDK (ANDROID_NDK), meson, ninja, bison >= 3 (BISON_DIR), python with mako.
# Nothing on the phone's system partition is touched; processes opt in with
#   LD_LIBRARY_PATH=/data/local/tmp/turnip
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ANDROID_NDK="${ANDROID_NDK:-/opt/homebrew/share/android-ndk}"
BISON_DIR="${BISON_DIR:-/opt/homebrew/opt/bison/bin}"
HOST_TAG="$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64"   # NDK ships x86_64 (universal on macOS)
TC="$ANDROID_NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin"
API=34
MESA="$ROOT/third_party/mesa"
OUT="$ROOT/build/turnip-android"
CROSS="$ROOT/build/android-aarch64.cross"
mkdir -p "$ROOT/build"

cat > "$CROSS" <<EOF
[binaries]
ar = '$TC/llvm-ar'
c = ['$TC/aarch64-linux-android$API-clang']
cpp = ['$TC/aarch64-linux-android$API-clang++', '-fno-exceptions', '-fno-unwind-tables', '-fno-asynchronous-unwind-tables']
c_ld = 'lld'
cpp_ld = 'lld'
strip = '$TC/llvm-strip'
pkg-config = ['env', 'PKG_CONFIG_LIBDIR=/nonexistent', '/usr/bin/false']

[host_machine]
system = 'android'
cpu_family = 'aarch64'
cpu = 'armv8'
endian = 'little'

[built-in options]
# -static-libstdc++ must be a *link* arg: as a compile arg every meson
# cpp.get_supported_arguments() probe fails (unused-command-line-argument), which silently drops
# Mesa's own -Wno-c++11-narrowing and the build then fails with NDK r30 clang.
cpp_link_args = ['-static-libstdc++']
EOF

PY="${PYTHON:-$ROOT/.venv/bin/python}"
export PATH="$BISON_DIR:$(dirname "$PY"):$PATH"
(cd "$MESA" && meson setup --wipe "$OUT" --cross-file "$CROSS" \
  -Dbuildtype=release -Dplatforms=android -Dplatform-sdk-version=$API -Dandroid-stub=true \
  -Dandroid-libbacktrace=disabled -Dgallium-drivers= -Dvulkan-drivers=freedreno \
  -Dfreedreno-kmds=kgsl -Dvulkan-beta=true -Dopengl=false -Dgles1=disabled -Dgles2=disabled \
  -Degl=disabled -Dglx=disabled -Dgbm=disabled -Dglvnd=disabled -Dllvm=disabled \
  -Dxmlconfig=disabled -Dzstd=disabled -Dexpat=disabled -Dvalgrind=disabled -Dlibunwind=disabled \
  -Dlmsensors=disabled -Dperfetto=false -Dbuild-tests=false -Dspirv-tools=disabled -Dvulkan-layers=)
ninja -C "$OUT"

SHIM="$ROOT/build/turnip-shim"
mkdir -p "$SHIM"
"$TC/aarch64-linux-android$API-clang" -O2 -shared -fPIC -fvisibility=hidden \
  -I"$MESA/include/android_stub" -I"$MESA/include" \
  -o "$SHIM/libvulkan.so" "$ROOT/runner/turnip/vulkan_shim.c" -ldl
"$TC/aarch64-linux-android$API-clang" -O2 -I"$ROOT/third_party/iree/third_party/vulkan_headers/include" \
  -o "$SHIM/vkinfo" "$ROOT/runner/turnip/vkinfo.c" -ldl
ls -la "$OUT/src/freedreno/vulkan/libvulkan_freedreno.so" "$SHIM/libvulkan.so" "$SHIM/vkinfo"

if [[ "${1:-}" == --push ]]; then
  adb shell mkdir -p /data/local/tmp/turnip
  adb push "$OUT/src/freedreno/vulkan/libvulkan_freedreno.so" "$SHIM/libvulkan.so" "$SHIM/vkinfo" /data/local/tmp/turnip/
  adb shell 'cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp/turnip ./turnip/vkinfo | head -8'
fi
