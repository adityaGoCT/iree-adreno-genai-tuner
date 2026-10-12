#!/usr/bin/env bash
# Fetch the pinned third-party sources into third_party/ and apply our patches.
#   third_party/iree       IREE v3.11.0  (runtime for the stock Qualcomm Vulkan 1.1 driver)
#   third_party/iree-3.12  IREE v3.12.0  (runtime for Mesa Turnip, Vulkan 1.3+)
#   third_party/mesa       Mesa main @ 745709a (Turnip with Adreno 710 support)
# Only runtime submodules are initialised (the compiler comes from pip).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"

fetch_iree() {  # <tag> <dir>
  local tag="$1" dir="$2"
  if [[ ! -d "$dir/.git" ]]; then
    git clone --depth 1 --branch "$tag" https://github.com/iree-org/iree.git "$dir"
  fi
  for s in $(cat "$dir/build_tools/scripts/git/runtime_submodules.txt"); do
    git -C "$dir" submodule update --init --depth 1 "$s"
  done
  local patch="$ROOT/runner/patches/iree-${tag}-hal-barrier-transfer-scope.patch"
  if git -C "$dir" apply --check "$patch" 2>/dev/null; then
    git -C "$dir" apply "$patch" && echo "applied $(basename "$patch")"
  else
    echo "patch already applied (or does not apply): $(basename "$patch")"
  fi
}

fetch_iree v3.11.0 "$TP/iree"
fetch_iree v3.12.0 "$TP/iree-3.12"

MESA_COMMIT=745709a000482a4a24c91faef5c5a40bd653a863
if [[ ! -d "$TP/mesa/.git" ]]; then
  git init -q "$TP/mesa"
  git -C "$TP/mesa" remote add origin https://gitlab.freedesktop.org/mesa/mesa.git
  git -C "$TP/mesa" fetch --depth 1 origin "$MESA_COMMIT"
  git -C "$TP/mesa" checkout -q FETCH_HEAD
fi
echo "third_party ready: $(ls "$TP")"
