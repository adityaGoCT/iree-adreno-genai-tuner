#!/usr/bin/env bash
# Compile an exported model.mlir for one target.
#   scripts/compile.sh <model_dir> <target> [extra iree-compile flags...]
# targets (precision = whatever dtype the model was exported in; see export_hf.py --dtype):
#   host-cpu        Mac CPU (correctness reference for the frontend)
#   android-cpu     Phone CPU (arm64, Cortex-A78) baseline
#   adreno710       Adreno 710 GPU via Vulkan/SPIR-V with OUR target description
#                   (targets/adreno710.target, injected as hal.device.targets). USE THIS.
#   turnip710       Adreno 710 under Mesa Turnip (targets/turnip710.target): IREE's adreno
#                   target + 8-bit storage. Use with ADRENO_DRIVER=turnip. PREFERRED.
#   adreno-generic  IREE's built-in `--iree-vulkan-target=adreno`. Wrong results on the stock
#                   Qualcomm driver (subgroup-reduce bug); correct and preferred under Turnip.
# Note: --iree-input-demote-f32-to-f16 / -i64-to-i32 cannot retype dense_resource weight
# constants, so cast weights in PyTorch before export instead of demoting here.
# Output: <model_dir>/<target>.vmfb (override with OUT=path) (+ <target>.benchmarks/ with per-dispatch benchmarks)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$1"; TARGET="$2"; shift 2
# IREE_VERSION=3.11 (default, .venv) or 3.12 (.venv-iree312; outputs get a .iree312 suffix).
IREE_VERSION="${IREE_VERSION:-3.11}"
if [[ "$IREE_VERSION" == 3.11 ]]; then VENV="$ROOT/.venv"; SFX=""; else VENV="$ROOT/.venv-iree${IREE_VERSION/./}"; SFX=".iree${IREE_VERSION/./}"; fi
COMPILE="$VENV/bin/iree-compile"

# Adreno 710 driver has no shaderInt64: demote i64 (token ids, indices) to i32 everywhere.
COMMON=(--iree-input-type=torch --iree-input-demote-i64-to-i32)
# IREE 3.11's SPIR-V backend has no codegen for the fused iree_linalg_ext.attention op
# (from torch sdpa; falls back to linalg-ext-to-loops, which fails). Export with
# `--attn eager` instead (plain matmul/softmax/matmul). DECOMPOSE_ATTN=1 splits the fused op
# in preprocessing, but that currently yields NaNs even on CPU (masked attention) -- see docs/NOTES.md.
if [[ "${DECOMPOSE_ATTN:-0}" == 1 ]]; then
  COMMON+=('--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-linalg-ext-decompose-aggregated-ops{filter-ops=iree_linalg_ext.attention}))')
fi
case "$TARGET" in
  host-cpu)    FLAGS=(--iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-cpu=host) ;;
  adreno-generic) FLAGS=(--iree-hal-target-device=vulkan --iree-vulkan-target=adreno) ;;
  adreno710)   FLAGS=() ;;
  turnip710)   FLAGS=() ;;
  android-cpu) FLAGS=(--iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu
                      --iree-llvmcpu-target-triple=aarch64-linux-android29 --iree-llvmcpu-target-cpu=cortex-a78) ;;
  *) echo "unknown target $TARGET" >&2; exit 2 ;;
esac

INPUT=("$DIR/model.mlir")
if [[ "$TARGET" == adreno710 || "$TARGET" == turnip710 ]]; then
  # Attach the custom device target to the top-level module and stream it into iree-compile.
  TGT="$(tr -d '\n' < "$ROOT/targets/$TARGET.target")"
  INPUT=(-)
  exec 3< <(awk -v t="$TGT" 'NR==1 && /^module/ { sub(/ \{$/, " attributes {hal.device.targets = [" t "]} {") } { print }' "$DIR/model.mlir")
  exec 0<&3
fi

rm -rf "$DIR/$TARGET$SFX.benchmarks"
time "$COMPILE" "${INPUT[@]}" "${COMMON[@]}" ${FLAGS[@]+"${FLAGS[@]}"} \
  --iree-hal-dump-executable-benchmarks-to="$DIR/$TARGET$SFX.benchmarks" \
  "$@" -o "${OUT:-$DIR/$TARGET$SFX.vmfb}"
ls -la "${OUT:-$DIR/$TARGET$SFX.vmfb}"
