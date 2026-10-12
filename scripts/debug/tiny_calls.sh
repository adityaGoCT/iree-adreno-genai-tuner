#!/usr/bin/env bash
# One stateless call per entry point on the phone, GPU vs CPU (zero KV caches).
D=/data/local/tmp/iree/llama-tiny
Z="--input=64x2x32xf16=0 --input=64x2x32xf16=0 --input=64x2x32xf16=0 --input=64x2x32xf16=0"
run() {  # fn dev module args...
  local fn=$1 dev=$2 mod=$3; shift 3
  local env=""; [[ "${ADRENO_DRIVER:-}" == turnip && $dev == vulkan ]] && env="LD_LIBRARY_PATH=/data/local/tmp/turnip"
  adb shell "cd $D && $env /data/local/tmp/iree/iree-run-module --device=$dev --module=$mod.vmfb --parameters=model=weights.irpa --function=$fn $* $Z" 2>&1 | tail -1 | cut -c1-120
}
for spec in "decode_l16|--input=1x1xi32=914 --input=1xi32=0 --input=1xi32=0" \
            "prefill_c8_l16|--input=1x8xi32=[5,77,300,914,12,8,999,450] --input=1xi32=0 --input=1xi32=7" \
            "decode_logits_l16|--input=1x1xi32=914 --input=1xi32=0 --input=1xi32=0"; do
  fn=${spec%%|*}; args=${spec#*|}
  echo "$fn GPU: $(run $fn vulkan adreno710 $args)"
  echo "$fn CPU: $(run $fn local-task android-cpu $args)"
done
