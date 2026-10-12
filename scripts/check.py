"""Run a compiled model and compare against the PyTorch reference.

  check.py <model_dir> <target>              # host-cpu: runs locally via iree.runtime
  check.py <model_dir> <target> --device     # adreno-*/android-cpu: runs on the phone over adb

On-device runs push the vmfb + inputs to /data/local/tmp/iree/<model>/ and pull the output back.
"""

import argparse
import json
import pathlib
import subprocess

import numpy as np

# IREE_VERSION=3.12 -> use *.iree312.vmfb and the 3.12 runtime tools in /data/local/tmp/iree312.
import os as _os  # noqa: E402
SFX = "" if _os.environ.get("IREE_VERSION", "3.11") == "3.11" else ".iree" + _os.environ["IREE_VERSION"].replace(".", "")
TOOLS = "/data/local/tmp/iree" + SFX.replace(".iree", "")
DEVICE_ROOT = "/data/local/tmp/iree"
# ADRENO_DRIVER=turnip -> run Vulkan through Mesa Turnip (runner/turnip) instead of Qualcomm's driver.
import os  # noqa: E402
DRIVER_ENV = "LD_LIBRARY_PATH=/data/local/tmp/turnip " if os.environ.get("ADRENO_DRIVER") == "turnip" else ""
DRIVER = {"adreno710": "vulkan", "turnip710": "vulkan", "adreno-generic": "vulkan", "android-cpu": "local-task"}


def adb(*args, check=True):
    return subprocess.run(["adb", *args], check=check, capture_output=True, text=True)


def run_host(d, target, inputs):
    import iree.runtime as rt

    cfg = rt.Config("local-task")
    vmfb = rt.VmModule.mmap(cfg.vm_instance, str(d / f"{target}{SFX}.vmfb"))
    ctx = rt.SystemContext(vm_modules=[vmfb], config=cfg)
    return np.asarray(ctx.modules.module["main"](*inputs).to_host())


def run_device(d, target, input_files):
    remote = f"{DEVICE_ROOT}/{d.name}"
    adb("shell", "mkdir", "-p", remote)
    for f in [d / f"{target}{SFX}.vmfb", *input_files]:
        adb("push", str(f), f"{remote}/{f.name}")
    cmd = (f"cd {remote} && {DRIVER_ENV}{TOOLS}/iree-run-module --device={DRIVER[target]} "
           f"--module={target}{SFX}.vmfb --function=main "
           + " ".join(f"--input=@{f.name}" for f in input_files)
           + f" --output=@out_{target}.npy")
    r = adb("shell", cmd, check=False)
    if r.returncode != 0:
        raise SystemExit(f"device run failed:\n{r.stdout}\n{r.stderr}")
    adb("pull", f"{remote}/out_{target}.npy", str(d / f"out_{target}.npy"))
    return np.load(d / f"out_{target}.npy")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir", type=pathlib.Path)
    p.add_argument("target")
    p.add_argument("--device", action="store_true")
    a = p.parse_args()
    d = a.model_dir
    meta = json.loads((d / "meta.json").read_text())

    # Inputs were demoted i64 -> i32 at compile time, so feed int32.
    files = []
    for name in ["input_ids", "attention_mask"]:
        arr = np.load(d / f"{name}.npy").astype(np.int32)
        np.save(d / f"{name}.i32.npy", arr)
        files.append(d / f"{name}.i32.npy")
    inputs = [np.load(f) for f in files]

    out = run_device(d, a.target, files) if a.device else run_host(d, a.target, inputs)
    ref = np.load(d / "ref_logits.npy")
    out = out.astype(np.float32)

    def softmax(x):
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    err = np.abs(out - ref)
    print(f"[{a.target}] max|Δlogit|={err.max():.3e}  mean={err.mean():.3e}")
    for text, r, o in zip(meta["samples"], softmax(ref), softmax(out)):
        lr, lo = meta["id2label"][str(r.argmax())], meta["id2label"][str(o.argmax())]
        flag = "OK " if lr == lo else "MISMATCH"
        print(f"  {flag} torch={lr}:{r.max():.4f}  iree={lo}:{o.max():.4f}  {text!r}")


if __name__ == "__main__":
    main()
