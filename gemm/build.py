"""GLSL -> SPIR-V with the NDK's glslc (shaderc), cached by source hash."""

import hashlib
import os
import pathlib
import subprocess
import threading

NDK = pathlib.Path(os.environ.get("ANDROID_NDK", "/opt/homebrew/share/android-ndk"))
GLSLC = next((p for p in (NDK / "shader-tools").glob("*/glslc")), None)
CACHE = pathlib.Path(__file__).resolve().parents[1] / "build/gemm/spv"


def compile_glsl(src: str, target_env="vulkan1.1"):
    """Returns (spv_path, None) or (None, error). Identical sources compile once."""
    CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(src.encode()).hexdigest()[:16]
    spv = CACHE / f"{h}.spv"
    if spv.exists():
        return spv, None
    # compiles run in parallel threads: work on private temp files, then rename atomically so a
    # concurrent caller never sees a half-written .comp / .spv
    tag = f"{os.getpid()}.{threading.get_ident()}"
    comp, tmp = CACHE / f"{h}.{tag}.comp", CACHE / f"{h}.{tag}.tmp"
    comp.write_text(src)
    r = subprocess.run([str(GLSLC), "-fshader-stage=compute", f"--target-env={target_env}", "-O",
                        str(comp), "-o", str(tmp)], capture_output=True, text=True)
    os.replace(comp, CACHE / f"{h}.comp")
    if r.returncode:
        tmp.unlink(missing_ok=True)
        err = next((l for l in r.stderr.splitlines() if "error" in l), r.stderr.strip()[:300])
        return None, err
    os.replace(tmp, spv)
    return spv, None


def build_candidate(shape, sched, kernels_fn):
    """Compile every dispatch of a candidate; returns ([(spv, grid)], None) or (None, error)."""
    out = []
    for name, src, grid in kernels_fn(shape, sched):
        spv, err = compile_glsl(src)
        if err:
            return None, f"{name}: {err}"
        out.append((spv, grid))
    return out, None
