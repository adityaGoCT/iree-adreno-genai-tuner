"""Phone side of the tuner: push shapes/kernels over adb, run the Vulkan harness, parse results.

The harness runs a whole batch of candidates in one process (one adb round trip per batch).
The phone is reached over adb Wi-Fi only (ADB_WIFI=ip:port, default below); the USB transport is
never used, even when a cable is attached. DRIVER=turnip|stock picks the Vulkan driver.
"""

import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV = "/data/local/tmp/gemm"
TURNIP = "/data/local/tmp/turnip"
HARNESS = ROOT / "build/gemm/vk_gemm_bench"
WIFI = os.environ.get("ADB_WIFI", "192.168.1.7:5555")
os.environ["ANDROID_SERIAL"] = WIFI  # pin every adb call to the Wi-Fi transport


def reconnect():
    """Re-establish the Wi-Fi transport (adb connect is a no-op when it is already up)."""
    subprocess.run(["adb", "connect", WIFI], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=20)


def adb(*args, check=True, timeout=None, retries=8):
    """adb over Wi-Fi with retries: the client intermittently fails (exit 1 / transport closed).
    After two failures each retry first re-runs `adb connect`. Harness runs
    (`shell ... vk_gemm_bench`) are not retried here; run_batch handles their failures."""
    import time
    harness = len(args) > 1 and "vk_gemm_bench" in args[-1]
    for attempt in range(1 if harness else retries):
        r = subprocess.run(["adb", *args], capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        if r.returncode == 0:
            break
        time.sleep(min(4.0, 0.5 * (attempt + 1)))
        if attempt >= 1:
            reconnect()
    if check and r.returncode:
        raise subprocess.CalledProcessError(r.returncode, ["adb", *args], r.stdout, r.stderr)
    return r


def setup(shapes, data_root: pathlib.Path):
    """Push the harness and every shape's data once."""
    from shapes import make_data
    adb("shell", f"mkdir -p {DEV}/shapes {DEV}/kern")
    adb("push", str(HARNESS), f"{DEV}/vk_gemm_bench")
    for s in shapes:
        d = data_root / s.id
        if not (d / "meta.txt").exists():
            make_data(s, d)
        files = [f for f in ("meta.txt", "A.bin", "B.bin", "Cref.bin") if (d / f).exists()]
        want = " ".join(str((d / f).stat().st_size) for f in files)
        have = adb("shell", "stat -c %s " + " ".join(f"{DEV}/shapes/{s.id}/{f}" for f in files) + " 2>/dev/null",
                   check=False).stdout.split()
        if " ".join(have) != want:  # (re)push unless every file is there with the right size
            adb("shell", f"mkdir -p {DEV}/shapes/{s.id}")
            for f in ("meta.txt", "A.bin", "B.bin", "Cref.bin"):
                if (d / f).exists():
                    adb("push", str(d / f), f"{DEV}/shapes/{s.id}/{f}")


WARMUP_MS = float(os.environ.get("GEMM_WARMUP_MS", "500"))


def run_batch(cands, local_dir: pathlib.Path, tag: str, driver="turnip", target_ms=25, reps=5, timeout=900):
    """cands: list of dicts {name, shape, sg, dispatches: [(spv_path, (gx, gy, gz)), ...]}.
    Pushes the SPIR-V, runs the harness once, returns {name: result}. A GPU hang (device_lost)
    aborts the harness; the remaining candidates are re-run in a fresh process."""
    remote = f"{DEV}/kern/{tag}"
    adb("shell", f"rm -rf {remote} && mkdir -p {remote}")
    # a dispatch's spv may carry an entry point: "path.spv@entry" (default entry: main)
    spvs = sorted({pathlib.Path(str(p).split("@")[0]) for c in cands for p, _ in c["dispatches"]})
    # push the whole directory of this batch's .spv files in one adb call
    stage = local_dir / f"push_{tag}"
    stage.mkdir(parents=True, exist_ok=True)
    for p in spvs:
        (stage / p.name).write_bytes(p.read_bytes())
    adb("push", f"{stage}/.", remote)
    results, todo, transport_retries = {}, list(cands), 0
    env = f"LD_LIBRARY_PATH={TURNIP} " if driver == "turnip" else ""
    while todo:
        lines = []
        for c in todo:
            parts = [c["name"], f"{DEV}/shapes/{c['shape']}", str(c.get("sg", 0)), str(len(c["dispatches"]))]
            for p, g in c["dispatches"]:
                parts += [f"{remote}/{pathlib.Path(str(p)).name}", *map(str, g)]
            lines.append(" ".join(parts))
        (local_dir / f"manifest_{tag}.txt").write_text("\n".join(lines) + "\n")
        adb("push", str(local_dir / f"manifest_{tag}.txt"), f"{remote}/manifest.txt")
        r = adb("shell", f"cd {remote} && {env}{DEV}/vk_gemm_bench --manifest=manifest.txt "
                         f"--target-ms={target_ms} --reps={reps} --warmup-ms={WARMUP_MS:.0f}", check=False, timeout=timeout)
        for line in r.stdout.splitlines():
            if line.startswith("{"):
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    import re
                    m = re.search(r'"name":"([^"]+)"', line)
                    if not m:
                        continue
                    o = {"name": m.group(1), "ok": False, "error": "bad json: " + line[:120]}
                results[o["name"]] = o
        rest = [c for c in todo if c["name"] not in results]
        if not rest:
            break
        if r.returncode == 255 and transport_retries < 4:
            # 255 is adb's own exit code (transport dropped), never the harness's: rerun the
            # remaining candidates without blaming one of them
            transport_retries += 1
            reconnect()
            adb("push", f"{stage}/.", remote)
            todo = rest
            continue
        # Harness died mid-batch. If it reported a device_lost for the candidate it was running,
        # that one is already recorded; otherwise blame the next unreported candidate.
        lost = any(results[c["name"]].get("error") == "device_lost" for c in todo if c["name"] in results
                   and c not in rest)
        if not lost:
            results[rest[0]["name"]] = {"name": rest[0]["name"], "ok": False,
                                        "error": f"harness exit {r.returncode}: {r.stderr[-200:]}"}
            rest = rest[1:]
        todo = rest
    return results
