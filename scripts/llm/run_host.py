"""Run an exported LlamaQ4 module on the Mac CPU through iree.runtime with the same schedule as
runner/llm_runner.c (chunked prefill, smallest KV bucket per step). Validates the compiled graph.

  run_host.py <model_dir> [--new 10] [--prompt-ids 1,2,3]   (default prompt/ref from meta.json)
"""

import argparse
import json
import pathlib

import iree.runtime as rt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=pathlib.Path)
    ap.add_argument("--target", default="host-cpu")
    ap.add_argument("--new", type=int, default=None)
    ap.add_argument("--prompt-ids", default=None)
    a = ap.parse_args()
    d = a.model_dir
    meta = json.loads((d / "meta.json").read_text())
    prompt = [int(t) for t in a.prompt_ids.split(",")] if a.prompt_ids else meta["prompt"]
    ref = meta.get("ref_tokens") if not a.prompt_ids else None
    n_new = a.new or (len(ref) if ref else 10)
    C, buckets = meta["chunk"], meta["buckets"]

    cfg = rt.Config("local-task")
    idx = rt.ParameterIndex()
    idx.load(str(d / "weights.irpa"))
    params = rt.create_io_parameters_module(cfg.vm_instance, idx.create_provider(scope="model"))
    hal = rt.create_hal_module(cfg.vm_instance, cfg.device)
    main_mod = rt.VmModule.mmap(cfg.vm_instance, str(d / f"{a.target}.vmfb"))
    ctx = rt.SystemContext(vm_modules=[params, hal, main_mod], config=cfg)
    fns = ctx.modules.module
    caches = [rt.asdevicearray(cfg.device, np.zeros((meta["max_ctx"], meta["n_kv_heads"], meta["head_dim"]),
                                                    np.float16)) for _ in range(2 * meta["n_layers"])]

    def bucket(n):
        return next(L for L in buckets if L >= n)

    i32 = lambda *v: np.array(v, np.int32)  # noqa: E731
    n = len(prompt)
    for s in range(0, n, C):
        m = min(C, n - s)
        chunk = np.zeros((1, C), np.int32)
        chunk[0, :m] = prompt[s:s + m]
        nxt = int(fns[f"prefill_c{C}_l{bucket(s + C)}"](chunk, i32(s), i32(m - 1), *caches).to_host()[0])
    toks = [nxt]
    for i in range(1, n_new):
        pos = n + i - 1
        nxt = int(fns[f"decode_l{bucket(pos + 1)}"](i32(nxt).reshape(1, 1), i32(pos), i32(0), *caches).to_host()[0])
        toks.append(nxt)
    print("iree:", toks)
    if ref:
        print("ref :", ref)
        print("MATCH" if toks == ref else "MISMATCH")


if __name__ == "__main__":
    main()
