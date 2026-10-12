"""Greedy reference generation on the Mac (eager PyTorch) for validating the phone.

  ref_generate.py --mode q4 --prompt "..." --new 16     # our int4 LlamaQ4, same schedule as the runner
  ref_generate.py --mode hf --prompt "..." --new 16     # original HF bf16 model (quality yardstick)

Writes <out>/ref_<mode>.json with prompt ids, generated ids and text; also the prompt file for
the runner (<out>/prompt_ref.txt).
"""

import argparse
import json
import pathlib
import sys
import time

import torch
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["q4", "hf"], default="q4")
    ap.add_argument("--prompt", default="Explain in two sentences why mobile GPUs are memory-bandwidth bound.")
    ap.add_argument("--new", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "models/llama32-3b-q4")
    a = ap.parse_args()
    snap = pathlib.Path(open(ROOT / "models/LLAMA_SNAPSHOT.txt").read().strip())
    tok = AutoTokenizer.from_pretrained(snap)
    ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = list(ids)
    (a.out / "prompt_ref.txt").write_text(" ".join(map(str, ids)))
    t0 = time.time()
    with torch.no_grad():
        if a.mode == "hf":
            from transformers import AutoModelForCausalLM
            m = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).eval()
            out = m.generate(torch.tensor([ids]), max_new_tokens=a.new, do_sample=False)
            gen = out[0, len(ids):].tolist()
        else:
            from llama_q4 import LlamaQ4
            m = LlamaQ4(snap, max_ctx=256).eval()
            caches = m.new_caches()
            C, n = a.chunk, len(ids)
            for s in range(0, n, C):
                chunk = torch.zeros(1, C, dtype=torch.int32)
                k = min(C, n - s)
                chunk[0, :k] = torch.tensor(ids[s:s + k])
                nxt, _ = m.step(chunk, torch.tensor([s], dtype=torch.int32), torch.tensor([k - 1], dtype=torch.int32),
                                caches, 256)
            gen = [int(nxt)]
            for i in range(1, a.new):
                nxt, _ = m.step(nxt.view(1, 1), torch.tensor([n + i - 1], dtype=torch.int32),
                                torch.tensor([0], dtype=torch.int32), caches, 256)
                gen.append(int(nxt))
    text = tok.decode(gen, skip_special_tokens=True)
    res = dict(mode=a.mode, prompt=a.prompt, prompt_ids=ids, tokens=gen, text=text, seconds=time.time() - t0)
    (a.out / f"ref_{a.mode}.json").write_text(json.dumps(res, indent=1))
    print(f"[{a.mode}] {len(ids)} prompt tokens, {time.time() - t0:.0f}s\n{gen}\n{text!r}")


if __name__ == "__main__":
    main()
