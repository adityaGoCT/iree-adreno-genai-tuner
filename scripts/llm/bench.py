"""Benchmark the int4 Llama on the phone across prompt types and context lengths.

  bench.py --model-dir models/llama32-3b-q4 --vmfb turnip710.iree312.vmfb \
           [--ctx 256 512 1024 2048 4096] [--kinds prose code chat] [--new 64]
           [--schedule bucketed max] [--chunk 128] [--tag baseline]

For each (kind, ctx): builds a prompt of ctx - new tokens (Llama-3 chat template), pushes it,
runs runner/llm_runner.c on the GPU (Turnip), decodes the generated text, and appends a JSON line
to <model-dir>/bench/<tag>.jsonl. Context length = prompt + generated tokens.
"""

import argparse
import json
import pathlib
import subprocess
import sys

from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEV = "/data/local/tmp/iree/llama32-3b-q4"
RUNNER = "/data/local/tmp/iree312/llm-runner"
ENV = "LD_LIBRARY_PATH=/data/local/tmp/turnip"


def corpus(kind: str) -> str:
    """Local, license-safe filler text of three different flavours."""
    if kind == "code":  # Python stdlib source (PSF licensed) from the venv's interpreter
        import inspect, json as j, textwrap, argparse as ap  # noqa: E401
        return "\n".join(inspect.getsource(m) for m in (textwrap, j.decoder, ap))
    if kind == "prose":  # project's own notes + README files we wrote
        return "\n\n".join(p.read_text() for p in [ROOT / "docs/NOTES.md", ROOT / "docs/DEVICE.md", ROOT / "docs/TUNER_PLAN.md"])
    if kind == "chat":  # synthetic multi-turn support conversation
        turns = []
        for i in range(400):
            turns.append(f"User: My phone's GPU run #{i} took {(i * 37) % 900 + 100} ms; is that slower than run "
                         f"#{max(0, i - 1)}? Battery was at {(i * 13) % 100}%.")
            turns.append(f"Assistant: Run #{i} vs #{max(0, i - 1)}: the difference is mostly thermal state and "
                         f"clock scaling; compare medians over several runs rather than single samples.")
        return "\n".join(turns)
    raise ValueError(kind)


INSTRUCTIONS = {
    "prose": "Summarize the key findings in the document above in three bullet points.",
    "code": "Explain what the code above does, briefly.",
    "chat": "Based on the conversation above, give one sentence of advice.",
}


def build_prompt(tok, kind, n_tokens):
    """Chat-formatted prompt of exactly n_tokens tokens (body truncated to fit)."""
    head = tok.apply_chat_template([{"role": "user", "content": "<<BODY>>\n\n" + INSTRUCTIONS[kind]}],
                                   tokenize=False, add_generation_prompt=True)
    pre, post = head.split("<<BODY>>")
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    post_ids = tok(post, add_special_tokens=False)["input_ids"]
    body = tok(corpus(kind), add_special_tokens=False)["input_ids"]
    need = n_tokens - len(pre_ids) - len(post_ids)
    while len(body) < need:
        body = body + body
    return pre_ids + body[:need] + post_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=pathlib.Path, default=ROOT / "models/llama32-3b-q4")
    ap.add_argument("--vmfb", default="turnip710.iree312.vmfb")
    ap.add_argument("--ctx", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--kinds", nargs="+", default=["prose", "code", "chat"])
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--schedule", nargs="+", default=["bucketed"])
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--buckets", default="256,512,1024,2048,4096")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--extra", default="", help="extra llm-runner flags")
    a = ap.parse_args()

    meta = json.loads((a.model_dir / "meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(pathlib.Path(open(ROOT / "models/LLAMA_SNAPSHOT.txt").read().strip()))
    out = a.model_dir / "bench" / f"{a.tag}.jsonl"
    out.parent.mkdir(exist_ok=True)
    for kind in a.kinds:
        for ctx in a.ctx:
            ids = build_prompt(tok, kind, ctx - a.new)
            pf = a.model_dir / "bench" / f"prompt_{kind}_{ctx}.txt"
            pf.write_text(" ".join(map(str, ids)))
            subprocess.run(["adb", "push", str(pf), f"{DEV}/{pf.name}"], check=True, capture_output=True)
            for sched in a.schedule:
                cmd = (f"cd {DEV} && {ENV} {RUNNER} --device=vulkan --module={a.vmfb} "
                       f"--parameters=model=weights.irpa --prompt={pf.name} --max_new_tokens={a.new} "
                       f"--chunk={a.chunk} --buckets={a.buckets} --schedule={sched} "
                       f"--n_layers={meta['n_layers']} --n_kv_heads={meta['n_kv_heads']} "
                       f"--head_dim={meta['head_dim']} --max_ctx={meta['max_ctx']} {a.extra}")
                r = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True)
                line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
                if not line:
                    print(f"{kind:5s} ctx={ctx:5d} {sched}: FAILED\n{r.stdout[-800:]}\n{r.stderr[-800:]}", flush=True)
                    continue
                res = json.loads(line)
                res.update(kind=kind, ctx=ctx, tag=a.tag, vmfb=a.vmfb,
                           text=tok.decode(res["tokens"], skip_special_tokens=True))
                with open(out, "a") as f:
                    f.write(json.dumps(res) + "\n")
                print(f"{kind:5s} ctx={ctx:5d} {sched:8s} prompt={res['prompt_tokens']:5d} "
                      f"prefill {res['prefill_tok_s']:7.1f} tok/s (TTFT {res['ttft_ms'] / 1e3:6.2f}s)  "
                      f"decode {res['decode_tok_s']:5.2f} tok/s  | {res['text'][:60]!r}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
