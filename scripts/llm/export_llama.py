"""Export LlamaQ4 to IREE: torch-dialect MLIR with static-shape entry points + int4 .irpa weights.

  export_llama.py --model-dir <hf snapshot> --out models/llama32-3b-q4 [--chunk 128]
                  [--buckets 256 512 1024 2048 4096] [--layers N]  (N<28: truncated dev model)

Entry points (consumed by runner/llm_runner.c):
  prefill_c{C}_l{L}(tokens[1,C] i32, start[1] i32, last[1] i32, k0, v0, ...) -> next[1] i32
  decode_l{L}      (tokens[1,1] i32, start[1] i32, last[1] i32, k0, v0, ...) -> next[1] i32
  decode_logits_l{L0} (same as decode_l{L0} but returns [1, vocab] f32)   -- for validation
"""

import argparse
import json
import pathlib
import sys
import time

import torch
from iree.turbine import aot

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from llama_q4 import LlamaQ4  # noqa: E402


def export(model, out: pathlib.Path, chunk, buckets, prompt=None, ref_tokens=None):
    import llama_q4
    llama_q4.EXPORT_FUSED_GEMV[0] = True  # decode projections -> single fused int4 GEMV kernels
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    aot.externalize_module_parameters(model, external_scope="model")
    aot.save_module_parameters(out / "weights.irpa", model)
    caches = model.new_caches()
    fxb = aot.FxProgramsBuilder(model)

    def add(name, C, L, logits=False):
        args = (torch.zeros(1, C, dtype=torch.int32), torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32), tuple(caches))

        def fn(m, tokens, start, last, kv):
            nxt, lg = m.step(tokens, start, last, kv, L)
            return lg if logits else nxt

        fxb.export_program(fn, name=name, args=args)

    for L in buckets:
        if L >= chunk:
            add(f"prefill_c{chunk}_l{L}", chunk, L)
        add(f"decode_l{L}", 1, L)
    add(f"decode_logits_l{buckets[0]}", 1, buckets[0], logits=True)
    aot.export(fxb).save_mlir(str(out / "model.mlir"))
    meta = dict(n_layers=model.n_layers, n_kv_heads=model.n_kv, head_dim=model.head_dim,
                max_ctx=model.max_ctx, vocab=model.embed.vocab, chunk=chunk, buckets=buckets,
                prompt=prompt, ref_tokens=ref_tokens)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"exported {out} in {time.time() - t0:.0f}s: model.mlir "
          f"{(out / 'model.mlir').stat().st_size / 1e6:.1f} MB, weights.irpa "
          f"{(out / 'weights.irpa').stat().st_size / 1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--buckets", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--group", type=int, default=32)
    a = ap.parse_args()
    t0 = time.time()
    model = LlamaQ4(a.model_dir, group=a.group, max_ctx=max(a.buckets), n_layers=a.layers).eval()
    print(f"loaded + quantized in {time.time() - t0:.0f}s")
    export(model, a.out, a.chunk, a.buckets)


if __name__ == "__main__":
    main()
