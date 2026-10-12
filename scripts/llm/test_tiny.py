"""Check LlamaQ4 (eager PyTorch) against HF LlamaForCausalLM on a tiny random Llama-3 config.

HF gets the *dequantized* int4 weights, so any mismatch is an implementation bug in LlamaQ4
(RoPE, GQA, masking, chunked prefill, KV-cache bucketing), not quantization error.
Also exports the tiny model (used to bring up the IREE/phone path quickly):
  test_tiny.py [--export DIR]
"""

import argparse
import pathlib
import sys

import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from llama_q4 import LlamaQ4, dequantize_q4, quantize_q4  # noqa: E402

TINY = dict(
    hidden_size=256, num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2, head_dim=32,
    intermediate_size=512, vocab_size=1000, rms_norm_eps=1e-5, rope_theta=500000.0,
    rope_scaling={"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                  "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
    max_position_embeddings=131072, tie_word_embeddings=True, hidden_act="silu",
    attention_bias=False, mlp_bias=False, initializer_range=0.2,
)
GROUP, CHUNK_ROWS, MAX_CTX = 32, 256, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=pathlib.Path)
    a = ap.parse_args()
    torch.manual_seed(0)
    hf = LlamaForCausalLM(LlamaConfig(**TINY, attn_implementation="eager")).eval()
    sd = {k: v.clone() for k, v in hf.state_dict().items()}
    ours = LlamaQ4(None, group=GROUP, max_ctx=MAX_CTX, vocab_chunk=CHUNK_ROWS, sd=sd, cfg=TINY).eval()

    # Give HF the same int4-rounded weights.
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if p.dim() == 2:
                p.copy_(dequantize_q4(*quantize_q4(p, GROUP), GROUP))

    prompt = torch.randint(0, TINY["vocab_size"], (1, 21))
    n_new, C, L = 10, 8, 32

    # HF reference: greedy with full recompute each step (no cache subtleties).
    ids = prompt.clone()
    hf_logits = []
    with torch.no_grad():
        for _ in range(n_new):
            lg = hf(ids).logits[0, -1]
            hf_logits.append(lg)
            ids = torch.cat([ids, lg.argmax().view(1, 1)], 1)
    hf_tokens = ids[0, prompt.shape[1]:].tolist()

    # Ours: chunked prefill (C=8, bucket 32) then single-token decode. fp32 cache here so the
    # comparison is exact (the deployed fp16 cache adds ~2e-2 logit error at this weight scale).
    caches = [c.float() for c in ours.new_caches()]
    n = prompt.shape[1]
    with torch.no_grad():
        for s in range(0, n, C):
            chunk = torch.zeros(1, C, dtype=torch.int32)
            m = min(C, n - s)
            chunk[0, :m] = prompt[0, s:s + m]
            nxt, lg = ours.step(chunk, torch.tensor([s], dtype=torch.int32),
                                torch.tensor([m - 1], dtype=torch.int32), caches, L)
        tokens, errs = [int(nxt)], [(lg[0] - hf_logits[0]).abs().max().item()]
        for i in range(1, n_new):
            nxt, lg = ours.step(nxt.view(1, 1), torch.tensor([n + i - 1], dtype=torch.int32),
                                torch.tensor([0], dtype=torch.int32), caches, L)
            tokens.append(int(nxt))
            errs.append((lg[0] - hf_logits[i]).abs().max().item())
    print("hf  :", hf_tokens)
    print("ours:", tokens)
    print("max |Δlogit| per step:", " ".join(f"{e:.1e}" for e in errs))
    ok = tokens == hf_tokens and max(errs) < 1e-3
    print("PASS" if ok else "FAIL")

    if a.export:
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        from export_llama import export
        export(ours, a.export, chunk=C, buckets=[16, 32, 64], prompt=prompt[0].tolist(), ref_tokens=tokens)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
