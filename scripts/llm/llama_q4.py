"""Minimal Llama-3 (int4 weight-only) written for IREE/SPIR-V export to Adreno.

Design constraints come from NOTES.md:
  * weights int4 symmetric, group-wise along K, packed along K into int32 words (8 nibbles per
    word). Decode GEMVs are written as one fused contraction that unpacks in registers
    (QLinear._gemv_fused); prefill dequantizes once per chunk and runs a GEMM.
  * activations, norms, softmax and attention math in fp32 (fp16 reductions overflow on the GPU);
    KV cache stored fp16.
  * KV cache = function arguments [max_ctx, kv_heads, head_dim] updated in place with
    index_copy_ along dim 0 (dim>0 emits a None index IREE 3.11 rejects).
  * attention reads only cache[:L] (L = static "bucket") -> decode cost tracks context length.
  * every tensor bound to a kernel < 128 MB (maxStorageBufferRange): vocab split in chunks.
  * q/k/v and gate/up fused into single projections (fewer, wider GEMV dispatches per token).
"""

import json
import math
import pathlib

import torch
import torch.nn.functional as F
from torch import nn


def quantize_q4(w: torch.Tensor, group: int):
    """w [N, K] float -> (qweight [N, K/8] int32, scales [N, K/group] f16).
    Symmetric int4 per group of `group` consecutive k. Packed along K, 8 nibbles per int32 word:
    nibble b (bits 4b..4b+3) of word w holds k = 8*w + b, stored as q + 8 in [0, 15].
    K-packing keeps every row contiguous and lets the GPU load 4/16 bytes at a time."""
    n, k = w.shape
    assert k % group == 0 and group % 8 == 0, (n, k, group)
    wg = w.float().reshape(n, k // group, group)
    scale = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7.0
    q = ((wg / scale).round().clamp(-8, 7).to(torch.int64) + 8).reshape(n, k // 8, 8)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    packed = (q << shifts).sum(-1)  # [n, k/8] in [0, 2^32)
    packed = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)
    return packed.contiguous(), scale.reshape(n, k // group).to(torch.float16).contiguous()


def unpack_q4(qweight):
    """[N, K/8] int32 -> [N, K] int nibbles in [0, 15] (interleaving stack: not fusable; used
    only where a materialized dequant is fine: prefill GEMM, embedding rows, eager reference)."""
    nibs = [(qweight >> (4 * b)) & 15 for b in range(8)]
    return torch.stack(nibs, -1).reshape(qweight.shape[0], -1)


def dequantize_q4(qweight, scales, group):
    q = unpack_q4(qweight).to(torch.float32) - 8
    n, k = q.shape
    return (q.reshape(n, k // group, group) * scales.float()[..., None]).reshape(n, k)


# Set to [True] at export time: decode (M=1) projections use the fused int4 GEMV formulation.
# Eager runs keep the dequant + F.linear path (the fused broadcast form would materialize a huge
# intermediate in eager PyTorch).
EXPORT_FUSED_GEMV = [False]


class QLinear(nn.Module):
    def __init__(self, w: torch.Tensor, group: int):
        super().__init__()
        qweight, scales = quantize_q4(w, group)
        self.qweight = nn.Parameter(qweight, requires_grad=False)
        self.scales = nn.Parameter(scales, requires_grad=False)
        self.group = group
        self.n, self.k = w.shape

    def forward(self, x):
        if x.shape[0] == 1 and EXPORT_FUSED_GEMV[0]:
            return self._gemv_fused(x)
        return F.linear(x, dequantize_q4(self.qweight, self.scales, self.group))

    def _gemv_fused(self, x):
        """Decode (M=1): ONE fused contraction; the 8 nibbles of each int32 word are extracted with
        scalar shifts in registers and multiplied with the matching (strided) activations:
          y[n] = sum_{g,w} s[n,g] * sum_{b<8} x[g, 8w+b] * (((q[n,g,w] >> 4b) & 15) - 8)
        Every op is elementwise over [N, K/G, G/8] -> IREE fuses it all into the reduction."""
        n, kg, wpg = self.n, self.k // self.group, self.group // 8
        q = self.qweight.reshape(n, kg, wpg)
        x4 = x.reshape(kg, wpg, 8)
        s = self.scales.to(torch.float32)[:, :, None]
        acc = None
        for b in range(8):
            # scale folded into every term so the final op is a *single-input* sum: IREE then fuses
            # the whole elementwise chain into the reduction (a 2-input mul+sum is treated as a
            # contraction and the producer is NOT fused -> a materialized [N, K/8] fp32 tensor).
            term = x4[:, :, b] * ((((q >> (4 * b)) & 15).to(torch.float32) - 8.0) * s)
            acc = term if acc is None else acc + term
        return acc.sum(dim=(1, 2)).reshape(1, n)


class QEmbedding(nn.Module):
    """Tied embedding / lm_head stored as int4 vocab chunks (each chunk < 128 MB on the GPU)."""

    def __init__(self, w: torch.Tensor, group: int, chunk_rows: int):
        super().__init__()
        self.vocab, self.dim = w.shape
        self.chunk_rows = chunk_rows
        self.group = group
        # Quantize chunk by chunk (each slice converted to fp32 on its own) to bound peak memory.
        self.chunks = nn.ModuleList(
            QLinear(w[i : i + chunk_rows], group) for i in range(0, self.vocab, chunk_rows)
        )

    def lookup(self, tokens):  # tokens [C] int64 -> [C, dim] f32
        out = None
        for c, ql in enumerate(self.chunks):
            rows = ql.n
            raw = tokens - c * self.chunk_rows
            idx = raw.clamp(0, rows - 1)
            valid = idx == raw  # token is in this chunk (bool `&` doesn't lower in IREE 3.11)
            words = ql.qweight.index_select(0, idx)  # [C, dim/8] int32
            q = unpack_q4(words).to(torch.float32) - 8.0  # [C, dim]
            s = ql.scales.index_select(0, idx).to(torch.float32)
            e = (q.reshape(-1, self.dim // self.group, self.group) * s[..., None]).reshape(-1, self.dim)
            e = torch.where(valid[:, None], e, torch.zeros_like(e))
            out = e if out is None else out + e
        return out

    def logits(self, x):  # x [M, dim] -> [M, vocab]
        return torch.cat([ql(x) for ql in self.chunks], -1)[:, : self.vocab]


class LazySafetensors:
    """dict-like view over HF safetensors shards; tensors are read (mmap) only when accessed, so
    the 3B checkpoint is quantized one tensor at a time instead of being loaded whole."""

    def __init__(self, model_dir: pathlib.Path):
        from safetensors import safe_open
        self.handles = [safe_open(str(f), "pt") for f in sorted(model_dir.glob("*.safetensors"))]
        self.where = {k: h for h in self.handles for k in h.keys()}

    def __getitem__(self, k):
        return self.where[k].get_tensor(k)

    def __contains__(self, k):
        return k in self.where


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class Layer(nn.Module):
    def __init__(self, cfg, sd, i, group):
        super().__init__()
        p = f"model.layers.{i}."
        self.wqkv = QLinear(torch.cat([sd[p + "self_attn.q_proj.weight"], sd[p + "self_attn.k_proj.weight"],
                                       sd[p + "self_attn.v_proj.weight"]], 0), group)
        self.wo = QLinear(sd[p + "self_attn.o_proj.weight"], group)
        self.wgu = QLinear(torch.cat([sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]], 0), group)
        self.wdown = QLinear(sd[p + "mlp.down_proj.weight"], group)
        self.ln_in = nn.Parameter(sd[p + "input_layernorm.weight"].float(), requires_grad=False)
        self.ln_post = nn.Parameter(sd[p + "post_attention_layernorm.weight"].float(), requires_grad=False)


class LlamaQ4(nn.Module):
    def __init__(self, model_dir, group=32, max_ctx=4096, vocab_chunk=32064, n_layers=None, sd=None, cfg=None):
        super().__init__()
        model_dir = pathlib.Path(model_dir) if model_dir else None
        cfg = cfg or json.loads((model_dir / "config.json").read_text())
        if sd is None:
            sd = LazySafetensors(model_dir)
        self.cfg = cfg
        self.n_layers = n_layers or cfg["num_hidden_layers"]
        self.dim = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim") or self.dim // self.n_heads
        self.inter = cfg["intermediate_size"]
        self.eps = cfg["rms_norm_eps"]
        self.max_ctx = max_ctx
        emb = sd["model.embed_tokens.weight"]
        self.embed = QEmbedding(emb, group, vocab_chunk)
        if not cfg.get("tie_word_embeddings", False):
            raise NotImplementedError("untied lm_head")
        self.layers = nn.ModuleList(Layer(cfg, sd, i, group) for i in range(self.n_layers))
        self.ln_f = nn.Parameter(sd["model.norm.weight"].float(), requires_grad=False)
        cos, sin = self._rope_tables(cfg, max_ctx)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _rope_tables(self, cfg, max_ctx):
        # Use HF's own rope init (handles llama3 frequency scaling) so we match it exactly.
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        rot = LlamaRotaryEmbedding(LlamaConfig(**cfg))
        pos = torch.arange(max_ctx)[None]
        cos, sin = rot(torch.zeros(1, dtype=torch.float32), pos)
        return cos[0].float(), sin[0].float()  # [max_ctx, head_dim]

    @staticmethod
    def _rotate_half(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], -1)

    def step(self, tokens, start, last, caches, L):
        """tokens [1,C] i32, start [1] i32, last [1] i32, caches = (k0, v0, k1, v1, ...)
        Writes KV for positions start..start+C-1, attends over cache[:L], returns greedy next
        token id for the row `last` as [1] int32."""
        C = tokens.shape[1]
        pos = start.to(torch.int64) + torch.arange(C, dtype=torch.int64)  # [C]
        h = self.embed.lookup(tokens[0].to(torch.int64))  # [C, D] f32
        cos = self.cos.index_select(0, pos)[:, None, :]
        sin = self.sin.index_select(0, pos)[:, None, :]
        key_pos = torch.arange(L, dtype=torch.int64)
        mask = key_pos[None, :] <= pos[:, None]  # [C, L] causal (also hides stale cache rows)
        g = self.n_heads // self.n_kv
        hd = self.head_dim
        scale = 1.0 / math.sqrt(hd)
        for i, layer in enumerate(self.layers):
            k_cache, v_cache = caches[2 * i], caches[2 * i + 1]
            x = rms_norm(h, layer.ln_in, self.eps)
            qkv = layer.wqkv(x)
            q = qkv[:, : self.n_heads * hd].reshape(C, self.n_heads, hd)
            k = qkv[:, self.n_heads * hd : (self.n_heads + self.n_kv) * hd].reshape(C, self.n_kv, hd)
            v = qkv[:, (self.n_heads + self.n_kv) * hd :].reshape(C, self.n_kv, hd)
            q = q * cos + self._rotate_half(q) * sin
            k = k * cos + self._rotate_half(k) * sin
            k_cache.index_copy_(0, pos, k.to(k_cache.dtype))
            v_cache.index_copy_(0, pos, v.to(v_cache.dtype))
            K = k_cache[:L].to(torch.float32).permute(1, 2, 0)  # [kv, hd, L]
            V = v_cache[:L].to(torch.float32).permute(1, 0, 2)  # [kv, L, hd]
            # GQA without broadcasting K/V: fold the query group into the row dim.
            qg = q.reshape(C, self.n_kv, g, hd).permute(1, 2, 0, 3).reshape(self.n_kv, g * C, hd)
            s = torch.bmm(qg, K) * scale  # [kv, g*C, L]
            s = s.reshape(self.n_kv, g, C, L)
            s = torch.where(mask[None, None], s, torch.full_like(s, -1e30))
            p = torch.softmax(s, -1).reshape(self.n_kv, g * C, L)
            o = torch.bmm(p, V).reshape(self.n_kv, g, C, hd).permute(2, 0, 1, 3).reshape(C, self.n_heads * hd)
            h = h + layer.wo(o)
            x = rms_norm(h, layer.ln_post, self.eps)
            gu = layer.wgu(x)
            h = h + layer.wdown(F.silu(gu[:, : self.inter]) * gu[:, self.inter :])
        x = rms_norm(h.index_select(0, last.to(torch.int64)), self.ln_f, self.eps)  # [1, D]
        logits = self.embed.logits(x)  # [1, vocab]
        return torch.argmax(logits, -1).to(torch.int32), logits

    def new_caches(self):
        return [torch.zeros(self.max_ctx, self.n_kv, self.head_dim, dtype=torch.float16)
                for _ in range(2 * self.n_layers)]


class StepFn(nn.Module):
    """Export wrapper: fixed bucket L, returns only the next-token id (or logits for debugging)."""

    def __init__(self, model, L, return_logits=False):
        super().__init__()
        self.model = model
        self.L = L
        self.return_logits = return_logits

    def forward(self, tokens, start, last, *caches):
        nxt, logits = self.model.step(tokens, start, last, caches, self.L)
        return logits if self.return_logits else nxt
