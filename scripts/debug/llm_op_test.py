"""GPU numerics tests for the LlamaQ4 building blocks (tiny dims). Uses op_test.py's harness.
  llm_op_test.py [--target adreno710] [case ...]
"""
import pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "llm"))
import op_test  # noqa: E402
from llama_q4 import QLinear, QEmbedding, rms_norm  # noqa: E402

torch.manual_seed(0)
Fn = op_test.Fn
D, G = 256, 32
ql = QLinear(torch.randn(512, D) * 0.05, G)
emb = QEmbedding(torch.randn(1000, D) * 0.05, G, 256)
cos, sin = torch.randn(64, 32), torch.randn(64, 32)


def rot(q):
    h = q.shape[-1] // 2
    return torch.cat([-q[..., h:], q[..., :h]], -1)


class KVAttn(torch.nn.Module):   # in-place cache write + bucketed GQA attention, like LlamaQ4.step
    def forward(self, q, k, pos, kc):
        kc.index_copy_(0, pos.to(torch.int64), k)
        K = kc[:32].to(torch.float32).permute(1, 2, 0)          # [2, 32, L]
        s = torch.bmm(q, K)                                      # [2, 4C, L]
        return torch.softmax(s, -1)


op_test.CASES.update({
    "q4_linear_m1": (Fn(lambda p, x: p["q"](x), {"q": ql}), (torch.randn(1, D),)),
    "q4_linear_m8": (Fn(lambda p, x: p["q"](x), {"q": ql}), (torch.randn(8, D),)),
    "q4_bitwise_only": (Fn(lambda p, x: ((p["q"].packed & 15).float() - (p["q"].packed >> 4).float()) * x,
                           {"q": ql}), (torch.randn(1, D),)),
    "q4_embed_lookup": (Fn(lambda p, t: p["e"].lookup(t.to(torch.int64)), {"e": emb}),
                        (torch.tensor([3, 300, 999, 512, 0, 767], dtype=torch.int32),)),
    "q4_lm_head": (Fn(lambda p, x: p["e"].logits(x), {"e": emb}), (torch.randn(1, D),)),
    "argmax": (Fn(lambda p, x: torch.argmax(x, -1).to(torch.int32)), (torch.randn(1, 1000),)),
    "rmsnorm": (Fn(lambda p, x, w: rms_norm(x, w, 1e-5)), (torch.randn(8, D) * 3, torch.randn(D))),
    "rope_gather": (Fn(lambda p, q, pos: q * cos.index_select(0, pos.to(torch.int64))[:, None]
                       + rot(q) * sin.index_select(0, pos.to(torch.int64))[:, None]),
                    (torch.randn(8, 4, 32), torch.arange(5, 13, dtype=torch.int32))),
    "kv_write_attn": (KVAttn(), (torch.randn(2, 8, 32), torch.randn(2, 2, 32).half(),
                                 torch.tensor([3, 4], dtype=torch.int32), torch.randn(64, 2, 32).half())),
})

if __name__ == "__main__":
    op_test.main()
