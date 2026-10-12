"""Prototype: does a function that mutates an argument (KV-cache style) export + run in place?
Position dim must be dim 0: index_copy_ on dim>0 emits a None index that IREE 3.11 rejects."""
import sys, pathlib, torch
from iree.turbine import aot

class Step(torch.nn.Module):
    def forward(self, x, pos, k_cache, v_cache):   # x [1,4,8] f16, pos [1] i32, caches [16,4,8] f16
        k_cache.index_copy_(0, pos, x)
        v_cache.index_copy_(0, pos, x * 2)
        L = 8                                      # attend over a fixed bucket of the cache
        return (k_cache[:L].float() + v_cache[:L].float()).sum(dim=(1, 2))

out = pathlib.Path(sys.argv[1])
args = (torch.ones(1, 4, 8, dtype=torch.float16), torch.tensor([3], dtype=torch.int32),
        torch.zeros(16, 4, 8, dtype=torch.float16), torch.zeros(16, 4, 8, dtype=torch.float16))
aot.export(Step(), args=args).save_mlir(str(out / "model.mlir"))
