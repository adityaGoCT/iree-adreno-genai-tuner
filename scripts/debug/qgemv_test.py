"""Fused int4 GEMV (decode form) vs dequant+linear: numerics + dispatch structure + GPU time.
  qgemv_test.py OUT_DIR [--n 5120 --k 3072]"""
import argparse, pathlib, sys, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "llm"))
import llama_q4
from iree.turbine import aot

ap = argparse.ArgumentParser(); ap.add_argument("out", type=pathlib.Path)
ap.add_argument("--n", type=int, default=5120); ap.add_argument("--k", type=int, default=3072)
ap.add_argument("--form", choices=["fused", "unfused"], default="fused")
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
ql = llama_q4.QLinear(torch.randn(a.n, a.k) * 0.02, 32).eval()
x = torch.randn(1, a.k)
ref = ql(x)                                   # eager: dequant + linear
llama_q4.EXPORT_FUSED_GEMV[0] = a.form == "fused"
with torch.no_grad():
    fused = ql(x)
print("eager fused vs unfused max|Δ|:", (fused - ref).abs().max().item())
class M(torch.nn.Module):
    def __init__(s): super().__init__(); s.q = ql
    def forward(s, x): return s.q(x)
aot.export(M(), args=(x,)).save_mlir(str(a.out / "model.mlir"))
import numpy as np; np.save(a.out / "in0.npy", x.numpy()); np.save(a.out / "ref.npy", ref.numpy())
