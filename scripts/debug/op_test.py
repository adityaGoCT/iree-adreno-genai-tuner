"""Per-op GPU numerics check: export tiny torch modules, compile for a target, run on phone,
compare to torch. Seed of a kernel regression suite.

  op_test.py [--target adreno710] [case ...]     (no cases -> run all)
"""

import argparse
import pathlib
import subprocess
import sys

import numpy as np
import torch
from iree.turbine import aot

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from check import run_device  # noqa: E402

torch.manual_seed(0)


class Fn(torch.nn.Module):
    def __init__(self, fn, params=None):
        super().__init__()
        self.fn = fn
        self.p = torch.nn.ModuleDict(params or {})

    def forward(self, *xs):
        return self.fn(self.p, *xs)


def lin(i, o):
    return torch.nn.Linear(i, o)


CASES = {
    # name: (module, example inputs)
    "linear_2x768x2": (Fn(lambda p, x: p["l"](x), {"l": lin(768, 2)}), (torch.randn(2, 768),)),
    "linear_2x768x768": (Fn(lambda p, x: p["l"](x), {"l": lin(768, 768)}), (torch.randn(2, 768),)),
    "linear_2x768x768_relu": (Fn(lambda p, x: torch.relu(p["l"](x)), {"l": lin(768, 768)}), (torch.randn(2, 768),)),
    "select_cls": (Fn(lambda p, h: h[:, 0] * 1.0), (torch.randn(2, 128, 768),)),
    "head": (Fn(lambda p, h: p["c"](torch.relu(p["pc"](h[:, 0]))), {"pc": lin(768, 768), "c": lin(768, 2)}),
             (torch.randn(2, 128, 768),)),
    "linear_256x768x768": (Fn(lambda p, x: p["l"](x), {"l": lin(768, 768)}), (torch.randn(256, 768),)),
    "linear_8x768x2": (Fn(lambda p, x: p["l"](x), {"l": lin(768, 2)}), (torch.randn(8, 768),)),
    "linear_2x768x4": (Fn(lambda p, x: p["l"](x), {"l": lin(768, 4)}), (torch.randn(2, 768),)),
    "linear_2x64x2": (Fn(lambda p, x: p["l"](x), {"l": lin(64, 2)}), (torch.randn(2, 64),)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="adreno710")
    ap.add_argument("cases", nargs="*")
    a = ap.parse_args()
    for name in a.cases or CASES:
        mod, xs = CASES[name]
        mod.eval()
        with torch.no_grad():
            ref = mod(*xs).numpy()
        d = ROOT / "models" / "optest" / name
        d.mkdir(parents=True, exist_ok=True)
        aot.export(mod, args=xs).save_mlir(str(d / "model.mlir"))
        files = []
        for i, x in enumerate(xs):
            np.save(d / f"in{i}.npy", x.numpy())
            files.append(d / f"in{i}.npy")
        r = subprocess.run([str(ROOT / "scripts/compile.sh"), str(d), a.target], capture_output=True, text=True)
        if r.returncode:
            print(f"{name:28s} COMPILE FAIL: {[l for l in r.stderr.splitlines() if 'error' in l][:1]}")
            continue
        out = run_device(d, a.target, files).astype(np.float32)
        err = np.abs(out - ref).max()
        tol = 1e-3 * max(1.0, np.abs(ref).max())
        print(f"{name:28s} {'PASS' if err <= tol else 'FAIL'}  max|Δ|={err:.3e}  |ref|max={np.abs(ref).max():.2f}",
              flush=True)


if __name__ == "__main__":
    main()
