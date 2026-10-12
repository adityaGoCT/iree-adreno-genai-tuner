"""Bisect a GPU numerics bug by layer: export DistilBERT truncated to K transformer layers
(K=0 -> embeddings only), compile for a target, run on the phone, compare hidden states.

  bisect_layers.py --target adreno710 --layers 0 1 2 6 [-- extra iree-compile flags]
"""

import argparse
import pathlib
import subprocess
import sys

import numpy as np
import torch
from iree.turbine import aot
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from check import run_device  # noqa: E402
from export_hf import SAMPLES  # noqa: E402

MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


class Hidden(torch.nn.Module):
    def __init__(self, db):
        super().__init__()
        self.db = db

    def forward(self, input_ids, attention_mask):
        b, s = attention_mask.shape
        mask = ((1.0 - attention_mask[:, None, None, :].to(torch.float32)) * -1e4).expand(b, 1, s, s)
        return self.db(input_ids=input_ids, attention_mask=mask).last_hidden_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="adreno710")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 1, 6])
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("flags", nargs="*")
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = tok(SAMPLES, padding="max_length", truncation=True, max_length=a.seq_len, return_tensors="pt")
    args = (enc["input_ids"].to(torch.int32), enc["attention_mask"].to(torch.int32))

    for k in a.layers:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, attn_implementation="eager").eval()
        for mod in model.modules():
            for n, buf in list(mod._buffers.items()):
                if buf is not None and buf.dtype == torch.int64:
                    mod._buffers[n] = buf.to(torch.int32)
        model.distilbert.transformer.layer = model.distilbert.transformer.layer[:k]
        w = Hidden(model.distilbert).eval()
        with torch.no_grad():
            ref = w(*args).numpy()

        d = ROOT / "models" / f"bisect-k{k}"
        d.mkdir(parents=True, exist_ok=True)
        aot.export(w, args=args).save_mlir(str(d / "model.mlir"))
        files = []
        for name, t in zip(["input_ids", "attention_mask"], args):
            np.save(d / f"{name}.i32.npy", t.numpy())
            files.append(d / f"{name}.i32.npy")
        subprocess.run([str(ROOT / "scripts/compile.sh"), str(d), a.target, *a.flags],
                       check=True, capture_output=True)
        out = run_device(d, a.target, files).astype(np.float32)
        err = np.abs(out - ref)
        valid = args[1].numpy().astype(bool)  # ignore padded positions
        print(f"K={k}: max|Δ|={err.max():.3e} mean={err.mean():.3e} "
              f"max|Δ| (non-pad tokens)={err[valid].max():.3e}  |ref|max={np.abs(ref).max():.2f}", flush=True)


if __name__ == "__main__":
    main()
