"""Export a Hugging Face sequence-classification model to MLIR (torch dialect).

Writes into <out_dir>:
  model.mlir                 torch-dialect MLIR with inlined weights (iree-turbine / FX importer)
  input_ids.npy, attention_mask.npy   fixed-shape example inputs (int32)
  ref_logits.npy             PyTorch fp32 reference output for those inputs (always fp32,
                             also for --dtype f16 exports)
  meta.json                  model id, seq len, sample texts, labels
"""

import argparse
import json
import pathlib

import numpy as np
import torch
from iree.turbine import aot
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SAMPLES = [
    "This phone's GPU is surprisingly fast for its price.",
    "The battery died after an hour and support never replied.",
]


class LogitsOnly(torch.nn.Module):
    def __init__(self, model, dtype=torch.float32):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, input_ids, attention_mask):
        # Pass a ready-made 4D additive mask [B,1,1,S] so HF skips its vmap-based mask
        # builder (masking_utils), whose 0-d bool `new_ones` IREE 3.11's torch lowering
        # rejects. -1e4 instead of finfo.min keeps the mask finite after f32->f16 demotion.
        # Expanded to [B,1,S,S]: torch-mlir's sdpa->tm_tensor.attention lowering does not
        # broadcast the mask over the query dim.
        b, s = attention_mask.shape
        mask = (1.0 - attention_mask[:, None, None, :].to(self.dtype)) * -1e4
        mask = mask.expand(b, 1, s, s)
        return self.model(input_ids=input_ids, attention_mask=mask).logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--out", type=pathlib.Path, required=True)
    # eager: plain matmul/softmax/matmul. sdpa produces iree_linalg_ext.attention, which the
    # IREE 3.11 SPIR-V backend cannot compile yet.
    p.add_argument("--attn", default="eager", choices=["sdpa", "eager"])
    # f16: weights/activations exported as fp16 (Adreno's fast path). iree's
    # --iree-input-demote-f32-to-f16 can't retype dense_resource weights, so cast here.
    p.add_argument("--dtype", default="f32", choices=["f32", "f16"])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, dtype=torch.float32, attn_implementation=a.attn
    ).eval()

    # Adreno 710 has no shaderInt64, and --iree-input-demote-i64-to-i32 cannot rewrite int64
    # weight constants (dense_resource blobs, e.g. position_ids). So make every int64 buffer
    # and every input int32 *before* export; remaining computed i64 values get demoted.
    for mod in model.modules():
        for k, buf in list(mod._buffers.items()):
            if buf is not None and buf.dtype == torch.int64:
                mod._buffers[k] = buf.to(torch.int32)

    enc = tok(SAMPLES, padding="max_length", truncation=True, max_length=a.seq_len, return_tensors="pt")
    args = (enc["input_ids"].to(torch.int32), enc["attention_mask"].to(torch.int32))
    wrapped = LogitsOnly(model).eval()

    with torch.no_grad():
        ref = wrapped(*args)
    for text, row in zip(SAMPLES, ref.softmax(-1).tolist()):
        label = model.config.id2label[int(np.argmax(row))]
        print(f"torch ref: {label:8s} {max(row):.4f}  {text!r}")

    if a.dtype == "f16":
        wrapped = LogitsOnly(model.half(), torch.float16).eval()
    exported = aot.export(wrapped, args=args)
    exported.save_mlir(str(a.out / "model.mlir"))

    np.save(a.out / "input_ids.npy", args[0].numpy())
    np.save(a.out / "attention_mask.npy", args[1].numpy())
    np.save(a.out / "ref_logits.npy", ref.numpy())
    (a.out / "meta.json").write_text(json.dumps({
        "model": a.model, "seq_len": a.seq_len, "batch": len(SAMPLES), "attn": a.attn, "dtype": a.dtype,
        "samples": SAMPLES, "id2label": model.config.id2label,
        "torch": torch.__version__,
    }, indent=2))
    print(f"wrote {a.out}/model.mlir ({(a.out / 'model.mlir').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
