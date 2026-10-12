"""fp16 op tests (runtime inputs) to isolate the Turnip VK_ERROR_DEVICE_LOST seen with fp16 DistilBERT."""
import pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import op_test  # noqa: E402

torch.manual_seed(0)
Fn = op_test.Fn
h = torch.float16
lin = torch.nn.Linear(768, 768).half()
ln = torch.nn.LayerNorm(768).half()
op_test.CASES.clear()
op_test.CASES.update({
    "f16_add": (Fn(lambda p, x, y: x + y), (torch.randn(256, 768, dtype=h), torch.randn(256, 768, dtype=h))),
    "f16_linear_m1": (Fn(lambda p, x: p["l"](x), {"l": lin}), (torch.randn(1, 768, dtype=h),)),
    "f16_linear_m256": (Fn(lambda p, x: p["l"](x), {"l": lin}), (torch.randn(256, 768, dtype=h),)),
    "f16_reduce_sum": (Fn(lambda p, x: x.sum(-1)), (torch.randn(256, 768, dtype=h) * 0.1,)),
    "f16_softmax": (Fn(lambda p, x: torch.softmax(x, -1)), (torch.randn(24, 128, 128, dtype=h),)),
    "f16_layernorm": (Fn(lambda p, x: p["n"](x), {"n": ln}), (torch.randn(256, 768, dtype=h),)),
    "f16_gelu": (Fn(lambda p, x: torch.nn.functional.gelu(x)), (torch.randn(256, 768, dtype=h),)),
})
if __name__ == "__main__":
    op_test.main()
