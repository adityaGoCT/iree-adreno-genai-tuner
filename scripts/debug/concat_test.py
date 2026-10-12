"""Minimal repro: matmul outputs concatenated along the last dim with M=1 (contiguous ->
IREE lowers the concat to buffer copies instead of a dispatch)."""
import pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import op_test  # noqa: E402

torch.manual_seed(0)
Fn = op_test.Fn
l1, l2 = torch.nn.Linear(256, 192, bias=False), torch.nn.Linear(256, 192, bias=False)
cat_slice = lambda p, x: torch.relu(torch.cat([p["a"](x), p["b"](x)], -1)[:, 320:384])  # noqa: E731
cat_full = lambda p, x: torch.relu(torch.cat([p["a"](x), p["b"](x)], -1))  # noqa: E731
op_test.CASES.clear()
op_test.CASES.update({
    "cat_m1_slice": (Fn(cat_slice, {"a": l1, "b": l2}), (torch.randn(1, 256),)),
    "cat_m8_slice": (Fn(cat_slice, {"a": l1, "b": l2}), (torch.randn(8, 256),)),
    "cat_m1_full": (Fn(cat_full, {"a": l1, "b": l2}), (torch.randn(1, 256),)),
})
if __name__ == "__main__":
    op_test.main()
