"""Adreno 8-bit integer op correctness with *runtime* uint8 inputs (all 256 values)."""
import pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import op_test  # noqa: E402

Fn = op_test.Fn
p = torch.arange(256, dtype=torch.uint8).repeat(4).reshape(8, 128)
f = lambda fn: (Fn(lambda _, x: fn(x)), (p,))  # noqa: E731
op_test.CASES.clear()
op_test.CASES.update({
    "u8_shr4": f(lambda x: (x >> 4).to(torch.float32)),
    "u8_and15": f(lambda x: (x & 15).to(torch.float32)),
    "u8_to_f32": f(lambda x: x.to(torch.float32)),
    "u8_to_i32": f(lambda x: x.to(torch.int32).to(torch.float32)),
    "u8_i32_shr4": f(lambda x: (x.to(torch.int32) >> 4).to(torch.float32)),
    "u8_i32_and15": f(lambda x: (x.to(torch.int32) & 15).to(torch.float32)),
    "u8_f32_floor": f(lambda x: torch.floor(x.to(torch.float32) * 0.0625)),
    # i8 -> i8 results stored as vector<4xi8> (the pattern hoisted into initializers):
    "u8_shr4_store_u8": f(lambda x: x >> 4),
    "u8_and15_store_u8": f(lambda x: x & 15),
    "u8_i32_shr4_store_u8": f(lambda x: (x.to(torch.int32) >> 4).to(torch.uint8)),
    "u8_i32_shr4_store_i32": f(lambda x: x.to(torch.int32) >> 4),
})
if __name__ == "__main__":
    op_test.main()
