"""The 8 f16 GEMM problems the tuner is evaluated on: C[M,N] = op(A) @ op(B), fp16 in/out.

Layout letters follow BLAS: N = stored as written, T = stored transposed.
  A 'N': A[M][K] row-major      A 'T': stored A^T, i.e. [K][M]
  B 'N': B[K][N] row-major      B 'T': stored B^T, i.e. [N][K]  (a PyTorch Linear weight)
"""

import dataclasses
import pathlib

import numpy as np


@dataclasses.dataclass(frozen=True)
class Shape:
    id: str
    name: str
    M: int
    N: int
    K: int
    layout: str  # "NN" | "NT" | "TN" | "TT"
    note: str

    @property
    def la(self):
        return self.layout[0]

    @property
    def lb(self):
        return self.layout[1]

    @property
    def flops(self):
        return 2.0 * self.M * self.N * self.K

    @property
    def min_bytes(self):  # compulsory traffic: read A and B once, write C once (fp16)
        return 2.0 * (self.M * self.K + self.K * self.N + self.M * self.N)


SHAPES = [
    Shape("S1", "square_nn", 1024, 1024, 1024, "NN", "regular square"),
    Shape("S2", "prefill_nt", 128, 3072, 3072, "NT", "LLM prefill: 128 tokens x Linear(3072, 3072)"),
    Shape("S3", "skinny_tn", 4096, 64, 1024, "TN", "tall-skinny (M >> N)"),
    Shape("S4", "flat_nt", 16, 4096, 4096, "NT", "short-wide: batch-16 decode"),
    Shape("S5", "largek_tt", 256, 256, 8192, "TT", "large K"),
    Shape("S6", "smallk_nn", 2048, 2048, 32, "NN", "small K (rank-32 update, output-bound)"),
    Shape("S7", "gemv_nt", 1, 4096, 4096, "NT", "matrix-vector: batch-1 decode"),
    Shape("S8", "regular_tn", 768, 768, 768, "TN", "regular, A transposed"),
]
BY_ID = {s.id: s for s in SHAPES}

TOL = 5e-3  # max|C - Cref| / max|Cref|; fp16 output rounding alone is ~5e-4


def make_data(s: Shape, out: pathlib.Path, seed=0):
    """Write meta.txt, A.bin, B.bin (fp16, stored layout) and Cref.bin (fp32 reference)."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + int(s.id[1:]))
    a = rng.uniform(-1, 1, (s.M, s.K)).astype(np.float16)  # logical A [M][K]
    b = rng.uniform(-1, 1, (s.K, s.N)).astype(np.float16)  # logical B [K][N]
    c = a.astype(np.float32) @ b.astype(np.float32)
    (a if s.la == "N" else np.ascontiguousarray(a.T)).tofile(out / "A.bin")
    (b if s.lb == "N" else np.ascontiguousarray(b.T)).tofile(out / "B.bin")
    c.astype(np.float32).tofile(out / "Cref.bin")
    (out / "meta.txt").write_text(f"{s.M} {s.N} {s.K} {s.la} {s.lb} {TOL}\n")
