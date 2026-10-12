"""Schedule space for fp16 GEMM on Adreno (Vulkan compute) and its lowering to GLSL.

A Schedule is a point in a TVM/Ansor-style space. It is applied to the loop nest

    for m, n:  for k:  C[m, n] += op(A)[m, k] * op(B)[k, n]

with these primitives (each one is a knob below):

  multi-level tiling   split m -> (block BM, thread TM), n -> (block BN, thread TN), k -> (tile BK)
  bind / vthread       block tile -> workgroup; thread tile -> invocations; each thread's TM x TN tile
                       is split into VTM x VTN "virtual thread" sub-tiles placed BM/VTM (BN/VTN) apart
                       (strided ownership); BIND picks which output axis the fastest lanes walk;
                       SG picks the hardware wave size (64 | 128)
  rfactor + cross-thread reduction
                       k is further split across KT lanes (KMAP: contiguous "block" or interleaved
                       "cyclic" slices); partials are combined with a subgroup clustered add or
                       through shared memory (RED)
  cache_read (shared)  SMEM_A / SMEM_B stage the operand tiles in workgroup memory (+ PAD to dodge
                       bank conflicts); otherwise threads read global memory directly (L1/UCHE).
                       Shared tiles use vector element types so a thread reads its rows/cols as f16vec4
  double_buffer        DB prefetches the next k-tile into registers while the current one is computed
  vectorize            VEC = width of global loads (along the operand's contiguous axis) and stores
  unroll               UNROLL copies of the k-chunk body
  compute_inline       a transposed operand (layout 'T') is either inlined into the GEMM (indices
                       swapped at load time, INLINE=1) or materialized by a separate transpose
                       kernel first (INLINE=0)
  accumulate dtype     ACC = f32 | f16 (fp16 math, ~2.5x peak; precision risk -> verifier) |
                       f16p (promoted: fp16 within each k-tile, added into fp32 after every tile, so
                       no fp16 chain is longer than BK/KT -- the FP8-GEMM "promotion" trick)

All index algebra lives in the small `rows/cols/kslice/loader` helpers, shared by the code
generator, the verifier's coverage check and the cost model, so they cannot disagree.
"""

import dataclasses
import itertools
from typing import List, Tuple

from shapes import Shape


@dataclasses.dataclass(frozen=True)
class Schedule:
    BM: int
    BN: int
    BK: int
    TM: int
    TN: int
    VTM: int = 1
    VTN: int = 1
    KT: int = 1
    RED: str = "subgroup"  # subgroup | smem
    KMAP: str = "block"  # block | cyclic
    SMEM_A: int = 0
    SMEM_B: int = 0
    DB: int = 0  # double-buffer shared tiles: prefetch the next k-tile into registers during compute
    VEC: int = 1
    PAD: int = 0
    UNROLL: int = 1
    ACC: str = "f32"  # f32 | f16 | f16p (fp16 per k-tile, promoted to fp32 accumulators)
    INLINE_A: int = 1
    INLINE_B: int = 1
    BIND: str = "n"  # n | m
    SG: int = 64

    # ---- derived ----
    @property
    def WX(self):
        return self.BN // self.TN

    @property
    def WY(self):
        return self.BM // self.TM

    @property
    def WGS(self):
        return self.WX * self.WY * self.KT

    @property
    def SM(self):
        return self.TM // self.VTM

    @property
    def SN(self):
        return self.TN // self.VTN

    @property
    def KS(self):  # k values per lane per BK tile
        return self.BK // self.KT

    def key(self):
        return "_".join(f"{f.name}{getattr(self, f.name)}" for f in dataclasses.fields(self))

    def short(self):
        s = (f"wg{self.BM}x{self.BN}x{self.BK} t{self.TM}x{self.TN}"
             + (f" vt{self.VTM}x{self.VTN}" if self.VTM * self.VTN > 1 else "")
             + (f" kt{self.KT}{self.RED[0]}{self.KMAP[0]}" if self.KT > 1 else "")
             + (" smA" if self.SMEM_A else "") + (" smB" if self.SMEM_B else "")
             + (" db" if self.DB else "")
             + f" v{self.VEC}" + (f" u{self.UNROLL}" if self.UNROLL > 1 else "")
             + f" {self.ACC} b{self.BIND} sg{self.SG}")
        return s


def eff_layouts(s: Shape, sc: Schedule):
    """Layouts the GEMM kernel reads (after optional materialized transposes) and their bindings."""
    la = "N" if (s.la == "T" and not sc.INLINE_A) else s.la
    lb = "N" if (s.lb == "T" and not sc.INLINE_B) else s.lb
    ba = 3 if (s.la == "T" and not sc.INLINE_A) else 0
    bb = 4 if (s.lb == "T" and not sc.INLINE_B) else 1
    return la, lb, ba, bb


def grid(s: Shape, sc: Schedule):
    return (-(-s.N // sc.BN), -(-s.M // sc.BM), 1)


# ---- index algebra (shared with verifier and cost model) -------------------------------------
def rows(sc: Schedule, ty: int) -> List[int]:
    """Local rows owned by thread row ty: VTM sub-tiles of SM rows, BM/VTM apart."""
    return [v * (sc.BM // sc.VTM) + ty * sc.SM + r for v in range(sc.VTM) for r in range(sc.SM)]


def cols(sc: Schedule, tx: int) -> List[int]:
    return [v * (sc.BN // sc.VTN) + tx * sc.SN + c for v in range(sc.VTN) for c in range(sc.SN)]


def kslice(sc: Schedule, kt: int) -> List[int]:
    """Local k indices (within a BK tile) reduced by rfactor lane kt."""
    if sc.KMAP == "block":
        return [kt * sc.KS + i for i in range(sc.KS)]
    return [i * sc.KT + kt for i in range(sc.KS)]


def thread_xy(sc: Schedule, tid: int) -> Tuple[int, int, int]:
    kt, t = tid % sc.KT, tid // sc.KT
    if sc.BIND == "n":
        return t % sc.WX, t // sc.WX, kt
    return t // sc.WY, t % sc.WY, kt


def vec_widths(s: Shape, sc: Schedule):
    """Vector widths actually used by the generated code for direct (non-shared) loads.
    KV: along k (A 'N' / B 'T' operands, needs contiguous 'block' k slices);
    VM: along m for A 'T'; VN: along n for B 'N'; VS: C stores along n."""
    la, lb, _, _ = eff_layouts(s, sc)

    def fit(w, *conds_div):
        while w > 1 and any(d % w for d in conds_div):
            w //= 2
        return w

    k_dir = (not sc.SMEM_A and la == "N") or (not sc.SMEM_B and lb == "T")
    KV = fit(sc.VEC, sc.KS, s.K) if (k_dir and sc.KMAP == "block") else 1
    VM = fit(sc.VEC, sc.SM, sc.BM // sc.VTM, s.M) if (not sc.SMEM_A and la == "T") else 1
    VN = fit(sc.VEC, sc.SN, sc.BN // sc.VTN, s.N) if (not sc.SMEM_B and lb == "N") else 1
    VS = fit(min(sc.VEC, 4) if sc.VEC > 1 else 1, sc.SN, s.N)
    return KV, VM, VN, VS


def loader_plan(s: Shape, sc: Schedule, op: str):
    """Cooperative loader of a shared-memory staged operand.

    The shared tile is [BK][BX + PAD] (BX = BM for A, BN for B) with vector elements of width sv
    along x. The tile is cut into loader *units*, one per thread per iteration, shaped so that every
    shared write is a whole vector (threads writing different lanes of one shared vector race):
      axis 'x' (global contiguous along x: A 'T', B 'N'): ux = max(vw, sv) consecutive x at one k
               -> ux/vw global vector loads, ux/sv shared vector stores
      axis 'k' (global contiguous along k: A 'N', B 'T'): sv consecutive x by vw consecutive k
               -> sv global vector loads, transposed in registers into vw shared vector stores
    Units are numbered with the global contiguous axis fastest so adjacent lanes coalesce."""
    la, lb, _, _ = eff_layouts(s, sc)
    contig_x = (la == "T") if op == "A" else (lb == "N")
    vw, sv = sc.VEC, smem_vec(sc, op)
    bx = sc.BM if op == "A" else sc.BN
    ux, uk = (max(vw, sv), 1) if contig_x else (sv, vw)
    return dict(axis="x" if contig_x else "k", vw=vw, sv=sv, ux=ux, uk=uk, bx=bx,
                n=(bx // ux) * (sc.BK // uk) if bx % ux == 0 and sc.BK % uk == 0 else 0)


def loader_unit(p, u):
    """Tile origin (k, x) of loader unit u -- the generated code uses the same algebra."""
    if p["axis"] == "x":
        per = p["bx"] // p["ux"]
        return u // per * p["uk"], u % per * p["ux"]
    per = p["n"] // (p["bx"] // p["ux"])  # units along k
    return u % per * p["uk"], u // per * p["ux"]


# ---- GLSL generation ---------------------------------------------------------------------------
def _vload(name, buf, idx, w, guard=None):
    """Emit a (guarded) vector load of w halfs at element index idx (aligned to w).
    Returns (code, component expressions). Out-of-bounds vectors (guard false) read as zero."""
    def one(v, sub, width):
        ld = f"{buf}[{sub}]" if width == 1 else f"{buf}_v{width}[({sub}) >> {width.bit_length() - 1}]"
        ty = "float16_t" if width == 1 else f"f16vec{width}"
        rhs = f"(({guard}) ? {ld} : {ty}(0))" if guard else ld
        return f"{ty} {v} = {rhs};"
    if w == 1:
        return one(name, idx, 1), [name]
    if w in (2, 4):
        return one(name, idx, w), [f"{name}.{c}" for c in "xyzw"[:w]]
    code = one(f"{name}a", idx, 4) + " " + one(f"{name}b", f"({idx}) + 4u", 4)
    return code, [f"{name}a.{c}" for c in "xyzw"] + [f"{name}b.{c}" for c in "xyzw"]


def _zero_vec(w):
    return "float16_t(0)" if w == 1 else f"f16vec{min(w, 4)}(0)"


def smem_vec(sc: Schedule, op: str) -> int:
    """Element vector width of a shared tile: a thread reads SVA consecutive rows (SVB cols)."""
    sub, stride, rowlen = ((sc.SM, sc.BM // sc.VTM, sc.BM + sc.PAD) if op == "A"
                           else (sc.SN, sc.BN // sc.VTN, sc.BN + sc.PAD))
    for w in (4, 2):
        if sub % w == 0 and stride % w == 0 and rowlen % w == 0:
            return w
    return 1


def gemm_glsl(s: Shape, sc: Schedule) -> str:
    la, lb, ba, bb = eff_layouts(s, sc)
    M, N, K = s.M, s.N, s.K
    KV, VM, VN, VS = vec_widths(s, sc)
    acc_t = "float16_t" if sc.ACC == "f16" else "float"   # f32 and f16p keep fp32 accumulators
    cvt = (lambda e: f"float({e})") if sc.ACC == "f32" else (lambda e: e)
    tgt = "h" if sc.ACC == "f16p" else "c"  # FMA target: fp16 partials for f16p
    gm_guard, gn_guard = M % sc.BM != 0, N % sc.BN != 0
    AS, BS = sc.BM + sc.PAD, sc.BN + sc.PAD
    SVA, SVB = smem_vec(sc, "A"), smem_vec(sc, "B")
    vt = lambda w: "float16_t" if w == 1 else f"f16vec{w}"  # noqa: E731
    sh = lambda w: w.bit_length() - 1  # noqa: E731
    L = []
    w = L.append
    w("#version 450")
    w("#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require")
    w("#extension GL_EXT_shader_16bit_storage : require")
    w("#extension GL_EXT_control_flow_attributes : enable")
    if sc.KT > 1 and sc.RED == "subgroup":
        w("#extension GL_KHR_shader_subgroup_clustered : require")
        if sc.ACC == "f16":
            w("#extension GL_EXT_shader_subgroup_extended_types_float16 : require")
    # per-dimension limit is 1024 (the device allows 2048 invocations in total): fold into y
    w(f"layout(local_size_x = {min(sc.WGS, 1024)}, local_size_y = {max(1, sc.WGS // 1024)}) in;")
    for buf, b in (("A", ba), ("B", bb)):
        w(f"layout(std430, binding = {b}) readonly buffer {buf}1 {{ float16_t {buf}[]; }};")
        w(f"layout(std430, binding = {b}) readonly buffer {buf}2 {{ f16vec2 {buf}_v2[]; }};")
        w(f"layout(std430, binding = {b}) readonly buffer {buf}4 {{ f16vec4 {buf}_v4[]; }};")
    w("layout(std430, binding = 2) writeonly buffer C1 { float16_t C[]; };")
    w("layout(std430, binding = 2) writeonly buffer C2 { f16vec2 C_v2[]; };")
    w("layout(std430, binding = 2) writeonly buffer C4 { f16vec4 C_v4[]; };")
    w(f"const uint M = {M}u, N = {N}u, K = {K}u;")
    if sc.SMEM_A:
        w(f"shared {vt(SVA)} As[{sc.BK * AS // SVA}];")
    if sc.SMEM_B:
        w(f"shared {vt(SVB)} Bs[{sc.BK * BS // SVB}];")
    if sc.KT > 1 and sc.RED == "smem":
        w(f"shared {acc_t} red[{sc.BM * sc.BN * sc.KT}];")

    def sm_vec(arr, sv, e):  # aligned vector of sv elements
        return f"{arr}[{e}]" if sv == 1 else f"{arr}[({e}) >> {sh(sv)}]"

    w("void main() {")
    w("  uint tid = gl_LocalInvocationIndex;")
    w(f"  uint kt = tid % {sc.KT}u; uint t = tid / {sc.KT}u;")
    if sc.BIND == "n":
        w(f"  uint tx = t % {sc.WX}u; uint ty = t / {sc.WX}u;")
    else:
        w(f"  uint ty = t % {sc.WY}u; uint tx = t / {sc.WY}u;")
    w(f"  uint m0 = gl_WorkGroupID.y * {sc.BM}u; uint n0 = gl_WorkGroupID.x * {sc.BN}u;")
    rbase = [v * (sc.BM // sc.VTM) for v in range(sc.VTM)]
    cbase = [v * (sc.BN // sc.VTN) for v in range(sc.VTN)]
    row = lambda i: f"({rbase[i // sc.SM] + i % sc.SM}u + ty * {sc.SM}u)"  # noqa: E731
    col = lambda j: f"({cbase[j // sc.SN] + j % sc.SN}u + tx * {sc.SN}u)"  # noqa: E731

    # ---- hoisted per-row / per-column offsets and guards (loop invariant) ----
    if not sc.SMEM_A:
        for i in range(sc.TM):
            if la == "N":
                w(f"  uint ao{i} = (m0 + {row(i)}) * K;")
            elif i % VM == 0:
                w(f"  uint ao{i} = m0 + {row(i)};")
            if gm_guard and (la == "N" or i % VM == 0):
                w(f"  bool aok{i} = m0 + {row(i)} < M;")
    if not sc.SMEM_B:
        for j in range(sc.TN):
            if lb == "T":
                w(f"  uint bo{j} = (n0 + {col(j)}) * K;")
            elif j % VN == 0:
                w(f"  uint bo{j} = n0 + {col(j)};")
            if gn_guard and (lb == "T" or j % VN == 0):
                w(f"  bool bok{j} = n0 + {col(j)} < N;")
    for i in range(sc.TM):
        for j in range(sc.TN):
            w(f"  {acc_t} c_{i}_{j} = {acc_t}(0);")

    # ---- cooperative tile loader, split into register load + shared store (for double buffering) ----
    # Strings only (no closures): load lines carry a @KX@ placeholder for the k-tile base.
    load_lines, store_lines = [], []
    for op, on in (("A", sc.SMEM_A), ("B", sc.SMEM_B)):
        if not on:
            continue
        p = loader_plan(s, sc, op)
        vw, sv, ux, uk, n = p["vw"], p["sv"], p["ux"], p["uk"], p["n"]
        arr, buf, rl = ("As", "A", AS) if op == "A" else ("Bs", "B", BS)
        x0, X, xg = ("m0", "M", gm_guard) if op == "A" else ("n0", "N", gn_guard)
        for it in range(-(-n // sc.WGS)):
            ue = f"(tid + {it * sc.WGS}u)"
            inb = f"{ue} < {n}u" if n % sc.WGS else None
            if p["axis"] == "x":
                per = p["bx"] // ux
                kk, xx = f"({ue} / {per}u)", f"(({ue} % {per}u) * {ux}u)"
                # runs of contiguous global elements: (global index, guard, [(k_off, x_off) per element])
                runs = [(f"(@KX@ + {kk}) * {X} + {x0} + {xx} + {q}u", f"{x0} + {xx} + {q}u < {X}" if xg else None,
                         [(0, q + e) for e in range(vw)]) for q in range(0, ux, vw)]
                vecs = [(0, q) for q in range(0, ux, sv)]
            else:
                per = sc.BK // uk
                kk, xx = f"(({ue} % {per}u) * {uk}u)", f"(({ue} / {per}u) * {ux}u)"
                runs = [(f"({x0} + {xx} + {r}u) * K + @KX@ + {kk}", f"{x0} + {xx} + {r}u < {X}" if xg else None,
                         [(e, r) for e in range(vw)]) for r in range(ux)]
                vecs = [(u, 0) for u in range(uk)]
            comp = {}
            for ri, (gidx, g, elems) in enumerate(runs):
                guard = " && ".join(x for x in (inb, g) if x)
                for e0 in range(0, vw, 4):
                    c = min(vw, 4)
                    reg = f"p{op}{it}_{ri}_{e0}"
                    w(f"  {vt(c)} {reg} = {vt(c)}(0);")
                    ld = f"{buf}[{gidx}]" if c == 1 else f"{buf}_v{c}[({gidx} + {e0}u) >> {sh(c)}]"
                    load_lines.append(f"{reg} = ({guard}) ? {ld} : {vt(c)}(0);" if guard else f"{reg} = {ld};")
                    for q in range(c):
                        comp[elems[e0 + q]] = reg if c == 1 else f"{reg}.{'xyzw'[q]}"
            out = []
            for ko, xo in vecs:
                dst = f"({kk} + {ko}u) * {rl}u + {xx} + {xo}u"
                vals = [comp[(ko, xo + q)] for q in range(sv)]
                out.append(f"{sm_vec(arr, sv, dst)} = {vals[0] if sv == 1 else vt(sv) + '(' + ', '.join(vals) + ')'};")
            store_lines += [f"if ({inb}) {{ " + " ".join(out) + " }"] if inb else out

    def emit_load(kx, ind):
        for line in load_lines:
            w(ind + line.replace("@KX@", kx))

    def emit_store(ind):
        for line in store_lines:
            w(ind + line)

    loaders = load_lines

    staged = bool(loaders)
    if staged and sc.DB:  # prologue: first tile
        emit_load("0u", "  ")
        emit_store("  ")
        w("  barrier();")
    w(f"  for (uint k0 = 0u; k0 < K; k0 += {sc.BK}u) {{")
    if staged and sc.DB:
        w(f"    bool more = k0 + {sc.BK}u < K;")
        w("    if (more) {")
        emit_load(f"(k0 + {sc.BK}u)", "      ")
        w("    }")
    elif staged:
        emit_load("k0", "    ")
        emit_store("    ")
        w("    barrier();")

    # ---- per-lane k loop: chunks of KV k-values, UNROLL chunks per iteration ----
    nkc = sc.KS // KV
    if sc.ACC == "f16p":
        for i in range(sc.TM):
            w("    " + " ".join(f"float16_t h_{i}_{j} = float16_t(0);" for j in range(sc.TN)))
    w(f"    for (uint kc = 0u; kc < {nkc}u; kc += {sc.UNROLL}u) {{")
    for uu in range(sc.UNROLL):
        w("    {")
        kcexp = f"(kc + {uu}u)"
        if sc.KMAP == "block":
            kl = lambda u: f"(kt * {sc.KS}u + {kcexp} * {KV}u + {u}u)"  # noqa: E731
        else:
            kl = lambda u: f"({kcexp} * {sc.KT}u + kt)"  # noqa: E731
        a = [[None] * KV for _ in range(sc.TM)]
        b = [[None] * KV for _ in range(sc.TN)]
        if sc.SMEM_A:
            for u in range(KV):
                w(f"      uint ka{u} = {kl(u)} * {AS}u;")
                for i0 in range(0, sc.TM, SVA):
                    v = f"sa{i0}_{u}"
                    w(f"      {vt(SVA)} {v} = {sm_vec('As', SVA, f'ka{u} + {row(i0)}')};")
                    for q in range(SVA):
                        a[i0 + q][u] = v if SVA == 1 else f"{v}.{'xyzw'[q]}"
        elif la == "N":
            for i in range(sc.TM):
                g = f"aok{i}" if gm_guard else None
                if KV > 1:
                    code, comps = _vload(f"a{i}", "A", f"ao{i} + k0 + {kl(0)}", KV, g)
                    w("      " + code)
                    a[i] = comps
                else:
                    e = f"A[ao{i} + k0 + {kl(0)}]"
                    a[i][0] = f"({g} ? {e} : float16_t(0))" if g else e
        else:
            for u in range(KV):
                w(f"      uint kma{u} = (k0 + {kl(u)}) * M;")
                for i0 in range(0, sc.TM, VM):
                    code, comps = _vload(f"a{i0}_{u}", "A", f"kma{u} + ao{i0}", VM, f"aok{i0}" if gm_guard else None)
                    w("      " + code)
                    for q in range(VM):
                        a[i0 + q][u] = comps[q]
        if sc.SMEM_B:
            for u in range(KV):
                w(f"      uint kb{u} = {kl(u)} * {BS}u;")
                for j0 in range(0, sc.TN, SVB):
                    v = f"sb{j0}_{u}"
                    w(f"      {vt(SVB)} {v} = {sm_vec('Bs', SVB, f'kb{u} + {col(j0)}')};")
                    for q in range(SVB):
                        b[j0 + q][u] = v if SVB == 1 else f"{v}.{'xyzw'[q]}"
        elif lb == "T":
            for j in range(sc.TN):
                g = f"bok{j}" if gn_guard else None
                if KV > 1:
                    code, comps = _vload(f"b{j}", "B", f"bo{j} + k0 + {kl(0)}", KV, g)
                    w("      " + code)
                    b[j] = comps
                else:
                    e = f"B[bo{j} + k0 + {kl(0)}]"
                    b[j][0] = f"({g} ? {e} : float16_t(0))" if g else e
        else:
            for u in range(KV):
                w(f"      uint knb{u} = (k0 + {kl(u)}) * N;")
                for j0 in range(0, sc.TN, VN):
                    code, comps = _vload(f"b{j0}_{u}", "B", f"knb{u} + bo{j0}", VN, f"bok{j0}" if gn_guard else None)
                    w("      " + code)
                    for q in range(VN):
                        b[j0 + q][u] = comps[q]
        for u in range(KV):
            for i in range(sc.TM):
                for j in range(sc.TN):
                    w(f"      {tgt}_{i}_{j} = fma({cvt(a[i][u])}, {cvt(b[j][u])}, {tgt}_{i}_{j});")
        w("    }")
    w("    }")
    if sc.ACC == "f16p":  # promote this k-tile's fp16 partials
        for i in range(sc.TM):
            w("    " + " ".join(f"c_{i}_{j} += float(h_{i}_{j});" for j in range(sc.TN)))
    if staged and sc.DB:
        w("    barrier();")
        w("    if (more) {")
        emit_store("      ")
        w("    }")
        w("    barrier();")
    elif staged:
        w("    barrier();")
    w("  }")
    # ---- cross-thread reduction over the KT rfactor lanes ----
    if sc.KT > 1:
        if sc.RED == "subgroup":
            for i in range(sc.TM):
                for j in range(sc.TN):
                    w(f"  c_{i}_{j} = subgroupClusteredAdd(c_{i}_{j}, {sc.KT}u);")
        else:
            for i in range(sc.TM):
                for j in range(sc.TN):
                    w(f"  red[({row(i)} * {sc.BN}u + {col(j)}) * {sc.KT}u + kt] = c_{i}_{j};")
            w("  barrier();")
            w("  if (kt == 0u) {")
            for i in range(sc.TM):
                for j in range(sc.TN):
                    base = f"({row(i)} * {sc.BN}u + {col(j)}) * {sc.KT}u"
                    w(f"    c_{i}_{j} = " + " + ".join(f"red[{base} + {q}u]" for q in range(sc.KT)) + ";")
            w("  }")
        w("  if (kt != 0u) return;")
    # ---- store C (vectorized along n) ----
    for i in range(sc.TM):
        w(f"  {{ uint cm = m0 + {row(i)};")
        for j0 in range(0, sc.TN, VS):
            gn = f"(n0 + {col(j0)})"
            conds = (["cm < M"] if gm_guard else []) + ([f"{gn} < N"] if gn_guard else [])
            pre = f"if ({' && '.join(conds)}) " if conds else ""
            vals = [f"float16_t(c_{i}_{j0 + q})" for q in range(VS)]
            if VS == 1:
                w(f"    {pre}C[cm * N + {gn}] = {vals[0]};")
            else:
                w(f"    {pre}C_v{VS}[(cm * N + {gn}) >> {sh(VS)}] = f16vec{VS}({', '.join(vals)});")
        w("  }")
    w("}")
    return "\n".join(L) + "\n"


def transpose_glsl(rows_: int, cols_: int, src: int, dst: int) -> str:
    """Materialized transpose (compute_inline=0): src stored [rows][cols] -> dst [cols][rows]."""
    return f"""#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = {src}) readonly buffer S {{ float16_t src[]; }};
layout(std430, binding = {dst}) writeonly buffer D {{ float16_t dst[]; }};
shared float16_t tile[16][17];
void main() {{
  uint lx = gl_LocalInvocationID.x, ly = gl_LocalInvocationID.y;
  uint x = gl_WorkGroupID.x * 16u + lx, y = gl_WorkGroupID.y * 16u + ly;
  if (x < {cols_}u && y < {rows_}u) tile[ly][lx] = src[y * {cols_}u + x];
  barrier();
  uint ox = gl_WorkGroupID.y * 16u + lx, oy = gl_WorkGroupID.x * 16u + ly;
  if (ox < {rows_}u && oy < {cols_}u) dst[oy * {rows_}u + ox] = tile[lx][ly];
}}
"""


def kernels(s: Shape, sc: Schedule):
    """[(name, glsl, grid)] dispatches in order: optional materialized transposes, then the GEMM."""
    out = []
    if s.la == "T" and not sc.INLINE_A:  # A^T stored [K][M] -> S0 [M][K]
        out.append(("transA", transpose_glsl(s.K, s.M, 0, 3), (-(-s.M // 16), -(-s.K // 16), 1)))
    if s.lb == "T" and not sc.INLINE_B:  # B^T stored [N][K] -> S1 [K][N]
        out.append(("transB", transpose_glsl(s.N, s.K, 1, 4), (-(-s.K // 16), -(-s.N // 16), 1)))
    out.append(("gemm", gemm_glsl(s, sc), grid(s, sc)))
    return out


def schedule_text(s: Shape, sc: Schedule) -> str:
    """TVM-style rendering of the schedule (for logs and the blog)."""
    la, lb, _, _ = eff_layouts(s, sc)
    lines = [f"# {s.id} {s.name}: C[{s.M},{s.N}] = A{'' if s.la == 'N' else 'ᵀ'}[...] @ B{'' if s.lb == 'N' else 'ᵀ'}[...], K={s.K}"]
    if s.la == "T":
        lines.append(f"s[A_T].compute_inline()" if sc.INLINE_A else "s[A_T].compute_root()  # materialized transpose kernel")
    if s.lb == "T":
        lines.append(f"s[B_T].compute_inline()" if sc.INLINE_B else "s[B_T].compute_root()  # materialized transpose kernel")
    lines += [
        f"mo, mi = s[C].split(m, {sc.BM}); no, ni = s[C].split(n, {sc.BN})",
        f"mt, mr = s[C].split(mi, {sc.TM}); nt, nr = s[C].split(ni, {sc.TN})",
        f"bind(mo, 'workgroup.y'); bind(no, 'workgroup.x')",
        f"bind({'nt' if sc.BIND == 'n' else 'mt'}, 'thread.fastest'); subgroup_size({sc.SG})",
    ]
    if sc.VTM * sc.VTN > 1:
        lines.append(f"vthread(mr, {sc.VTM}); vthread(nr, {sc.VTN})  # strided ownership, stride BM/{sc.VTM}, BN/{sc.VTN}")
    lines.append(f"ko, ki = s[C].split(k, {sc.BK})")
    if sc.KT > 1:
        lines.append(f"kr, kl = s[C].split(ki, nparts={sc.KT}{', cyclic' if sc.KMAP == 'cyclic' else ''}); "
                     f"CF = s.rfactor(C, kr); bind(kr, 'thread.reduce'); allreduce('{sc.RED}')")
    for op, on in (("A", sc.SMEM_A), ("B", sc.SMEM_B)):
        if on:
            lines.append(f"{op}S = s.cache_read({op}, 'shared', [C]); s[{op}S].compute_at(ko); vectorize({sc.VEC}); "
                         f"pad({sc.PAD}); smem_vector({smem_vec(sc, op)})" + ("; double_buffer()" if sc.DB else ""))
    KV, VM, VN, VS = vec_widths(s, sc)
    lines.append(f"vectorize(loads: k={KV} m={VM} n={VN}); vectorize(store, {VS}); unroll(kc, {sc.UNROLL})")
    lines.append("accumulate('f16', promote_to='f32', every=ki)" if sc.ACC == "f16p" else f"accumulate('{sc.ACC}')")
    return "\n".join(lines)
