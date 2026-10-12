"""Verifier: every candidate must pass this before it is allowed onto the GPU.

Three stages, cheapest first:
  1. static rules   legality against the device + numerics, each with a reason code
  2. mapping check  replay the schedule's thread -> (m, n, k) and loader index algebra (the same
                    helpers the code generator uses) and prove: every output of a workgroup is
                    owned exactly once, every k is reduced exactly once (rfactor coverage),
                    out-of-range rows/cols are guarded, each shared-memory tile element is loaded
                    exactly once
  3. compile        glslc + SPIR-V validation (done by the caller right before measuring)

After measurement the harness still compares C with the reference; a candidate that passes the
verifier but fails there counts against the verifier's precision.
"""

import collections
import dataclasses

from schedule import Schedule, cols, eff_layouts, grid, kslice, loader_plan, loader_unit, rows, thread_xy, vec_widths

DEVICES = {
    # Limits measured on the phone (vkinfo). Driver capability rules come from bugs/features we
    # hit while porting (see docs/BUGS.md).
    # max_waves / wave_granularity: Mesa freedreno_devices.py (FD710 = a7xx gen1)
    "turnip": dict(max_invocations=2048, shared_bytes=32768, max_groups=65535, clustered_subgroup=True,
                   subgroup_sizes=(64, 128), max_waves=16, wave_granularity=2),
    "stock": dict(max_invocations=1024, shared_bytes=32768, max_groups=65535, clustered_subgroup=False,
                  subgroup_sizes=(64,), max_waves=16, wave_granularity=2),
}
REG_CAP = 128          # 32-bit registers per invocation before we expect spills / starved occupancy
# fp16 accumulation: measured on the GPU (run1), error vs each lane's sequential chain length K/KT:
# <=256 -> max 4.0e-3 (0/59 over tolerance), 512 -> 7.0e-3 (8/8 over), 1024 -> 1.07e-2 (12/12 over).
F16_ACC_MAX_CHAIN = 256


def is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


def reg_estimate(s, sc: Schedule):
    KV, VM, VN, _ = vec_widths(s, sc)
    acc = sc.TM * sc.TN * {"f32": 1.0, "f16": 0.5, "f16p": 1.5}[sc.ACC]
    frags = (sc.TM + sc.TN) * KV * 0.5
    prefetch = 0.0
    if sc.DB:  # registers holding the next tile's loader vectors
        for op, on in (("A", sc.SMEM_A), ("B", sc.SMEM_B)):
            if on:
                p = loader_plan(s, sc, op)
                prefetch += -(-p["n"] // sc.WGS) * p["ux"] * p["uk"] * 0.5
    return acc + frags + prefetch + 12  # + indices / addresses


def smem_bytes(sc: Schedule):
    accb = 2 if sc.ACC == "f16" else 4
    b = 0
    if sc.SMEM_A:
        b += sc.BK * (sc.BM + sc.PAD) * 2
    if sc.SMEM_B:
        b += sc.BK * (sc.BN + sc.PAD) * 2
    if sc.KT > 1 and sc.RED == "smem":
        b += sc.BM * sc.BN * sc.KT * accb
    return b


def static_check(s, sc: Schedule, driver="turnip"):
    d = DEVICES[driver]
    v = []
    if sc.TM > sc.BM or sc.TN > sc.BN or sc.VTM > sc.TM or sc.VTN > sc.TN \
            or sc.BM % sc.TM or sc.BN % sc.TN or sc.TM % sc.VTM or sc.TN % sc.VTN:
        return ["tile-algebra"]  # nothing else is well defined (the workgroup could be empty)
    if s.K % sc.BK or sc.BK % sc.KT:
        v.append("k-tiling")
    else:
        KV = vec_widths(s, sc)[0]
        if (sc.KS // KV) % sc.UNROLL:
            v.append("unroll-divisibility")
    if not (1 <= sc.WGS <= d["max_invocations"]) or (sc.WGS > 1024 and sc.WGS % 1024):
        v.append("workgroup-size")
    if sc.SG not in d["subgroup_sizes"]:
        v.append("subgroup-size-unsupported")
    # a workgroup with a barrier must have all its waves resident at once (ir3 exits otherwise:
    # "impossible to have enough concurrent waves"); waves are allocated in pairs
    g = d["wave_granularity"]
    has_barrier = sc.SMEM_A or sc.SMEM_B or (sc.KT > 1 and sc.RED == "smem")
    waves = -(-sc.WGS // (sc.SG * g)) * g
    if has_barrier and waves > d["max_waves"]:
        v.append("barrier-wave-occupancy")
    # measured (run2: 5 of 5 exit, while KT >= 4 passes at 16 waves): with a 2-lane clustered add,
    # ir3 keeps a per-cluster loop whose branch stack lowers max_waves below 16
    elif has_barrier and waves == d["max_waves"] and sc.KT == 2 and sc.RED == "subgroup":
        v.append("barrier-wave-occupancy-kt2")
    if sc.KT > 1 and sc.RED == "subgroup":
        if not d["clustered_subgroup"]:
            v.append("no-clustered-subgroup-ops")
        if not is_pow2(sc.KT) or sc.SG % sc.KT:
            v.append("cluster-size")
        # >1024 invocations need a 2-D local size; the GPU then does not pack waves by linear
        # invocation index, so clustered adds combine the wrong lanes (measured: err ~0.9 at
        # 2048 threads, correct with RED=smem or at 1024 threads)
        if sc.WGS > 1024:
            v.append("subgroup-lane-order-2d")
    if sc.KT > 1 and sc.KMAP == "cyclic" and sc.VEC > 1 and not (sc.SMEM_A and sc.SMEM_B):
        pass  # allowed: cyclic k disables k-vectorization of direct loads (vec_widths handles it)
    if smem_bytes(sc) > d["shared_bytes"]:
        v.append("shared-memory")
    la, lb, _, _ = eff_layouts(s, sc)
    for op, on in (("A", sc.SMEM_A), ("B", sc.SMEM_B)):
        if not on:
            continue
        p = loader_plan(s, sc, op)
        glob = s.K if p["axis"] == "k" else (s.M if op == "A" else s.N)
        if p["n"] == 0 or glob % p["vw"]:
            v.append(f"loader-alignment-{op}")
    if sc.PAD and not (sc.SMEM_A or sc.SMEM_B):
        v.append("pad-without-smem")
    if reg_estimate(s, sc) > REG_CAP:
        v.append("register-pressure")
    if sc.ACC == "f16" and s.K // sc.KT > F16_ACC_MAX_CHAIN:
        v.append("numerics-f16-accumulation")
    if sc.ACC == "f16p" and sc.KS > F16_ACC_MAX_CHAIN:  # promoted: the fp16 chain is one k-tile slice
        v.append("numerics-f16-accumulation")
    if sc.DB and not (sc.SMEM_A or sc.SMEM_B):
        v.append("double-buffer-without-smem")
    gx, gy, _ = grid(s, sc)
    if gx > d["max_groups"] or gy > d["max_groups"]:
        v.append("grid-size")
    if (s.la == "N" and not sc.INLINE_A) or (s.lb == "N" and not sc.INLINE_B):
        v.append("inline-flag-on-untransposed-operand")
    # waste: tiles far larger than the problem (all-padding workgroups)
    if sc.BM > 2 * max(s.M, 1) and sc.BM > 8 or sc.BN > 2 * s.N:
        v.append("tile-exceeds-problem")
    return v


def mapping_check(s, sc: Schedule):
    """Replay the index algebra for one interior workgroup and the last (boundary) workgroup."""
    v = []
    owned = collections.Counter()
    for tid in range(sc.WGS):
        tx, ty, kt = thread_xy(sc, tid)
        if not (0 <= tx < sc.WX and 0 <= ty < sc.WY and 0 <= kt < sc.KT):
            return ["thread-decomposition"]
        if kt == 0:
            for r in rows(sc, ty):
                for c in cols(sc, tx):
                    owned[(r, c)] += 1
    if len(owned) != sc.BM * sc.BN or any(n != 1 for n in owned.values()):
        v.append("output-coverage")
    if any(not (0 <= r < sc.BM and 0 <= c < sc.BN) for r, c in owned):
        v.append("output-out-of-tile")
    kcount = collections.Counter(k for kt in range(sc.KT) for k in kslice(sc, kt))
    if sorted(kcount) != list(range(sc.BK)) or any(n != 1 for n in kcount.values()):
        v.append("reduction-coverage")
    # boundary workgroup: outputs beyond M/N must be guarded by the generated code
    gx, gy, _ = grid(s, sc)
    needs_m = gy * sc.BM > s.M
    needs_n = gx * sc.BN > s.N
    if needs_m != (s.M % sc.BM != 0) or needs_n != (s.N % sc.BN != 0):
        v.append("guard-mismatch")
    # shared-memory loaders: each tile element exactly once, and every shared write a whole vector
    # (two threads writing lanes of one shared vector race -- found on the GPU, see docs)
    for op, on in (("A", sc.SMEM_A), ("B", sc.SMEM_B)):
        if not on:
            continue
        p = loader_plan(s, sc, op)
        seen = collections.Counter()
        for u in range(p["n"]):
            k, x = loader_unit(p, u)
            seen.update((k + a, x + b) for a in range(p["uk"]) for b in range(p["ux"]))
            if x % p["sv"] or p["ux"] % p["sv"]:
                v.append(f"shared-vector-partial-write-{op}")
                break
        if len(seen) != sc.BK * p["bx"] or any(c != 1 for c in seen.values()) \
                or any(not (0 <= k < sc.BK and 0 <= x < p["bx"]) for k, x in seen):
            v.append(f"loader-coverage-{op}")
    return v


@dataclasses.dataclass
class Verdict:
    ok: bool
    reasons: list


def verify(s, sc: Schedule, driver="turnip"):
    r = static_check(s, sc, driver)
    if not r:
        r = mapping_check(s, sc)
    return Verdict(not r, r)
