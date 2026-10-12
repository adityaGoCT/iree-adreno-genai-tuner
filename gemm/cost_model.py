"""Cost model: ranks verified candidates before any of them touches the GPU.

  analytic(s, sc)   roofline estimate from the measured peaks (runs/roofline.json): padded FLOPs
                    at a reuse/occupancy-derated peak vs DRAM bytes implied by the tiling, with a
                    coalescing penalty per operand, + launch / reduction overheads.
  Learned           gradient-boosted trees on ~40 schedule + shape features (incl. the analytic
                    estimate) -> log(time). Trained online on every measurement, across shapes,
                    so later shapes start with transferred knowledge. Used once it has enough data;
                    until then the analytic estimate ranks.
"""

import json
import math
import pathlib

import numpy as np

from schedule import Schedule, eff_layouts, grid, vec_widths
from verifier import reg_estimate, smem_bytes

ROOF = json.loads((pathlib.Path(__file__).parent / "runs/roofline.json").read_text()) \
    if (pathlib.Path(__file__).parent / "runs/roofline.json").exists() else \
    {"f32_gflops": 368.4, "f16_gflops": 920.8, "read_gbs": 15.8, "copy_gbs": 14.4}
LAUNCH_US = 12.0


def lane_strides(s, sc: Schedule):
    """Bytes between the addresses two adjacent lanes read for each directly-loaded operand
    (0 = broadcast, 2 = perfectly coalesced fp16, large = scattered). Lanes vary fastest in kt
    (rfactor lanes) when KT > 1, then in tx (BIND n) or ty (BIND m)."""
    la, lb, _, _ = eff_layouts(s, sc)
    out = []
    for op, staged, lay, sub, along in (("A", sc.SMEM_A, la, sc.SM, "m"), ("B", sc.SMEM_B, lb, sc.SN, "n")):
        k_contig = (lay == "N") if op == "A" else (lay == "T")
        row = s.M if op == "A" else s.N  # elements between consecutive k when k is strided
        if staged:
            st = 2.0  # cooperative loader walks the contiguous axis
        elif sc.KT > 1:
            dk = sc.KS if sc.KMAP == "block" else 1
            st = dk * (1 if k_contig else row) * 2.0
        elif (sc.BIND == "n") == (op == "A"):
            st = 0.0  # neighbours share this operand's rows (A) / cols (B): broadcast
        else:
            st = sub * (s.K if k_contig else 1) * 2.0
        out.append(st)
    return tuple(out)


def bank_conflicts(s, sc: Schedule):
    """Shared-memory read conflict degree (lanes per bank among 32) for As and Bs in the k loop."""
    AS, BS = sc.BM + sc.PAD, sc.BN + sc.PAD
    dk = (sc.KS if sc.KMAP == "block" else 1) if sc.KT > 1 else None
    out = []
    for op, staged, rl, sub in (("A", sc.SMEM_A, AS, sc.SM), ("B", sc.SMEM_B, BS, sc.SN)):
        if not staged:
            out.append(1)
            continue
        if dk is not None:
            d = dk * rl
        elif (sc.BIND == "n") == (op == "A"):
            d = 0
        else:
            d = sub
        w = d // 2  # 4-byte words between neighbouring lanes
        out.append(1 if d <= 1 else math.gcd(w, 32) if w else 1)
    return tuple(out)


# Residency rules of Turnip's ir3 compiler for FD710 (a7xx gen1; Mesa freedreno_devices.py, ir3.c):
# a workgroup occupies ceil(WGS / (2*SG)) * 2 waves of SG lanes; a core holds at most 16 waves,
# fewer when registers run out (96 vec4 registers per lane-slot), and 32 KB of shared memory.
CORE = dict(max_waves=16, wave_granularity=2, reg_size_vec4=96, shared=32768, n_cores=1)
# Three constants calibrated on the correctness-fuzz measurements (fuzz.py -> runs/fuzz2, runs/fuzz3:
# random verified schedules, disjoint from the tuning runs): mean per-shape Spearman rho 0.62 (min 0.52).
LANES_FULL = 256  # active lanes per core needed to approach the measured peaks
PEN_CAP = 4.0     # max traffic multiplier of an uncoalesced operand (caches absorb the rest)
BANK_W = 0.25     # weight of shared-memory bank conflicts on the k loop


def occupancy(s, sc: Schedule):
    """(active lanes per core, resident workgroups per core, waves per workgroup)."""
    g = CORE["wave_granularity"]
    wg_waves = -(-sc.WGS // (sc.SG * g)) * g
    regs_vec4 = max(1, math.ceil(reg_estimate(s, sc) / 4))
    reg_waves = CORE["reg_size_vec4"] // (regs_vec4 * (2 if sc.SG == 128 else 1)) * g
    by_waves = min(CORE["max_waves"], reg_waves) // wg_waves
    sh = smem_bytes(sc)
    by_smem = CORE["shared"] // (-(-sh // 1024) * 1024) if sh else 1 << 30
    wgs = max(1, min(by_waves, by_smem))
    return wgs * sc.WGS, wgs, wg_waves


def analytic_us(s, sc: Schedule):
    gx, gy, _ = grid(s, sc)
    mp, np_ = gy * sc.BM, gx * sc.BN
    flops = 2.0 * mp * np_ * s.K
    peak = ROOF["f16_gflops"] if sc.ACC in ("f16", "f16p") else ROOF["f32_gflops"]
    reuse = (sc.TM * sc.TN) / (sc.TM + sc.TN)            # FMAs per loaded element per k
    eff_c = min(1.0, 0.25 + reuse / 4.0)
    lanes, wgs, _ = occupancy(s, sc)
    n_wg = gx * gy
    # small grids cannot fill the resident slots
    lanes = min(lanes, max(1, n_wg // CORE["n_cores"]) * sc.WGS) if n_wg < wgs * CORE["n_cores"] else lanes
    occ = min(1.0, lanes / LANES_FULL)
    ca, cb = bank_conflicts(s, sc)
    na, nb = (sc.TM if sc.SMEM_A else 0), (sc.TN if sc.SMEM_B else 0)
    bank = 1 + BANK_W * ((ca - 1) * na + (cb - 1) * nb) / (sc.TM + sc.TN)
    t_comp = flops * bank / (peak * 1e3 * eff_c * occ)   # us
    sa, sb = lane_strides(s, sc)
    pen = lambda st: 1.0 if st <= 8 else min(PEN_CAP, st / 8.0)  # noqa: E731
    a_bytes = gx * mp * s.K * 2.0 * pen(sa)
    b_bytes = gy * np_ * s.K * 2.0 * pen(sb)
    c_bytes = s.M * s.N * 2.0
    mat = 0.0
    n_disp = 1
    if s.la == "T" and not sc.INLINE_A:
        mat += 2 * s.M * s.K * 2.0
        n_disp += 1
    if s.lb == "T" and not sc.INLINE_B:
        mat += 2 * s.K * s.N * 2.0
        n_disp += 1
    # L2 absorbs re-reads of small operands: cap re-read factor by the operand's size
    a_bytes = min(a_bytes, s.M * s.K * 2.0 * max(1.0, gx / 4.0) * pen(sa))
    b_bytes = min(b_bytes, s.K * s.N * 2.0 * max(1.0, gy / 4.0) * pen(sb))
    occ_m = min(1.0, lanes / (LANES_FULL / 2))  # few lanes cannot keep enough loads in flight
    t_mem = (a_bytes + b_bytes + c_bytes + mat) / (ROOF["read_gbs"] * 1e3 * occ_m)
    red = (math.log2(sc.KT) * sc.TM * sc.TN * gx * gy * sc.WGS / sc.KT) / (peak * 1e3 / 8) if sc.KT > 1 else 0
    # barriers: two per k-tile when staging through shared memory; each wave of every workgroup waits
    bar = 0.0
    if sc.SMEM_A or sc.SMEM_B:
        bar = 2 * (s.K // sc.BK) * -(-n_wg // (wgs * CORE["n_cores"])) * 0.05  # ~50 ns per barrier round
    return max(t_comp, t_mem) + red + bar + n_disp * LAUNCH_US


FEATURE_NAMES = None


def features(s, sc: Schedule):
    global FEATURE_NAMES
    la, lb, _, _ = eff_layouts(s, sc)
    gx, gy, _ = grid(s, sc)
    KV, VM, VN, VS = vec_widths(s, sc)
    sa, sb = lane_strides(s, sc)
    f = {
        "logM": math.log2(s.M), "logN": math.log2(s.N), "logK": math.log2(s.K),
        "A_T": s.la == "T", "B_T": s.lb == "T",
        "logBM": math.log2(sc.BM), "logBN": math.log2(sc.BN), "logBK": math.log2(sc.BK),
        "TM": sc.TM, "TN": sc.TN, "VTM": sc.VTM, "VTN": sc.VTN, "logKT": math.log2(sc.KT),
        "red_smem": sc.RED == "smem" and sc.KT > 1, "kmap_cyclic": sc.KMAP == "cyclic" and sc.KT > 1,
        "SMEM_A": sc.SMEM_A, "SMEM_B": sc.SMEM_B, "DB": sc.DB, "VEC": sc.VEC, "PAD": sc.PAD, "UNROLL": sc.UNROLL,
        "acc_f16": sc.ACC == "f16", "acc_f16p": sc.ACC == "f16p", "mat_A": s.la == "T" and not sc.INLINE_A,
        "mat_B": s.lb == "T" and not sc.INLINE_B, "bind_m": sc.BIND == "m", "SG": sc.SG / 64,
        "logWGS": math.log2(sc.WGS), "logWG": math.log2(gx * gy), "logThreads": math.log2(gx * gy * sc.WGS),
        "tile_eff": (s.M * s.N) / (gx * sc.BN * gy * sc.BM), "regs": reg_estimate(s, sc),
        "smem": smem_bytes(sc) / 1024, "KV": KV, "VM": VM, "VN": VN, "VS": VS,
        "reuse": (sc.TM * sc.TN) / (sc.TM + sc.TN), "strideA": math.log2(1 + sa), "strideB": math.log2(1 + sb),
        "eff_A": la == "N", "eff_B": lb == "N",
        "bankA": bank_conflicts(s, sc)[0], "bankB": bank_conflicts(s, sc)[1],
        "log_lanes": math.log2(occupancy(s, sc)[0]), "wgs_resident": occupancy(s, sc)[1],
        "wg_waves": occupancy(s, sc)[2], "lane_eff": sc.WGS / (occupancy(s, sc)[2] * sc.SG),
        "log_analytic": math.log(analytic_us(s, sc)),
    }
    FEATURE_NAMES = list(f)
    return np.array([float(x) for x in f.values()])


class Learned:
    """Online gradient-boosted regressor on log(us); falls back to the analytic model."""

    def __init__(self, min_samples=24):
        self.X, self.y = [], []
        self.model = None
        self.min_samples = min_samples

    def add(self, s, sc, us):
        self.X.append(features(s, sc))
        self.y.append(math.log(us))

    def fit(self):
        if len(self.y) < self.min_samples:
            self.model = None
            return
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=15,
                                                   min_samples_leaf=4, l2_regularization=0.1, random_state=0)
        self.model.fit(np.array(self.X), np.array(self.y))

    def predict_us(self, s, scheds):
        if not scheds:
            return np.array([])
        if self.model is None:
            return np.array([analytic_us(s, sc) for sc in scheds])
        X = np.array([features(s, sc) for sc in scheds])
        return np.exp(self.model.predict(X))

    @property
    def kind(self):
        return "learned" if self.model is not None else "analytic"
