"""Diff --iree-flow-trace-dispatch-tensors logs from two devices; report first divergent dispatch."""
import re, sys
import numpy as np

def parse(path):
    out, cur = [], None
    for line in open(path, errors="replace"):
        m = re.match(r"=== (\S+) (inputs|outputs) ===", line)
        if m:
            cur = [m.group(1), m.group(2), []]
            out.append(cur)
        elif cur is not None and "=" in line and re.match(r"^[\dx]*x?[a-z]+\d*=", line.strip()):
            shape_t, vals = line.strip().split("=", 1)
            vals = re.sub(r"[\[\]]", " ", vals).split()
            try:
                cur[2].append((shape_t, np.array([float(v) for v in vals])))
            except ValueError:
                cur[2].append((shape_t, None))
    return out

a, b = parse(sys.argv[1]), parse(sys.argv[2])
print(f"{len(a)} vs {len(b)} trace records")
shown = 0
for (na, ka, ta), (nb, kb, tb) in zip(a, b):
    assert na == nb and ka == kb, (na, nb)
    for i, ((sa, va), (sb, vb)) in enumerate(zip(ta, tb)):
        if va is None or vb is None or va.size != vb.size:
            continue
        err = np.abs(va - vb).max() if va.size else 0
        scale = max(1.0, np.abs(vb).max() if vb.size else 1)
        if err > 1e-3 * scale:
            print(f"DIVERGE {ka:7s} #{i} {na}  {sa}  max|Δ|={err:.3g}  |ref|max={scale:.3g}")
            shown += 1
            if shown >= int(sys.argv[3]) if len(sys.argv) > 3 else 8:
                sys.exit()
print("no more divergences" if shown else "traces match")
