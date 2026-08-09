"""
DMD at Re=100 gave delta 2e-8 in one run and 4.6e-4 in another. The sweep shows
truncation is NOT the cause: delta is flat at 4.6e-4 from 4 to 32 modes. The
other variable was the BASIS -- one run used a POD basis pooled over six
Reynolds numbers, the other a basis from Re=100 alone. Isolate it.
"""
import math
import numpy as np
from param_wake import TRAIN_RE, build_basis, project, period_from_signal
from rom_wake import dmd_fit, dmd_period

D = "data/resweep"
d = np.load(f"{D}/Re100_signal.npz"); t, v = d["t"], d["v"]
m = t > 0.5 * t.max()
T_lat = period_from_signal(t[m], v[m])

print(f"{'basis':>34} {'modes':>6} {'energy%':>10} {'delta':>12} "
      f"{'|lambda|':>12} {'horizon':>10}")
print("-" * 88)
for label, paths in [("Re=100 only", [f"{D}/Re100_snaps.npz"]),
                     ("pooled over 6 training Re",
                      [f"{D}/Re{R}_snaps.npz" for R in TRAIN_RE])]:
    for r in [8, 16]:
        modes, mean, energy = build_basis(paths, r, per_re_sub=300)
        coef, ts = project(f"{D}/Re100_snaps.npz", modes, mean)
        dt_lat = float(np.median(np.diff(ts)))
        dt = 2 * math.pi / (T_lat / dt_lat)
        A = coef / float(np.abs(coef).max())
        Am = dmd_fit(A)
        Td, mag = dmd_period(Am, dt, 2 * math.pi)
        dl = (Td - 2 * math.pi) / (2 * math.pi)
        print(f"{label:>34} {r:>6} {energy[:r].sum()*100:>10.5f} "
              f"{dl:>+12.3e} {mag:>12.8f} {1/(4*abs(dl)):>10.0f}")
print("\nIf the two bases differ at matched mode count, the earlier 4-orders-of-")
print("magnitude spread was a basis effect, not a truncation effect.")
