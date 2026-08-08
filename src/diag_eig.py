"""Is the parametric-DMD failure a frequency error, or loss of neutral stability?"""
import math, os
import numpy as np
from param_wake import (TRAIN_RE, INTERP_RE, EXTRAP_RE, build_basis, project,
                        period_from_signal)
from rom_wake import dmd_fit

allre = TRAIN_RE + INTERP_RE + EXTRAP_RE
D = "data/resweep"
T_lat = {}
for R in allre:
    d = np.load(f"{D}/Re{R}_signal.npz"); t, v = d["t"], d["v"]
    m = t > 0.5*t.max(); T_lat[R] = period_from_signal(t[m], v[m])
modes, mean, _ = build_basis([f"{D}/Re{R}_snaps.npz" for R in TRAIN_RE], 16)
coef, ts = {}, {}
for R in allre:
    coef[R], ts[R] = project(f"{D}/Re{R}_snaps.npz", modes, mean)
dtl = float(np.median(np.diff(ts[TRAIN_RE[0]])))
Tsn = {R: T_lat[R]/dtl for R in allre}
Tmid = float(np.median([Tsn[R] for R in TRAIN_RE]))
dt = 2*math.pi/Tmid
scale = float(np.abs(np.concatenate([coef[R] for R in TRAIN_RE])).max())
A = {R: coef[R]/scale for R in allre}
muf = lambda R: (R-130.0)/70.0
mu_tr = np.array([muf(R) for R in TRAIN_RE])
Ams = np.stack([dmd_fit(A[R]) for R in TRAIN_RE])
cA = np.polyfit(mu_tr, Ams.reshape(len(TRAIN_RE),-1), 4)

print("spectral radius of the DMD operator")
print(f"{'Re':>5} {'split':>8} {'fitted at Re':>14} {'entry-interp':>14} {'growth/3000 steps':>20}")
print("-"*66)
for R in allre:
    tag = "train" if R in TRAIN_RE else "interp" if R in INTERP_RE else "extrap"
    direct = max(abs(np.linalg.eigvals(dmd_fit(A[R]))))
    interp = max(abs(np.linalg.eigvals(np.polyval(cA, muf(R)).reshape(Ams.shape[1:]))))
    g = interp**3000
    gs = f"{g:.3e}" if g < 1e100 else "overflow"
    print(f"{R:>5} {tag:>8} {direct:>14.8f} {interp:>14.8f} {gs:>20}")
print("\nA rollout of 100 periods is ~3000 steps. Neutral stability is not")
print("something entrywise interpolation preserves, and the frequency")
print("correction alone does not restore it.")
