"""
optDMD against exact DMD, measured the same way as every learned model.

Two things were wrong with the first optDMD run. Its delta was read
ANALYTICALLY from the fitted omega while every learned model's was measured by
rolling out 100 periods, and the estimator used at the time was the biased
Hilbert one. Redo with a single protocol: roll every model forward 100 periods
and measure with period.period_from_signal. The analytic value is reported
alongside, as a check on the rollout path rather than as the headline.
"""
import json, math
import numpy as np
from period import period_from_signal
from param_wake import TRAIN_RE
from rom_wake import dmd_fit
from optdmd import optdmd_free, optdmd_harmonic, dominant_period, \
    _basis_free, _basis_harm, _coeffs

D = "data/resweep"
_c = {}


def load(R):
    if R not in _c:
        z = np.load(f"{D}/Re{R}_snaps.npz")
        _c[R] = (z["snaps"].reshape(len(z["snaps"]), -1).astype(np.float64),
                 z["t"].astype(float))
    return _c[R]


def basis(rs, r, sub=300, seed=0):
    rng = np.random.default_rng(seed)
    chunks = []
    for R in rs:
        X, _ = load(R)
        i = np.sort(rng.choice(len(X), min(sub, len(X)), False))
        chunks.append(X[i])
    X = np.concatenate(chunks, 0)
    mean = X.mean(0); Xc = X - mean
    with np.errstate(all="ignore"):
        G = Xc @ Xc.T
    w, V = np.linalg.eigh(G)
    w, V = np.clip(w[::-1], 0, None), V[:, ::-1]
    return (Xc.T @ V[:, :r]) / np.sqrt(np.maximum(w[:r], 1e-30)), mean


d = np.load(f"{D}/Re100_signal.npz"); tp, v = d["t"], d["v"]
m = tp > 0.5 * tp.max()
T_lat = period_from_signal(tp[m], v[m])
T_M = 2 * math.pi
NP = 100

rows = []
hdr = "{:>20} {:>3} {:>17} {:>13} {:>13} {:>11}".format(
    "basis", "r", "method", "rolled out", "analytic", "horizon")
print(hdr); print("-" * len(hdr))
for label, rs in [("Re=100 only", [100]), ("pooled over 6 Re", TRAIN_RE)]:
    for r in [8, 16]:
        modes, mean = basis(rs, r)
        Xf, ts = load(100)
        with np.errstate(all="ignore"):
            coef = (Xf - mean) @ modes
        dt = 2 * math.pi / (T_lat / float(np.median(np.diff(ts))))
        A = coef / float(np.abs(coef).max())
        t = np.arange(len(A)) * dt
        n = int(NP * T_M / dt)
        t_ro = np.arange(n + 1) * dt

        res = []
        # exact DMD: iterate the one-step map
        Am = dmd_fit(A)
        tr = np.empty((n + 1, r)); x = A[0].copy(); tr[0] = x
        ok = True
        for k in range(n):
            x = Am @ x
            if not np.isfinite(x).all() or np.abs(x).max() > 1e8:
                ok = False; break
            tr[k + 1] = x
        T_ro = period_from_signal(t_ro, tr[:, 0]) if ok else float("nan")
        lam = np.linalg.eigvals(Am)
        mu = np.log(lam.astype(complex)) / dt
        osc = [k for k in range(len(mu)) if abs(np.imag(mu[k])) > 1e-12]
        kk = min(osc, key=lambda k: abs(2*math.pi/abs(np.imag(mu[k])) - T_M))
        res.append(("exact DMD", T_ro, 2 * math.pi / abs(np.imag(mu[kk]))))

        # optDMD free: evaluate the fitted exponential model forward
        p = max(2, r // 2)
        sig, om, amp, _ = optdmd_free(A, t, p)
        T_an, _k = dominant_period(om, amp, T_M)
        tc = t - t.mean()
        B = _coeffs(_basis_free(np.concatenate([sig, om]), tc, p), A)
        traj = _basis_free(np.concatenate([sig, om]), t_ro - t.mean(), p) @ B
        res.append(("optDMD free", period_from_signal(t_ro, traj[:, 0]), T_an))

        # optDMD harmonic
        wh, _ = optdmd_harmonic(A, t, min(6, p), 2 * math.pi / T_M)
        K = min(6, p)
        Bh = _coeffs(_basis_harm(wh, tc, K), A)
        trh = _basis_harm(wh, t_ro - t.mean(), K) @ Bh
        res.append(("optDMD harmonic", period_from_signal(t_ro, trh[:, 0]),
                    2 * math.pi / wh))

        for name, Tr, Ta in res:
            dr = (Tr - T_M) / T_M
            da = (Ta - T_M) / T_M
            rows.append(dict(basis=label, modes=r, method=name,
                             delta_rollout=float(dr), delta_analytic=float(da)))
            print("{:>20} {:>3} {:>17} {:>13.3e} {:>13.3e} {:>11.0f}".format(
                label, r, name, dr, da,
                1/(4*abs(dr)) if dr and np.isfinite(dr) else float('inf')))

print("\n--- exact DMD vs optDMD, both rolled out and measured identically ---")
for label in ["Re=100 only", "pooled over 6 Re"]:
    for r in [8, 16]:
        g = {x["method"]: abs(x["delta_rollout"]) for x in rows
             if x["basis"] == label and x["modes"] == r}
        if g.get("optDMD free"):
            print(f"  {label:>18} r={r:<3} exact/optDMD = "
                  f"{g['exact DMD']/g['optDMD free']:>10.1f}x")
print("\n--- basis sensitivity of each method (Re=100-only / pooled) ---")
for meth in ["exact DMD", "optDMD free"]:
    for r in [8, 16]:
        a = [abs(x["delta_rollout"]) for x in rows if x["method"] == meth
             and x["modes"] == r and x["basis"] == "Re=100 only"]
        b = [abs(x["delta_rollout"]) for x in rows if x["method"] == meth
             and x["modes"] == r and x["basis"] == "pooled over 6 Re"]
        if a and b and b[0] > 0:
            print(f"  {meth:>14} r={r:<3} ratio = {a[0]/b[0]:>10.1f}x")
json.dump(rows, open("results/rom/optdmd_fair.json", "w"), indent=1)
print("\nwrote results/rom/optdmd_fair.json")
