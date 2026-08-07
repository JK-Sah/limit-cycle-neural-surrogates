"""
Classical reduced-order models of the wake, measured with the same delta.

The question this answers: is the period error a property of LEARNED dynamics,
or of reduced-order modelling of limit cycles generally? If a POD-based ROM
shows the same pathology, the finding is much broader than neural surrogates.
If it does not, the contrast says something specific about what learned
dynamics lose.

Two classical baselines, both standard and both non-intrusive:

  DMD     a_{k+1} = A a_k by least squares. The frequency is an explicit
          eigenvalue of A, so delta can be read off directly as well as
          measured from a rollout. Those two routes agreeing is a check on the
          period estimator itself.

  OpInf   quadratic operator inference (Peherstorfer & Willcox): fit
          adot = c + L a + Q (a kron a) to the projected data. This is the
          data-driven form of POD-Galerkin. Intrusive Galerkin is avoided
          deliberately: the snapshots live on a cropped wake window, so the
          domain is open and the boundary terms that intrusive projection
          discards do not vanish here.

Everything reuses the POD basis, the reference periods and the Hilbert period
estimator from param_wake.py, so the numbers are directly comparable to the
neural results rather than merely adjacent to them.
"""
import argparse, json, math, os
import numpy as np

from param_wake import (TRAIN_RE, INTERP_RE, EXTRAP_RE, build_basis, project,
                        period_from_signal, st_williamson)


# ------------------------------------------------------------------------ DMD
def dmd_fit(A_coef):
    """Least-squares one-step map on POD coefficients."""
    X, Y = A_coef[:-1].T, A_coef[1:].T
    return Y @ np.linalg.pinv(X)


def dmd_period(Amat, dt, T_guess):
    """
    Period from the eigenvalue whose implied period is closest to the reference.
    Taking the closest rather than the most energetic avoids reporting a
    harmonic as though it were the fundamental.
    """
    ev = np.linalg.eigvals(Amat)
    cand = []
    for lam in ev:
        if abs(lam) < 1e-12 or abs(np.imag(lam)) < 1e-12:
            continue
        mu = np.log(lam) / dt                      # continuous-time eigenvalue
        w = abs(np.imag(mu))
        if w > 1e-12:
            cand.append((2 * math.pi / w, abs(lam)))
    if not cand:
        return None, None
    T, mag = min(cand, key=lambda c: abs(c[0] - T_guess))
    return T, mag


# ---------------------------------------------------------------------- OpInf
def quad_index(r):
    return [(i, j) for i in range(r) for j in range(i, r)]


def quad_features(a, idx):
    """Unique quadratic products, [n, r(r+1)/2]."""
    return np.stack([a[..., i] * a[..., j] for i, j in idx], -1)


def opinf_fit(A_coef, dt, ridge=1e-8):
    """
    adot = c + L a + Q(a,a), least squares with ridge, central differences for
    adot. Ridge matters: without it the quadratic block is badly conditioned on
    data confined to a limit cycle, because the trajectory explores a
    two-dimensional set inside an r-dimensional space.
    """
    r = A_coef.shape[1]
    idx = quad_index(r)
    a = A_coef[1:-1]
    adot = (A_coef[2:] - A_coef[:-2]) / (2 * dt)
    D = np.concatenate([np.ones((len(a), 1)), a, quad_features(a, idx)], 1)
    G = D.T @ D + ridge * len(a) * np.eye(D.shape[1])
    W = np.linalg.solve(G, D.T @ adot)             # [1+r+q, r]
    return W, idx


def opinf_rhs(a, W, idx, r):
    f = np.concatenate([[1.0], a, np.array([a[i] * a[j] for i, j in idx])])
    return f @ W


def opinf_rollout(a0, W, idx, r, dt, n):
    out = np.empty((n + 1, r))
    a = a0.copy()
    out[0] = a
    for k in range(n):
        k1 = opinf_rhs(a, W, idx, r)
        k2 = opinf_rhs(a + 0.5 * dt * k1, W, idx, r)
        k3 = opinf_rhs(a + 0.5 * dt * k2, W, idx, r)
        k4 = opinf_rhs(a + dt * k3, W, idx, r)
        a = a + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(a).all() or np.abs(a).max() > 1e6:
            return out[:k + 1], False
        out[k + 1] = a
    return out, True


def rollout_delta(a0, W, idx, r, dt, T_true, n_periods):
    n = int(n_periods * T_true / dt)
    tr, ok = opinf_rollout(a0, W, idx, r, dt, n)
    if not ok or len(tr) < n // 2:
        return None, False
    t = np.arange(len(tr)) * dt
    h = len(tr) // 2
    return (period_from_signal(t[h:], tr[h:, 0]) - T_true) / T_true, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/resweep")
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--periods", type=int, default=100)
    ap.add_argument("--ridge", type=float, default=1e-8)
    ap.add_argument("--out", default="rom_wake.json")
    a = ap.parse_args()

    allre = TRAIN_RE + INTERP_RE + EXTRAP_RE
    pth = {R: os.path.join(a.data, f"Re{R}_snaps.npz") for R in allre}
    sig = {R: os.path.join(a.data, f"Re{R}_signal.npz") for R in allre}

    T_lat = {}
    for R in allre:
        d = np.load(sig[R])
        t, v = d["t"], d["v"]
        m = t > 0.5 * t.max()
        T_lat[R] = period_from_signal(t[m], v[m])

    print("shared POD basis from training Re only (same as the neural runs)")
    modes, mean, energy = build_basis([pth[R] for R in TRAIN_RE], a.modes)
    print(f"  energy: first 2 {energy[:2].sum()*100:.2f}%, "
          f"first {a.modes} {energy[:a.modes].sum()*100:.3f}%")

    coef, tsnap = {}, {}
    for R in allre:
        coef[R], tsnap[R] = project(pth[R], modes, mean)
    dt_lat = float(np.median(np.diff(tsnap[TRAIN_RE[0]])))
    T_snap = {R: T_lat[R] / dt_lat for R in allre}
    T_mid = float(np.median([T_snap[R] for R in TRAIN_RE]))
    dt = 2 * math.pi / T_mid
    T_model = {R: T_snap[R] * dt for R in allre}
    scale = float(np.abs(np.concatenate([coef[R] for R in TRAIN_RE])).max())
    A = {R: coef[R] / scale for R in allre}

    rows = []
    print("\n=== SINGLE-Re ROMs (fitted and evaluated at the same Re) ===")
    hdr = "{:>5} {:>8} {:>13} {:>13} {:>11} {:>11}".format(
        "Re", "split", "DMD delta", "OpInf delta", "DMD |lam|", "OpInf ok")
    print(hdr); print("-" * len(hdr))
    for R in allre:
        tag = ("train" if R in TRAIN_RE else
               "interp" if R in INTERP_RE else "extrap")
        Am = dmd_fit(A[R])
        Td, mag = dmd_period(Am, dt, T_model[R])
        dd = (Td - T_model[R]) / T_model[R] if Td else None
        W, idx = opinf_fit(A[R], dt, a.ridge)
        do, ok = rollout_delta(A[R][0], W, idx, a.modes, dt,
                               T_model[R], a.periods)
        rows.append(dict(kind="single-Re", Re=R, split=tag, dmd_delta=dd,
                         opinf_delta=do, opinf_stable=bool(ok),
                         dmd_mag=mag))
        print("{:>5} {:>8} {:>13} {:>13} {:>11} {:>11}".format(
            R, tag,
            f"{dd:+.3e}" if dd is not None else "n/a",
            f"{do:+.3e}" if do is not None else "unstable",
            f"{mag:.6f}" if mag else "n/a",
            "yes" if ok else "NO"))

    # ---- parametric: fit operators per training Re, interpolate across Re
    print("\n=== PARAMETRIC ROMs (operators interpolated to held-out Re) ===")
    mu = lambda R: (R - 130.0) / 70.0
    mu_tr = np.array([mu(R) for R in TRAIN_RE])
    Ws = np.stack([opinf_fit(A[R], dt, a.ridge)[0] for R in TRAIN_RE])
    _, idx = opinf_fit(A[TRAIN_RE[0]], dt, a.ridge)
    Ams = np.stack([dmd_fit(A[R]) for R in TRAIN_RE])
    # entrywise polynomial interpolation of the operators, the standard pROM
    degW = min(4, len(TRAIN_RE) - 1)
    cW = np.polyfit(mu_tr, Ws.reshape(len(TRAIN_RE), -1), degW)
    cA = np.polyfit(mu_tr, Ams.reshape(len(TRAIN_RE), -1), degW)

    hdr = "{:>5} {:>8} {:>13} {:>13} {:>11}".format(
        "Re", "split", "DMD delta", "OpInf delta", "OpInf ok")
    print(hdr); print("-" * len(hdr))
    for R in allre:
        tag = ("train" if R in TRAIN_RE else
               "interp" if R in INTERP_RE else "extrap")
        Wi = np.polyval(cW, mu(R)).reshape(Ws.shape[1:])
        Ai = np.polyval(cA, mu(R)).reshape(Ams.shape[1:])
        Td, _ = dmd_period(Ai, dt, T_model[R])
        dd = (Td - T_model[R]) / T_model[R] if Td else None
        do, ok = rollout_delta(A[R][0], Wi, idx, a.modes, dt,
                               T_model[R], a.periods)
        rows.append(dict(kind="parametric", Re=R, split=tag, dmd_delta=dd,
                         opinf_delta=do, opinf_stable=bool(ok)))
        print("{:>5} {:>8} {:>13} {:>13} {:>11}".format(
            R, tag,
            f"{dd:+.3e}" if dd is not None else "n/a",
            f"{do:+.3e}" if do is not None else "unstable",
            "yes" if ok else "NO"))

    print("\n=== SUMMARY: median |delta| ===")
    hdr = "{:>12} {:>10} {:>8} {:>13} {:>13}".format(
        "ROM", "kind", "split", "median", "n stable")
    print(hdr); print("-" * len(hdr))
    for kind in ["single-Re", "parametric"]:
        for key, lab in [("dmd_delta", "DMD"), ("opinf_delta", "OpInf")]:
            for tag in ["train", "interp", "extrap"]:
                g = [r for r in rows if r["kind"] == kind and r["split"] == tag]
                d = [abs(r[key]) for r in g if r[key] is not None]
                if d:
                    print("{:>12} {:>10} {:>8} {:>13.3e} {:>8}/{:<4}".format(
                        lab, kind, tag, np.median(d), len(d), len(g)))
    json.dump(dict(rows=rows, dt=dt, modes=a.modes,
                   T_model={str(k): v for k, v in T_model.items()}),
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
