"""
Parametric DMD done three ways, because the first way is a strawman.

Interpolating operator ENTRIES across Reynolds number gives a 3-period horizon
at held-out Re. That is a real result about a real practice, but it is not the
best a careful person would do, and reporting it alone would be unfair to the
classical side. Small entrywise perturbations move eigenvalues a long way, and
in an oscillatory system the frequency IS an eigenvalue.

Three constructions:

  entry   interpolate A(Re) entrywise.                      (the strawman)
  eig     match modes across Re, interpolate the continuous-time eigenvalues
          and gauge-fixed eigenvectors separately, rebuild A.  (fair pDMD)
  freq    take the entry-interpolated operator and correct only its dominant
          eigenvalue to a frequency taken from a constrained fit to the
          MEASURED training frequencies.               (classical + anchoring)

The third is the interesting one. It is phase anchoring applied to a classical
ROM rather than a neural one. If it recovers the loss, then the remedy is
"impose the frequency" and has nothing to do with whether the model is learned.
"""
import argparse, json, math, os
import numpy as np

from param_wake import (TRAIN_RE, INTERP_RE, EXTRAP_RE, build_basis, project,
                        period_from_signal)
from rom_wake import dmd_fit, dmd_period


def sorted_eig(A, dt):
    """
    Eigendecomposition ordered so modes can be matched across Re, with the
    eigenvector gauge fixed. Eigenvectors are defined up to a complex scale;
    without fixing it, interpolating them entrywise is meaningless.
    """
    lam, V = np.linalg.eig(A)
    mu = np.log(lam.astype(complex)) / dt
    order = np.argsort(np.imag(mu))          # conjugate pairs sit symmetrically
    lam, mu, V = lam[order], mu[order], V[:, order]
    for k in range(V.shape[1]):
        j = np.argmax(np.abs(V[:, k]))       # largest component -> real positive
        ph = V[j, k] / abs(V[j, k]) if abs(V[j, k]) > 0 else 1.0
        V[:, k] = V[:, k] / ph
        V[:, k] = V[:, k] / np.linalg.norm(V[:, k])
    return lam, mu, V


def interp_eig(mus, Vs, mu_train, mu_query, deg):
    """Interpolate continuous-time eigenvalues and eigenvectors across Re."""
    cm = np.polyfit(mu_train, mus, deg)
    cv = np.polyfit(mu_train, Vs.reshape(len(mu_train), -1), deg)
    mu_q = np.polyval(cm, mu_query)
    V_q = np.polyval(cv, mu_query).reshape(Vs.shape[1:])
    return mu_q, V_q


def rebuild(mu, V, dt):
    """A = V exp(mu dt) V^-1, real part; V may be mildly ill-conditioned."""
    lam = np.exp(mu * dt)
    A = V @ np.diag(lam) @ np.linalg.pinv(V)
    return np.real(A)


def repair(A, dt, T_guess, omega=None, neutral=False, stabilize=False):
    """
    Repair an interpolated operator's spectrum, one property at a time.

      stabilize  clip every non-dominant |lambda| > 1 back onto the unit
                 circle. Generic ROM stabilisation, uses no frequency
                 knowledge.
      neutral    put the dominant oscillatory pair exactly ON the unit circle,
                 which is what Floquet theory requires of a limit cycle.
      omega      additionally set that pair's frequency to a measured value.

    Separating these answers which property actually matters. Entrywise
    interpolation destroys both: operators fitted at a single Re have spectral
    radius 1 to eight decimals, and interpolated ones reach 31.
    """
    lam, V = np.linalg.eig(A)
    mu = np.log(lam.astype(complex)) / dt
    osc = [k for k in range(len(mu)) if abs(np.imag(mu[k])) > 1e-12]
    dom = []
    if osc:
        k = min(osc, key=lambda k: abs(2 * math.pi / abs(np.imag(mu[k]))
                                       - T_guess))
        w0 = abs(np.imag(mu[k]))
        dom = [j for j in range(len(mu))
               if abs(abs(np.imag(mu[j])) - w0) < 1e-9 * max(1.0, w0)]
    for j in range(len(mu)):
        re, im = np.real(mu[j]), np.imag(mu[j])
        if j in dom:
            if neutral:
                re = 0.0
            if omega is not None:
                im = math.copysign(omega, im) if im != 0 else omega
        elif stabilize and re > 0:
            re = 0.0                          # clip onto the unit circle
        mu[j] = re + 1j * im
    return np.real(V @ np.diag(np.exp(mu * dt)) @ np.linalg.pinv(V))


def rollout_period(A, a0, dt, T_true, n_periods):
    n = int(n_periods * T_true / dt)
    tr = np.empty((n + 1, len(a0)))
    a = a0.copy(); tr[0] = a
    for k in range(n):
        a = A @ a
        if not np.isfinite(a).all() or np.abs(a).max() > 1e8:
            return None
        tr[k + 1] = a
    t = np.arange(len(tr)) * dt
    h = len(tr) // 2
    return period_from_signal(t[h:], tr[h:, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/resweep")
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--periods", type=int, default=100)
    ap.add_argument("--out", default="rom_parametric.json")
    a = ap.parse_args()

    allre = TRAIN_RE + INTERP_RE + EXTRAP_RE
    pth = {R: os.path.join(a.data, f"Re{R}_snaps.npz") for R in allre}
    sig = {R: os.path.join(a.data, f"Re{R}_signal.npz") for R in allre}

    T_lat = {}
    for R in allre:
        d = np.load(sig[R]); t, v = d["t"], d["v"]
        m = t > 0.5 * t.max()
        T_lat[R] = period_from_signal(t[m], v[m])

    modes, mean, _ = build_basis([pth[R] for R in TRAIN_RE], a.modes)
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

    mu_of = lambda R: (R - 130.0) / 70.0
    mu_tr = np.array([mu_of(R) for R in TRAIN_RE])
    re_tr = np.array([float(R) for R in TRAIN_RE])
    deg = min(4, len(TRAIN_RE) - 1)

    # per-Re DMD fits and their measured frequencies
    Ams, mus, Vs, om_meas = [], [], [], []
    for R in TRAIN_RE:
        Am = dmd_fit(A[R])
        lam, mu, V = sorted_eig(Am, dt)
        Ams.append(Am); mus.append(mu); Vs.append(V)
        Td, _ = dmd_period(Am, dt, T_model[R])
        om_meas.append(2 * math.pi / Td)
    Ams, mus, Vs = np.stack(Ams), np.stack(mus), np.stack(Vs)
    om_meas = np.array(om_meas)

    cA = np.polyfit(mu_tr, Ams.reshape(len(TRAIN_RE), -1), deg)
    # constrained frequency interpolant, same physical form used for the
    # neural anchored model
    W = np.stack([1.0 / re_tr, np.ones_like(re_tr), re_tr], 1)
    cw = np.linalg.lstsq(W, om_meas, rcond=None)[0]
    om_fit = lambda R: cw[0] / R + cw[1] + cw[2] * R

    print("parametric DMD, three constructions")
    hdr = "{:>5} {:>8} {:>11} {:>11} {:>11} {:>11} {:>13}".format(
        "Re", "split", "entry", "eig", "stab", "neutral", "neutral+freq")
    print(hdr); print("-" * len(hdr))
    rows = []
    for R in allre:
        tag = ("train" if R in TRAIN_RE else
               "interp" if R in INTERP_RE else "extrap")
        q = mu_of(R)
        out = {}

        A_entry = np.polyval(cA, q).reshape(Ams.shape[1:])
        mu_q, V_q = interp_eig(mus, Vs, mu_tr, q, deg)
        A_eig = rebuild(mu_q, V_q, dt)
        Tg = T_model[R]
        A_stab = repair(A_entry, dt, Tg, stabilize=True)
        A_neut = repair(A_entry, dt, Tg, neutral=True, stabilize=True)
        A_full = repair(A_entry, dt, Tg, omega=om_fit(R), neutral=True,
                        stabilize=True)

        for name, Amat in [("entry", A_entry), ("eig", A_eig),
                           ("stab", A_stab), ("neutral", A_neut),
                           ("neutral+freq", A_full)]:
            Tp = rollout_period(Amat, A[R][0], dt, T_model[R], a.periods)
            out[name] = ((Tp - T_model[R]) / T_model[R]) if Tp else None
        rows.append(dict(Re=R, split=tag, **out,
                         omega_fit_delta=(om_fit(R) - 2 * math.pi / T_model[R])
                         / (2 * math.pi / T_model[R])))
        print("{:>5} {:>8} {:>11} {:>11} {:>11} {:>11} {:>13}".format(
            R, tag, *[f"{out[k]:+.2e}" if out[k] is not None else "unstable"
                      for k in ["entry", "eig", "stab", "neutral",
                                "neutral+freq"]]))

    print("\nmedian |delta| and usable horizon")
    hdr = "{:>10} {:>8} {:>13} {:>12} {:>9}".format(
        "method", "split", "median", "horizon", "stable")
    print(hdr); print("-" * len(hdr))
    for k in ["entry", "eig", "stab", "neutral", "neutral+freq"]:
        for tag in ["train", "interp", "extrap"]:
            g = [r for r in rows if r["split"] == tag]
            v = [abs(r[k]) for r in g if r[k] is not None]
            if v:
                m = np.median(v)
                print("{:>10} {:>8} {:>13.3e} {:>10.0f} P {:>6}/{:<3}".format(
                    k, tag, m, 1 / (4 * m), len(v), len(g)))
    # what the frequency interpolant alone implies, for comparison
    print("\nfor reference, the frequency interpolant's own error:")
    for tag, grp in [("train", TRAIN_RE), ("interp", INTERP_RE),
                     ("extrap", EXTRAP_RE)]:
        v = [abs(r["omega_fit_delta"]) for r in rows if r["split"] == tag]
        print(f"  {tag:>6}: {np.median(v):.3e}  ({1/(4*np.median(v)):.0f} P)")

    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
