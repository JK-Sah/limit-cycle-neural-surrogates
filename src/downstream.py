"""
What does the period error actually cost you?

A diagnostic is only worth reporting if it predicts something a practitioner
cares about. The claim under test is a split, not a slope:

  phase-INVARIANT quantities  (time-mean field, RMS fluctuation field, the
                               shedding frequency itself) should be immune to
                               delta, because they do not depend on where in the
                               cycle the surrogate thinks it is;

  phase-RESOLVED quantities   (the field at a given instant, anything
                               phase-locked to physical time -- forecasting,
                               data assimilation, phase-locked actuation)
                               should decorrelate at N ~ 1/(4 delta) cycles.

If that split holds, the practical statement is sharp: time-averaged statistics
from a surrogate are trustworthy indefinitely, and anything phase-resolved has a
shelf life you can compute in advance from a ten-period rollout.

Ground truth over long horizons comes from phase-wrapping the reference
trajectory. The wake is a converged limit cycle -- measured period jitter is
5e-7 -- so the true state at time t is the reference state at t mod T, to well
below any error being measured here.

Everything is computed in POD coefficient space. The modes are orthonormal, so
the L2 error of a decoded field equals the L2 error of its coefficients, and
there is no need to reconstruct 36864-dimensional fields for 600,000 steps.
"""
import argparse, json, math, os, time
import numpy as np
import torch

from period import period_from_signal
from param_wake import build_basis, project
from arch_breadth import ARCH, rollout, train
from rom_wake import dmd_fit

torch.set_default_dtype(torch.float64)


def wrapped_truth(A_ref, t_query, T_snap):
    """
    Reference state at arbitrary times, by wrapping into one period and
    interpolating. A_ref is one long reference trajectory sampled every step.
    """
    idx = np.mod(t_query, T_snap)
    lo = np.floor(idx).astype(int)
    frac = (idx - lo)[:, None]
    hi = (lo + 1) % len(A_ref)
    lo = lo % len(A_ref)
    return A_ref[lo] * (1 - frac) + A_ref[hi] * frac


def measure(pred, truth, n_per_period):
    """
    pred, truth : [n, r] over the same times.

    Phase-invariant quantities are normalised by the fluctuation scale, not by
    their own magnitude. POD coefficients have zero time-mean by construction,
    so dividing the mean error by the mean itself divides by ~0 and produces
    numbers that swing over three orders of magnitude with the endpoint phase.
    """
    d = dict()
    scale = float(np.linalg.norm(truth.std(0))) + 1e-30

    # phase-invariant
    d["err_mean"] = float(np.linalg.norm(pred.mean(0) - truth.mean(0)) / scale)
    d["err_rms"] = float(np.linalg.norm(pred.std(0) - truth.std(0)) / scale)
    ct = np.cov(truth, rowvar=False)
    cp = np.cov(pred, rowvar=False)
    d["err_cov"] = float(np.linalg.norm(cp - ct) /
                         (np.linalg.norm(ct) + 1e-30))

    # phase-resolved: instantaneous error over the final period
    k = int(n_per_period)
    p, t = pred[-k:], truth[-k:]
    d["err_inst"] = float(np.linalg.norm(p - t) /
                          (np.linalg.norm(t) + 1e-30))
    return d


def predicted_err_inst(delta, N):
    """
    For two signals of the same shape differing only in period, the relative
    L2 error after N cycles is 2|sin(pi delta N)|: a phase slip of
    dtheta = 2 pi delta N gives RMS error 2|sin(dtheta/2)| relative to the
    signal's own RMS. It does not saturate -- it recurs to zero whenever the
    slip passes a multiple of 2 pi -- and it first reaches sqrt(2) at
    N = 1/(4 delta).
    """
    if delta is None:
        return None
    return float(2 * abs(math.sin(math.pi * delta * N)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/resweep")
    ap.add_argument("--Re", type=int, default=100)
    ap.add_argument("--modes", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--seg-len", type=int, default=25)
    ap.add_argument("--seed-list", type=int, nargs="+", default=[0])
    ap.add_argument("--horizons", type=int, nargs="+",
                    default=[10, 30, 100, 300, 1000, 3000, 10000, 30000])
    ap.add_argument("--out", default="downstream.json")
    a = ap.parse_args()

    snap = os.path.join(a.data, f"Re{a.Re}_snaps.npz")
    d = np.load(os.path.join(a.data, f"Re{a.Re}_signal.npz"))
    tt, v = d["t"], d["v"]
    m = tt > 0.5 * tt.max()
    T_lat = period_from_signal(tt[m], v[m])

    modes, mean, energy = build_basis([snap], a.modes, per_re_sub=600)
    coef, tsnap = project(snap, modes, mean)
    dt_lat = float(np.median(np.diff(tsnap)))
    T_snap = T_lat / dt_lat                      # period in snapshots
    dt = 2 * math.pi / T_snap                    # model time per snapshot
    scale = float(np.abs(coef).max())
    A = coef / scale
    print(f"Re={a.Re}  T={T_snap:.4f} snapshots/period  "
          f"POD {energy[:a.modes].sum()*100:.4f}%  "
          f"{len(A)} reference snapshots = {len(A)/T_snap:.1f} periods")

    segs = torch.tensor(np.stack(
        [A[i:i + a.seg_len + 1] for i in range(len(A) - a.seg_len - 1)]))

    rows = []
    for name, (cls, kw) in ARCH.items():
        for seed in a.seed_list:
            t0 = time.time()
            torch.manual_seed(seed)
            model = cls(a.modes, **kw)
            mse = train(model, segs, dt, a.epochs, 3e-3, seed=seed)

            with torch.no_grad():
                n = int(max(a.horizons) * T_snap)
                tr = rollout(model, torch.tensor(A[0]).unsqueeze(0),
                             dt, n)[:, 0, :].numpy()
            good = np.isfinite(tr).all() and np.abs(tr).max() < 1e6
            # delta from the first 100 periods, which is what a practitioner
            # would actually be able to measure
            k100 = int(100 * T_snap)
            dl = None
            if good:
                Tm = period_from_signal(np.arange(k100 + 1) * dt,
                                        tr[:k100 + 1, 0])
                dl = (Tm - 2 * math.pi) / (2 * math.pi)

            for N in a.horizons:
                k = int(N * T_snap)
                if not good or k >= len(tr):
                    rows.append(dict(arch=name, seed=seed, N=N, delta=dl,
                                     train_mse=mse, diverged=not good))
                    continue
                tq = np.arange(k + 1, dtype=float)
                res = measure(tr[:k + 1], wrapped_truth(A, tq, T_snap),
                              T_snap)
                res["err_inst_pred"] = predicted_err_inst(dl, N)
                rows.append(dict(arch=name, seed=seed, N=N, delta=dl,
                                 train_mse=mse, diverged=False, **res))
            dstr = f"{dl:+.2e}" if dl is not None else "diverged"
            print(f"  {name:12s} s={seed} mse={mse:.2e} delta={dstr:>11} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # reference: exact DMD, the deterministic classical baseline
    Am = dmd_fit(A)
    n = int(max(a.horizons) * T_snap)
    tr = np.empty((n + 1, a.modes)); x = A[0].copy(); tr[0] = x
    ok = True
    for k in range(n):
        x = Am @ x
        if not np.isfinite(x).all() or np.abs(x).max() > 1e8:
            ok = False; break
        tr[k + 1] = x
    if ok:
        k100 = int(100 * T_snap)
        Tm = period_from_signal(np.arange(k100 + 1) * dt, tr[:k100 + 1, 0])
        dl = (Tm - 2 * math.pi) / (2 * math.pi)
        for N in a.horizons:
            k = int(N * T_snap)
            tq = np.arange(k + 1, dtype=float)
            res = measure(tr[:k + 1], wrapped_truth(A, tq, T_snap), T_snap)
            res["err_inst_pred"] = predicted_err_inst(dl, N)
            rows.append(dict(arch="exact DMD", seed=-1, N=N, delta=dl,
                             train_mse=None, diverged=False, **res))
        print(f"  {'exact DMD':12s}        delta={dl:+.2e}")

    json.dump(dict(rows=rows, T_snap=T_snap, dt=dt, Re=a.Re,
                   modes=a.modes), open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
