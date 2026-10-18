"""
A re-entrant spiral wave in cardiac tissue, as a limit cycle outside fluids.

Every system the study had measured was a flow or a generic oscillator, which
leaves the reading that the period error is a property of fluid surrogates. The
argument uses nothing about fluids -- only an autonomous system with a stable
periodic attractor -- so it should hold for any rhythm a surrogate is built
for. A rotating spiral in excitable cardiac tissue is the standard model of
re-entrant arrhythmia, and surrogates of cardiac electrophysiology are an
active application of learned operators. In the lab frame a rigidly rotating
spiral is a periodic orbit of the PDE: one neutral direction (rotation phase),
the rest contracting.

Model: Aliev & Panfilov (1996), the two-variable canine-myocardium model,

    du/dt = D lap(u) - k u (u - a)(u - 1) - u v
    dv/dt = eps(u, v) (-v - k u (u - a - 1)),   eps = e0 + mu1 v / (u + mu2)

with the published parameters k = 8, a = 0.15, e0 = 0.002, mu1 = 0.2,
mu2 = 0.3, D = 1 (dimensionless; t[ms] = 12.9 t). The paper reports that these
support a stationary rotating vortex in two dimensions; that is checked here
rather than assumed, with the same split-half test used on Navier-Stokes.

Numerics: forward Euler, five-point Laplacian, no-flux boundaries by edge
padding, on an N x N grid of spacing dx, snapshots block-averaged 2 x 2 onto the
grid the surrogate sees. The spiral is started by cutting a planar wave: excite
a strip at the left edge, and when the front reaches 0.8 of the width reset the
upper half to rest, leaving a free wave end that curls into a vortex. The tip
forms near the back of the broken wave, so cutting at mid-domain leaves the core
close to the left wall; at 0.8 it settles near the centre (checked visually).

The period is not known in advance, so it is measured from a probe during the
transient and the snapshot interval set to give ~40 snapshots per period, the
same resolution as the Navier-Stokes case.

  python src/cardiac.py --out data/cardiac/ap_default
"""
import argparse, json, math, os, time
import numpy as np

from period import period_from_signal


def lap(u, dx):
    p = np.pad(u, 1, mode="edge")                  # zero normal flux
    return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * u) / dx ** 2


def step(u, v, P, dt):
    eps = P["e0"] + P["mu1"] * v / (u + P["mu2"])
    du = P["D"] * lap(u, P["dx"]) - P["k"] * u * (u - P["a"]) * (u - 1) - u * v
    dv = eps * (-v - P["k"] * u * (u - P["a"] - 1))
    return u + dt * du, v + dt * dv


def coarsen(f):
    n = f.shape[0] // 2
    return f.reshape(n, 2, n, 2).mean((1, 3))


def pod_coef1(snaps, r=8):
    X = snaps.reshape(len(snaps), -1).astype(np.float64)
    Xc = X - X.mean(0)
    w, V = np.linalg.eigh(Xc @ Xc.T)
    w, V = np.clip(w[::-1], 0, None), V[:, ::-1]
    coefs = V[:, :r] * np.sqrt(w[:r])               # temporal coefficients
    return coefs, w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256, help="simulation grid")
    ap.add_argument("--dx", type=float, default=0.4)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--a", type=float, default=0.15)
    ap.add_argument("--e0", type=float, default=0.002)
    ap.add_argument("--mu1", type=float, default=0.2)
    ap.add_argument("--mu2", type=float, default=0.3)
    ap.add_argument("--D", type=float, default=1.0)
    ap.add_argument("--transient", type=float, default=600.0,
                    help="time units discarded before recording")
    ap.add_argument("--periods", type=float, default=60.0, help="periods recorded")
    ap.add_argument("--per-period", type=int, default=40, help="snapshots per period")
    ap.add_argument("--cut-x", type=float, default=0.8,
                    help="cut the wave when its front reaches this fraction of the "
                         "domain; 0.8 leaves the vortex core near the centre")
    ap.add_argument("--out", required=True, help="path stem; writes _snaps.npz and .json")
    a = ap.parse_args()
    P = dict(k=a.k, a=a.a, e0=a.e0, mu1=a.mu1, mu2=a.mu2, D=a.D, dx=a.dx)
    N, dt = a.N, a.dt
    assert dt <= a.dx ** 2 / (4 * a.D), "explicit diffusion step unstable"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time()

    u = np.zeros((N, N)); v = np.zeros((N, N))
    u[:, : max(2, N // 20)] = 1.0                  # planar wave from the left
    t, cut = 0.0, False
    probe_ij = (N // 4, N // 4)
    probe_t, probe_u = [], []
    n_trans = int(round(a.transient / dt))
    for n in range(n_trans):
        u, v = step(u, v, P, dt); t += dt
        if not cut and u[N // 4, int(a.cut_x * N)] > 0.5:
            u[N // 2:, :] = 0.0; v[N // 2:, :] = 0.0
            cut = True
            print(f"  wave cut at t = {t:.1f}", flush=True)
        if n >= n_trans // 2:
            probe_t.append(t); probe_u.append(u[probe_ij])
    if not cut:
        raise SystemExit("front never reached the cut position; no spiral was initiated")
    if not np.isfinite(u).all():
        raise SystemExit("solution blew up during the transient")
    T0 = period_from_signal(np.array(probe_t), np.array(probe_u))
    stride = max(1, int(round(T0 / (a.per_period * dt))))
    dts = stride * dt
    n_rec = int(round(a.periods * T0 / dts))
    print(f"  probe period {T0:.4f} t.u. ({T0 * 12.9:.0f} ms); snapshot every "
          f"{stride} steps = {dts:.3f} t.u. -> {T0 / dts:.2f} per period; "
          f"{n_rec} snapshots", flush=True)

    H = N // 2
    snaps = np.empty((n_rec, 2, H, H), np.float32)
    ts = np.empty(n_rec)
    for j in range(n_rec):
        for _ in range(stride):
            u, v = step(u, v, P, dt); t += dt
        snaps[j, 0], snaps[j, 1] = coarsen(u), coarsen(v)
        ts[j] = t
    if not np.isfinite(snaps).all():
        raise SystemExit("solution blew up while recording")

    coefs, energy = pod_coef1(snaps)
    T = period_from_signal(ts, coefs[:, 0])
    h = n_rec // 2
    Ta = period_from_signal(ts[:h], coefs[:h, 0])
    Tb = period_from_signal(ts[h:], coefs[h:, 0])
    split = abs(Ta - Tb) / T
    # an excited fraction that never changes would mean the spiral died out
    active = float(np.mean(snaps[:, 0] > 0.5))
    regime = ("periodic" if split < 1e-7 else "not well-posed") if active > 0.01 else "quiescent"
    summ = dict(params=P, N=N, dt=dt, grid=[H, H], T=T, T_probe=T0,
                T_snap=T / dts, dt_snap=dts, snapshots=n_rec,
                periods=n_rec * dts / T, split_half=split, regime=regime,
                pod8_energy=float(energy[:8].sum()), pod2_energy=float(energy[:2].sum()),
                active_fraction=active, u_range=[float(snaps[:, 0].min()), float(snaps[:, 0].max())],
                v_range=[float(snaps[:, 1].min()), float(snaps[:, 1].max())],
                secs=time.time() - t0)
    np.savez_compressed(a.out + "_snaps.npz", snaps=snaps, t=ts)
    json.dump(summ, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
