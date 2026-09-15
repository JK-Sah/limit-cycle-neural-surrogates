"""
2D Navier-Stokes on a torus, in the exact setting of the Fourier neural operator
benchmark -- and the question of whether delta is defined there at all.

The FNO paper's Navier-Stokes case solves the vorticity form

    dw/dt + u . grad w = nu * lap w + f,    div u = 0,    w = curl u
    f(x) = 0.1 ( sin(2 pi (x1 + x2)) + cos(2 pi (x1 + x2)) )

on [0, 1]^2 at nu = 1e-3, 1e-4, 1e-5. Those viscosities correspond to roughly
Re = 20, 200 and 2000. Two-dimensional Kolmogorov-type flow has a Hopf
bifurcation near Re = 32 and sustained chaos above Re ~ 103, with periodic
windows between. The benchmark's three settings therefore sit below the first
bifurcation and above the onset of chaos: they bracket the periodic windows
without landing in one.

That matters because delta is only defined on a limit cycle. Quoting a period
error for a chaotic trajectory is exactly the error section 9 exists to prevent,
so this module does not assume a periodic attractor -- it locates one. The
viscosity sweep applies the split-half well-posedness test at each nu and
reports where a single-period fit is meaningful, which is the same test a
practitioner would run before trusting any delta.

Integration is by integrating-factor RK4 in Fourier space with 2/3 dealiasing:
the viscous term is handled exactly, so the only time-stepping error is in the
advection, which matters because the period has to be resolved far better than
the amplitude.
"""
import argparse, json, math, os, time
import numpy as np


class Kolmogorov:
    def __init__(self, N=128, nu=1e-3, L=1.0, forcing="fno"):
        self.N, self.nu, self.L = N, nu, L
        k1 = 2 * math.pi / L * np.fft.fftfreq(N, 1.0 / N)
        self.kx = k1[:, None] * np.ones((1, N // 2 + 1))
        self.ky = (2 * math.pi / L * np.fft.rfftfreq(N, 1.0 / N))[None, :] * np.ones((N, 1))
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.k2inv = np.where(self.k2 == 0, 0.0, 1.0 / np.where(self.k2 == 0, 1.0, self.k2))
        # 2/3 dealiasing
        kmax = (2.0 / 3.0) * (math.pi * N / L)
        self.mask = (np.abs(self.kx) < kmax) & (np.abs(self.ky) < kmax)

        x = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="ij")
        if forcing == "fno":
            f = 0.1 * (np.sin(2 * math.pi * (X + Y)) + np.cos(2 * math.pi * (X + Y)))
        else:                                    # classical Kolmogorov
            f = -4.0 * np.cos(4.0 * Y)
        self.fh = np.fft.rfft2(f)

    def rhs(self, wh):
        """-u.grad(w) in spectral space, dealiased, plus forcing."""
        psih = wh * self.k2inv
        uh = 1j * self.ky * psih
        vh = -1j * self.kx * psih
        u = np.fft.irfft2(uh, s=(self.N, self.N))
        v = np.fft.irfft2(vh, s=(self.N, self.N))
        wx = np.fft.irfft2(1j * self.kx * wh, s=(self.N, self.N))
        wy = np.fft.irfft2(1j * self.ky * wh, s=(self.N, self.N))
        nl = np.fft.rfft2(u * wx + v * wy)
        return (-nl + self.fh) * self.mask

    def step(self, wh, dt):
        """Integrating-factor RK4: viscosity exact, advection fourth order."""
        E = np.exp(-self.nu * self.k2 * dt)
        E2 = np.exp(-self.nu * self.k2 * dt / 2)
        a = self.rhs(wh)
        b = self.rhs(E2 * (wh + 0.5 * dt * a))
        c = self.rhs(E2 * wh + 0.5 * dt * b)
        d = self.rhs(E * wh + dt * E2 * c)
        return (E * wh + dt / 6.0 * (E * a + 2 * E2 * b + 2 * E2 * c + d))


def run(nu=1e-3, N=128, dt=2e-3, T_transient=200.0, T_record=400.0,
        probe_every=20, seed=0, forcing="fno", verbose=True,
        snap_every=0, snaps_out=None):
    sim = Kolmogorov(N=N, nu=nu, forcing=forcing)
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(N, N)) * 0.01
    wh = np.fft.rfft2(w) * sim.mask

    n_tr = int(T_transient / dt)
    for _ in range(n_tr):
        wh = sim.step(wh, dt)

    n_rec = int(T_record / dt)
    ts, sig, ens = [], [], []
    snaps, snap_t = [], []
    t0 = time.time()
    for n in range(n_rec):
        wh = sim.step(wh, dt)
        if n % probe_every == 0:
            w = np.fft.irfft2(wh, s=(N, N))
            ts.append(n * dt)
            # a scalar observable with the symmetry of the forcing
            sig.append(float(w[N // 4, N // 4]))
            ens.append(float(0.5 * np.mean(w * w)))
        if snap_every and n % snap_every == 0:
            snaps.append(np.fft.irfft2(wh, s=(N, N)).astype(np.float32))
            snap_t.append(n * dt)
        if verbose and n and n % max(1, n_rec // 5) == 0:
            print(f"    step {n}/{n_rec}  enstrophy={ens[-1]:.6f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    if snaps_out and snaps:
        np.savez_compressed(snaps_out, snaps=np.array(snaps),
                            t=np.array(snap_t), nu=nu, N=N, dt=dt,
                            forcing=forcing)
        if verbose:
            print(f"    saved {len(snaps)} snapshots -> {snaps_out}", flush=True)
    return np.array(ts), np.array(sig), np.array(ens)


def classify(ts, sig, verbose=True):
    """Split-half well-posedness test: is a single-period fit meaningful here?"""
    import sys
    sys.path.insert(0, "src")
    from period import period_from_signal
    m = ts > ts[0]
    t, v = ts[m], sig[m]
    T = period_from_signal(t, v)
    h = len(t) // 2
    T1 = period_from_signal(t[:h], v[:h])
    T2 = period_from_signal(t[h:], v[h:])
    inst = abs(T1 - T2) / T
    amp = float(np.ptp(v) / 2)
    # a steady state has no oscillation to fit
    steady = amp < 1e-8 * max(1.0, float(np.abs(v).mean()))
    regime = ("steady" if steady else
              "periodic" if inst < 1e-7 else
              "quasi-periodic or chaotic")
    if verbose:
        print(f"    T={T:.6f}  split-half={inst:.2e}  amp={amp:.3e}  -> {regime}")
    return dict(T=float(T), split_half=float(inst), amp=amp, regime=regime,
                periods_in_record=float((t[-1] - t[0]) / T) if T else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, nargs="+",
                    default=[1e-3, 5e-4, 3e-4, 2e-4, 1.5e-4, 1e-4])
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--dt", type=float, default=2e-3)
    ap.add_argument("--transient", type=float, default=200.0)
    ap.add_argument("--record", type=float, default=400.0)
    ap.add_argument("--forcing", default="fno", choices=["fno", "kolmogorov"])
    ap.add_argument("--snap-every", type=int, default=0,
                    help="save a vorticity field every N steps")
    ap.add_argument("--save-snaps", default=None)
    ap.add_argument("--out", default="kolmogorov_sweep.json")
    a = ap.parse_args()

    rows = []
    for nu in a.nu:
        print(f"nu = {nu:g}  (Re ~ {1/nu:.0f})", flush=True)
        ts, sig, ens = run(nu=nu, N=a.N, dt=a.dt, T_transient=a.transient,
                           T_record=a.record, forcing=a.forcing,
                           snap_every=a.snap_every, snaps_out=a.save_snaps)
        r = classify(ts, sig)
        r.update(nu=nu, N=a.N, dt=a.dt, enstrophy=float(np.mean(ens)),
                 enstrophy_std=float(np.std(ens)))
        rows.append(r)
        json.dump(rows, open(a.out, "w"), indent=1)
    print()
    print("%10s %12s %14s %12s %s" % ("nu", "T", "split-half", "amp", "regime"))
    for r in rows:
        print("%10.1e %12.5f %14.2e %12.3e %s" %
              (r["nu"], r["T"], r["split_half"], r["amp"], r["regime"]))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
