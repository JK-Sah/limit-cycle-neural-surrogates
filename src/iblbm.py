"""
D2Q9 lattice Boltzmann with immersed-boundary forcing.

This is the flow half of the flexible-FSI case. It differs from
`cylinder_lbm.py` in one respect only: the body is carried as Lagrangian
markers and enters through a Guo force term rather than as a bounce-back mask.
Everything else -- lattice, BGK collision, inlet, outflow, far-field -- is kept
deliberately identical, because the first validation of this file is that it
reproduces the Strouhal number the bounce-back solver already produces on the
same problem. If the two disagree on a rigid cylinder, the coupling is wrong,
and no result on a moving filament would be worth reading.

Running the rigid-cylinder case through the immersed-boundary path is therefore
not a formality. It is the only place in this build where the answer is already
known independently.
"""
import argparse, json, math, time
import numpy as np

from ibm import Stencil, guo_source, multi_direct_force

C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
              [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=int)
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# The populations are carried in single precision, matching cylinder_lbm.py.
# This is not a corner cut: the period is extracted by a Fourier fit over tens
# of cycles, which averages the round-off down far below the 1e-6 jitter the
# delta measurements need -- the bounce-back solver reaches 5e-7 the same way.
# Doubling the memory traffic to carry digits the estimator discards would cost
# a factor of two on a kernel that is bandwidth bound.
DT = np.float32


def equilibrium(rho, ux, uy):
    cu = C[:, 0][:, None, None] * ux + C[:, 1][:, None, None] * uy
    usq = ux * ux + uy * uy
    out = W[:, None, None].astype(rho.dtype) * rho * \
        (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq)
    return out.astype(rho.dtype, copy=False)


class IBLBM:
    def __init__(self, nx, ny, nu, u_in, rho0=1.0, dtype=DT):
        self.nx, self.ny, self.u_in = nx, ny, u_in
        self.nu = nu
        self.tau = 3 * nu + 0.5
        self.dtype = dtype
        rho = np.full((nx, ny), rho0, dtype=dtype)
        ux = np.full((nx, ny), u_in, dtype=dtype)
        uy = np.zeros((nx, ny), dtype=dtype)
        self.f = equilibrium(rho, ux, uy)
        self._buf = np.empty_like(self.f)

    def macro_unforced(self):
        """rho and the pre-force velocity sum(c f)/rho.

        Written as explicit adds over the nine populations rather than a
        broadcast multiply-and-reduce: the lattice velocities are 0 and +-1, so
        the multiply is wasted work and the broadcast allocates a [9, nx, ny]
        temporary per component on a kernel that is already bandwidth bound.
        """
        f = self.f
        rho = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8]
        np.maximum(rho, self.dtype(1e-9), out=rho)
        ux = (f[1] - f[3]) + (f[5] - f[6]) - (f[7] - f[8])
        uy = (f[2] - f[4]) + (f[5] + f[6]) - (f[7] + f[8])
        return rho, ux / rho, uy / rho

    def ib_force(self, X, u_target, ds, n_iter=5):
        """Multi-direct forcing. Returns (gx, gy) Eulerian and F on the markers.

        The marker force accumulated over the correction passes is what the
        structure feels, negated. Interpolation and spreading use the same
        weights, so momentum removed from the body equals momentum given to the
        fluid to machine precision.
        """
        rho, ux, uy = self.macro_unforced()
        st = Stencil(X, self.nx, self.ny)
        gx = np.zeros((self.nx, self.ny))
        gy = np.zeros((self.nx, self.ny))
        Fx = np.zeros(len(X))
        Fy = np.zeros(len(X))
        for _ in range(n_iter):
            # velocity the collision will actually see, given the force so far
            ex = ux + 0.5 * gx / rho
            ey = uy + 0.5 * gy / rho
            dFx = multi_direct_force(st.interpolate(ex), u_target[:, 0])
            dFy = multi_direct_force(st.interpolate(ey), u_target[:, 1])
            gx += st.spread(dFx, self.nx, self.ny, ds)
            gy += st.spread(dFy, self.nx, self.ny, ds)
            Fx += dFx
            Fy += dFy
        return gx, gy, np.stack([Fx, Fy], axis=1), st

    def step(self, gx, gy):
        rho, ux, uy = self.macro_unforced()
        ux = ux + 0.5 * gx / rho
        uy = uy + 0.5 * gy / rho
        feq = equilibrium(rho, ux, uy)
        S = guo_source(ux, uy, gx, gy, C, W, self.tau)
        fpost = self.f - (self.f - feq) / self.tau + S

        for i in range(9):
            self.f[i] = np.roll(fpost[i], shift=(C[i, 0], C[i, 1]), axis=(0, 1))

        one = np.ones((1, self.ny))
        self.f[:, :1, :] = equilibrium(one, np.full_like(one, self.u_in),
                                       np.zeros_like(one))
        self.f[:, -1:, :] = self.f[:, -2:-1, :]
        onex = np.ones((self.nx, 1))
        self.f[:, :, :1] = equilibrium(onex, np.full_like(onex, self.u_in),
                                       np.zeros_like(onex))
        self.f[:, :, -1:] = equilibrium(onex, np.full_like(onex, self.u_in),
                                        np.zeros_like(onex))
        return rho, ux, uy


# ------------------------------------------------------- rigid-cylinder check
def cylinder_markers(D, xc, yc):
    """Markers on a circle at roughly unit arc spacing, plus that spacing."""
    n = max(8, int(round(math.pi * D)))
    th = np.arange(n) * 2 * math.pi / n
    X = np.stack([xc + 0.5 * D * np.cos(th), yc + 0.5 * D * np.sin(th)], 1)
    return X, math.pi * D / n


def run_cylinder(D=24, Re=100, u_in=0.1, nx_D=30, ny_D=20, xc_D=8,
                 steps=80000, probe_every=5, n_iter=5, verbose=True):
    nx, ny = int(nx_D * D), int(ny_D * D)
    xc, yc = int(xc_D * D), ny // 2
    sim = IBLBM(nx, ny, nu=u_in * D / Re, u_in=u_in)
    X, ds = cylinder_markers(D, xc, yc)
    u_target = np.zeros_like(X)
    px, py = xc + 2 * D, yc

    if verbose:
        print(f"IB-LBM cylinder: grid {nx}x{ny} D={D} tau={sim.tau:.4f} "
              f"markers={len(X)} ds={ds:.3f}", flush=True)

    tp, vp, slip = [], [], []
    t0 = time.time()
    for n in range(steps):
        gx, gy, F, st = sim.ib_force(X, u_target, ds, n_iter=n_iter)
        # symmetry-breaking nudge, same device as the bounce-back solver
        if n < 2000:
            gy += 0.0
            sim.f[:, :1, :] = equilibrium(np.ones((1, ny)),
                                          np.full((1, ny), u_in),
                                          np.full((1, ny), 0.05 * u_in))
        rho, ux, uy = sim.step(gx, gy)
        if n % probe_every == 0:
            tp.append(n)
            vp.append(float(uy[px, py]))
            slip.append(float(np.hypot(st.interpolate(ux),
                                       st.interpolate(uy)).max()))
        if verbose and n and n % max(1, steps // 10) == 0:
            print(f"  step {n:>7}/{steps}  v_probe={vp[-1]:+.5f}  "
                  f"max slip={slip[-1]:.2e}  "
                  f"{nx*ny*n/(time.time()-t0)/1e6:.1f} MLUPS", flush=True)

    tp = np.array(tp, float)
    vp = np.array(vp)
    m = tp > 0.5 * tp.max()
    tq, vq = tp[m], vp[m]
    from period import period_from_signal
    T = period_from_signal(tq, vq)
    St = D / (u_in * T)

    # Cycle-to-cycle jitter and the split-half refit. These are the numbers that
    # decide whether this data can carry a delta measurement at all: the whole
    # study resolves period errors down to 1e-5, so a reference period that is
    # itself uncertain at that level would measure nothing. The split-half test
    # is the same well-posedness check the quasi-periodic section applies before
    # any delta is quoted.
    s = vq - vq.mean()
    cr = np.array([tq[i] + (tq[i+1] - tq[i]) * (-s[i] / (s[i+1] - s[i]))
                   for i in range(len(s) - 1) if s[i] < 0 <= s[i+1]])
    per = np.diff(cr) if len(cr) > 2 else np.array([np.nan])
    h = len(tq) // 2
    T1 = period_from_signal(tq[:h], vq[:h])
    T2 = period_from_signal(tq[h:], vq[h:])

    return dict(D=D, Re=Re, nx=nx, ny=ny, tau=sim.tau, n_markers=len(X),
                ds=ds, n_iter=n_iter, steps=steps, T_lattice=float(T),
                St=float(St), max_slip=float(np.max(slip[len(slip)//2:])),
                n_cycles=int(len(per)),
                T_jitter=float(np.std(per) / np.mean(per)),
                split_half_rel=float(abs(T1 - T2) / T1),
                secs=time.time() - t0), dict(t=tp, v=vp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=24)
    ap.add_argument("--Re", type=float, default=100)
    ap.add_argument("--u-in", type=float, default=0.1)
    ap.add_argument("--nx-D", type=int, default=30)
    ap.add_argument("--ny-D", type=int, default=20)
    ap.add_argument("--xc-D", type=int, default=8)
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--n-iter", type=int, default=2)
    ap.add_argument("--out", default="ib_cyl.json")
    ap.add_argument("--save-signal", default=None)
    a = ap.parse_args()
    res, sig = run_cylinder(D=a.D, Re=a.Re, u_in=a.u_in, nx_D=a.nx_D,
                            ny_D=a.ny_D, xc_D=a.xc_D, steps=a.steps,
                            n_iter=a.n_iter)
    print("\n" + json.dumps(res, indent=1))
    print(f"\nSt = {res['St']:.5f}   bounce-back solver at the same box: 0.16851"
          f"   Williamson unconfined: 0.1643")
    json.dump(res, open(a.out, "w"), indent=1)
    if a.save_signal:
        np.savez_compressed(a.save_signal, **sig)


if __name__ == "__main__":
    main()
