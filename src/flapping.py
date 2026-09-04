"""
Flapping filament in uniform flow: the coupled flexible-FSI case.

This is the system the whole build exists to produce. Every other case in the
study takes its period from the flow -- the wake sheds at a frequency set by
Reynolds number, and a surrogate's only job is to reproduce a number it was
shown. Here the structure has a frequency of its own, set by bending rigidity
and mass ratio, and the limit cycle that appears is a property of the coupling
rather than of either side. That is the untested class: if the phase-drift
argument holds when the period is not inherited, it is not an artefact of
problems whose frequency was handed over.

Non-dimensional parameters, and how they land in lattice units with rho_f = 1:

    Re    = U L / nu                     ->  nu   = u_in * L / Re
    KB*   = KB / (rho_f U^2 L^3)         ->  KB   = KB* * u_in^2 * L^3
    rho*  = rho_s A / (rho_f L^2)        ->  rho1 = rho* * L^2

Coupling is staggered: the fluid is driven to the filament's current velocity,
and the filament is advanced with the reaction. That is the cheap scheme, and
it is the one that fails first -- a partitioned explicit coupling goes unstable
when the displaced fluid mass is comparable to the structural mass, which is
precisely the regime a flapping filament lives in. `n_couple` sub-iterations
turn it into a fixed-point solve on the interface force for when it does. The
default is 1 because the added-mass limit is a property of the parameters, not
of the method, and paying for sub-iteration everywhere would be waste; the
`--couple-scan` entry point is there to find where the boundary actually is
rather than guess it.

Marker spacing is held near one lattice unit. Wider than that and fluid leaks
through the filament; much narrower and the spread force develops a checkerboard
the delta function cannot smooth. That ties N to L, which is why both are set
together.
"""
import argparse, json, math, time
import numpy as np

from filament import Filament
from iblbm import IBLBM


class FlappingFilament:
    def __init__(self, L=40, Re=200, u_in=0.08, KB_star=1e-3, rho_star=1.0,
                 N=None, nx_L=20, ny_L=10, x0_L=4, root_type="clamped",
                 n_ib=5, n_couple=1, relax=1.0):
        self.L, self.Re, self.u_in = L, Re, u_in
        self.KB_star, self.rho_star = KB_star, rho_star
        self.N = int(L) if N is None else N          # ds ~ 1 lattice unit
        self.nx, self.ny = int(nx_L * L), int(ny_L * L)
        self.x0, self.y0 = int(x0_L * L), self.ny // 2
        self.n_ib, self.n_couple, self.relax = n_ib, n_couple, relax

        nu = u_in * L / Re
        self.flow = IBLBM(self.nx, self.ny, nu=nu, u_in=u_in)
        self.fil = Filament(N=self.N, L=float(L),
                            rho1=rho_star * L ** 2,
                            KB=KB_star * u_in ** 2 * L ** 3,
                            dt=1.0, root=(self.x0, self.y0),
                            tangent=(1.0, 0.0), root_type=root_type)
        self.tau = self.flow.tau
        # probe two filament-lengths downstream of the tip, on the centreline
        self.px, self.py = min(self.nx - 2, self.x0 + 3 * L), self.y0

    # ------------------------------------------------------------------ step
    def _interface_force(self):
        """Lagrangian force driving the fluid to the filament's velocity."""
        X = self.fil.X
        U = (self.fil.X - self.fil.Xprev) / self.fil.dt
        return self.flow.ib_force(X, U, self.fil.ds, n_iter=self.n_ib)

    def step(self):
        if self.n_couple <= 1:
            gx, gy, F, st = self._interface_force()
            self.flow.step(gx, gy)
            self.fil.step(F=-F[1:])
            return F, st

        # strong coupling: fixed point on the interface force, rolling the
        # filament back each pass so only the converged force is ever applied
        Xs, Xp = self.fil.X.copy(), self.fil.Xprev.copy()
        Fold = None
        for k in range(self.n_couple):
            self.fil.X, self.fil.Xprev = Xs.copy(), Xp.copy()
            gx, gy, F, st = self._interface_force()
            if Fold is not None:
                F = Fold + self.relax * (F - Fold)
            self.fil.step(F=-F[1:])
            if Fold is not None and np.abs(F - Fold).max() < 1e-10:
                Fold = F
                break
            Fold = F
        self.flow.step(gx, gy)
        return Fold, st

    # ------------------------------------------------------------ diagnostics
    def tip(self):
        return self.fil.X[-1].copy()

    def state(self):
        return dict(tip_y=float(self.fil.X[-1, 1] - self.y0),
                    tip_x=float(self.fil.X[-1, 0] - self.x0),
                    stretch=self.fil.max_stretch())


def run(L=40, Re=200, u_in=0.08, KB_star=1e-3, rho_star=1.0, steps=60000,
        n_ib=5, n_couple=1, kick=0.02, kick_steps=400, probe_every=5,
        snap_every=0, warmup_frac=0.5, verbose=True, **kw):
    sim = FlappingFilament(L=L, Re=Re, u_in=u_in, KB_star=KB_star,
                           rho_star=rho_star, n_ib=n_ib, n_couple=n_couple,
                           **kw)
    if verbose:
        print(f"flapping filament: grid {sim.nx}x{sim.ny}  L={L}  N={sim.N}  "
              f"ds={sim.fil.ds:.3f}  tau={sim.tau:.4f}  "
              f"Re={Re} KB*={KB_star:g} rho*={rho_star:g}", flush=True)

    ts, tipy, stretch, slip = [], [], [], []
    snaps, fil_snaps, snap_t = [], [], []
    t0 = time.time()
    diverged = False
    for n in range(steps):
        # transverse kick to break the unstable symmetric state
        if n < kick_steps:
            f = np.zeros((sim.N, 2))
            f[:, 1] = kick * u_in ** 2 * np.linspace(0, 1, sim.N)
            gx, gy, F, st = sim._interface_force()
            sim.flow.step(gx, gy)
            sim.fil.step(F=-F[1:] + f)
        else:
            F, st = sim.step()

        if not np.isfinite(sim.fil.X).all():
            diverged = True
            if verbose:
                print(f"  DIVERGED at step {n}", flush=True)
            break

        if n % probe_every == 0:
            ts.append(n)
            tipy.append(float(sim.fil.X[-1, 1] - sim.y0))
            stretch.append(sim.fil.max_stretch())
        if snap_every and n >= warmup_frac * steps and n % snap_every == 0:
            _, ux, uy = sim.flow.macro_unforced()
            x0, x1 = sim.x0 - L, min(sim.nx, sim.x0 + 6 * L)
            y0, y1 = sim.y0 - 2 * L, sim.y0 + 2 * L
            snaps.append(np.stack([ux[x0:x1, y0:y1], uy[x0:x1, y0:y1]]))
            fil_snaps.append(sim.fil.X.copy())
            snap_t.append(n)
        if verbose and n and n % max(1, steps // 10) == 0:
            print(f"  step {n:>7}/{steps}  tip_y={tipy[-1]:+.4f}  "
                  f"stretch={stretch[-1]:.1e}  "
                  f"{sim.nx*sim.ny*n/(time.time()-t0)/1e6:.2f} MLUPS",
                  flush=True)

    ts = np.array(ts, float)
    tipy = np.array(tipy)
    res = dict(L=L, Re=Re, u_in=u_in, KB_star=KB_star, rho_star=rho_star,
               N=sim.N, nx=sim.nx, ny=sim.ny, tau=sim.tau, steps=steps,
               n_ib=n_ib, n_couple=n_couple, diverged=diverged,
               max_stretch=float(np.max(stretch)) if stretch else float("nan"),
               secs=time.time() - t0)

    m = ts > warmup_frac * ts.max() if len(ts) else np.array([], bool)
    if not diverged and m.sum() > 50:
        amp = float(np.abs(tipy[m] - tipy[m].mean()).max())
        res["tip_amp"] = amp
        res["flapping"] = bool(amp > 1e-3 * L)
        if res["flapping"]:
            from period import period_from_signal
            tq, vq = ts[m], tipy[m]
            T = period_from_signal(tq, vq)
            h = len(tq) // 2
            T1 = period_from_signal(tq[:h], vq[:h])
            T2 = period_from_signal(tq[h:], vq[h:])
            s = vq - vq.mean()
            cr = np.array([tq[i] + (tq[i+1]-tq[i]) * (-s[i]/(s[i+1]-s[i]))
                           for i in range(len(s)-1) if s[i] < 0 <= s[i+1]])
            per = np.diff(cr) if len(cr) > 2 else np.array([np.nan])
            res.update(T_lattice=float(T),
                       St=float(L / (u_in * T)),
                       n_cycles=int(len(per)),
                       T_jitter=float(np.std(per) / np.mean(per)),
                       split_half_rel=float(abs(T1 - T2) / T1))
    return res, dict(t=ts, tip_y=tipy), \
        (np.array(snaps) if snaps else None, np.array(fil_snaps) if fil_snaps
         else None, np.array(snap_t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=40)
    ap.add_argument("--Re", type=float, default=200)
    ap.add_argument("--u-in", type=float, default=0.08)
    ap.add_argument("--KB", type=float, default=1e-3, dest="KB_star")
    ap.add_argument("--rho", type=float, default=1.0, dest="rho_star")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--n-ib", type=int, default=5)
    ap.add_argument("--n-couple", type=int, default=1)
    ap.add_argument("--snap-every", type=int, default=0)
    ap.add_argument("--out", default="flap.json")
    ap.add_argument("--save-signal", default=None)
    ap.add_argument("--save-snaps", default=None)
    a = ap.parse_args()
    res, sig, (snaps, fils, snap_t) = run(
        L=a.L, Re=a.Re, u_in=a.u_in, KB_star=a.KB_star, rho_star=a.rho_star,
        steps=a.steps, n_ib=a.n_ib, n_couple=a.n_couple,
        snap_every=a.snap_every)
    print("\n" + json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)
    if a.save_signal:
        np.savez_compressed(a.save_signal, **sig)
    if a.save_snaps and snaps is not None:
        np.savez_compressed(a.save_snaps, snaps=snaps, filament=fils,
                            t=snap_t, T_lattice=res.get("T_lattice", np.nan),
                            L=a.L, Re=a.Re, u_in=a.u_in,
                            KB_star=a.KB_star, rho_star=a.rho_star)
        print(f"saved {snaps.shape} flow snapshots + {fils.shape} filament")


if __name__ == "__main__":
    main()
