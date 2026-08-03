"""
Tier-2 ground truth: 2D flow past a fixed circular cylinder at Re = 100.

Vortex shedding here is a genuine limit cycle in the Navier-Stokes phase space,
and the Strouhal number is known from experiment and DNS to three digits
(St = 0.164-0.167 at Re = 100; Williamson 1989 correlation gives 0.1643). That
makes it the first case where the period error delta of a learned surrogate can
be checked against literature rather than against another simulation.

D2Q9 lattice Boltzmann, BGK collision, halfway bounce-back on the cylinder.
Far-field Dirichlet on top/bottom, zero-gradient outflow.

Shedding period is measured the same way as in the Tier-0 oscillator work: a
Poincare section (here, upward zero-crossings of the transverse velocity at a
wake probe) with linear interpolation of the crossing times.
"""
import argparse, json, math, time
import numpy as np
import torch

# D2Q9
C = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.long)
OPP = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.long)
W = torch.tensor([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])


def equilibrium(rho, ux, uy, w, c):
    """f_eq_i = w_i rho (1 + 3 c.u + 4.5 (c.u)^2 - 1.5 u^2)"""
    cu = c[:, 0].view(-1, 1, 1) * ux + c[:, 1].view(-1, 1, 1) * uy
    usq = ux * ux + uy * uy
    return w.view(-1, 1, 1) * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq)


class Cylinder:
    def __init__(self, D=30, Re=100, u_in=0.1, nx_D=30, ny_D=20, xc_D=8,
                 device="cpu", dtype=torch.float32):
        self.D, self.Re, self.u_in = D, Re, u_in
        self.nx, self.ny = int(nx_D * D), int(ny_D * D)
        self.xc, self.yc = int(xc_D * D), self.ny // 2
        self.nu = u_in * D / Re
        self.tau = 3 * self.nu + 0.5
        self.dev, self.dt_ = device, dtype
        self.blockage = D / self.ny

        self.c = C.to(device)
        self.opp = OPP.to(device)
        self.w = W.to(device=device, dtype=dtype)

        y, x = torch.meshgrid(torch.arange(self.ny, device=device),
                              torch.arange(self.nx, device=device),
                              indexing="ij")
        x, y = x.T.to(dtype), y.T.to(dtype)          # -> [nx, ny]
        self.solid = ((x - self.xc) ** 2 + (y - self.yc) ** 2) <= (D / 2) ** 2
        self.fluid = ~self.solid

        rho = torch.ones(self.nx, self.ny, device=device, dtype=dtype)
        ux = torch.full_like(rho, u_in)
        uy = torch.zeros_like(rho)
        # solid nodes carry a rest-state distribution: zeroing them would make
        # rho = 0 there and u = 0/0 would poison the whole field
        ux = torch.where(self.solid, torch.zeros_like(ux), ux)
        self.f = equilibrium(rho, ux, uy, self.w, self.c)

        # probe on the centreline, 2D downstream: v(t) there oscillates at the
        # shedding frequency
        self.px, self.py = self.xc + 2 * D, self.yc

    def macro(self):
        rho = self.f.sum(0).clamp_min(1e-9)
        ux = (self.c[:, 0].view(-1, 1, 1).to(self.dt_) * self.f).sum(0) / rho
        uy = (self.c[:, 1].view(-1, 1, 1).to(self.dt_) * self.f).sum(0) / rho
        return rho, ux, uy

    def step(self, perturb=0.0):
        rho, ux, uy = self.macro()
        ux = torch.where(self.solid, torch.zeros_like(ux), ux)
        uy = torch.where(self.solid, torch.zeros_like(uy), uy)

        feq = equilibrium(rho, ux, uy, self.w, self.c)
        fpost = self.f - (self.f - feq) / self.tau
        # no collision inside the solid: those nodes only bounce back
        fpost[:, self.solid] = self.f[:, self.solid]

        # Momentum exchange (Ladd/Mei) summed over links that cross the surface:
        #   F = sum_links c_i [ f_i(x_f) + f_ibar(x_f) ]
        fx = fy = 0.0
        for i in range(1, 9):
            sx, sy = int(self.c[i, 0]), int(self.c[i, 1])
            nbr_solid = torch.roll(self.solid, shifts=(-sx, -sy), dims=(0, 1))
            link = self.fluid & nbr_solid
            if link.any():
                amt = float((fpost[i][link] + fpost[self.opp[i]][link]).sum())
                fx -= sx * amt          # sign: force on the body, +x = drag
                fy -= sy * amt

        # halfway bounce-back: reverse populations sitting on solid nodes
        fbb = fpost.clone()
        fbb[:, self.solid] = fpost[self.opp][:, self.solid]

        for i in range(9):
            sx, sy = int(self.c[i, 0]), int(self.c[i, 1])
            self.f[i] = torch.roll(fbb[i], shifts=(sx, sy), dims=(0, 1))

        # boundaries
        one = torch.ones(1, self.ny, device=self.dev, dtype=self.dt_)
        uin = torch.full_like(one, self.u_in)
        vin = torch.full_like(one, perturb * self.u_in)
        self.f[:, :1, :] = equilibrium(one, uin, vin, self.w, self.c)
        self.f[:, -1:, :] = self.f[:, -2:-1, :]                    # outflow
        onex = torch.ones(self.nx, 1, device=self.dev, dtype=self.dt_)
        uinx = torch.full_like(onex, self.u_in)
        zx = torch.zeros_like(onex)
        self.f[:, :, :1] = equilibrium(onex, uinx, zx, self.w, self.c)
        self.f[:, :, -1:] = equilibrium(onex, uinx, zx, self.w, self.c)
        return float(fx), float(fy)


def crossings(t, s):
    """Upward zero-crossings of a mean-subtracted signal, linearly interpolated."""
    s = np.asarray(s) - np.mean(s)
    out = []
    for i in range(len(s) - 1):
        if s[i] < 0 <= s[i + 1]:
            out.append(t[i] + (t[i + 1] - t[i]) * (-s[i] / (s[i + 1] - s[i])))
    return np.array(out)


def run(D=30, Re=100, u_in=0.1, steps=120000, warmup_frac=0.5, perturb_steps=2000,
        probe_every=5, device="cpu", snap_every=0, snap_crop=None, verbose=True):
    sim = Cylinder(D=D, Re=Re, u_in=u_in, device=device)
    if verbose:
        print(f"grid {sim.nx}x{sim.ny}  D={D}  tau={sim.tau:.4f}  "
              f"nu={sim.nu:.5f}  blockage={sim.blockage*100:.1f}%  "
              f"Ma={u_in*math.sqrt(3):.3f}", flush=True)

    tp, vp, cl, cd = [], [], [], []
    snaps, snap_t = [], []
    t0 = time.time()
    for n in range(steps):
        # brief inlet tilt to break symmetry and seed shedding
        pert = 0.05 if n < perturb_steps else 0.0
        fx, fy = sim.step(perturb=pert)
        if n % probe_every == 0:
            _, ux, uy = sim.macro()
            tp.append(n)
            vp.append(float(uy[sim.px, sim.py]))
            norm = 0.5 * u_in ** 2 * D
            cd.append(fx / norm)
            cl.append(fy / norm)
        if snap_every and n >= warmup_frac * steps and n % snap_every == 0:
            _, ux, uy = sim.macro()
            if snap_crop:
                x0, x1, y0, y1 = snap_crop
                snaps.append(torch.stack([ux[x0:x1, y0:y1],
                                          uy[x0:x1, y0:y1]]).cpu().numpy())
            else:
                snaps.append(torch.stack([ux, uy]).cpu().numpy())
            snap_t.append(n)
        if verbose and n and n % max(1, steps // 10) == 0:
            mlups = sim.nx * sim.ny * n / (time.time() - t0) / 1e6
            print(f"  step {n:>7}/{steps}  Cl={cl[-1]:+.4f}  Cd={cd[-1]:+.4f}  "
                  f"{mlups:.1f} MLUPS", flush=True)

    tp = np.array(tp, dtype=float)
    m = tp > warmup_frac * steps                       # discard transient
    cr = crossings(tp[m], np.array(vp)[m])
    res = dict(D=D, Re=Re, u_in=u_in, nx=sim.nx, ny=sim.ny, tau=sim.tau,
               blockage=sim.blockage, steps=steps, n_cycles=len(cr) - 1,
               secs=time.time() - t0)
    if len(cr) >= 4:
        per = np.diff(cr)
        T = float(per.mean())
        res.update(T_lattice=T, T_std=float(per.std()),
                   St=float(D / (u_in * T)),
                   Cd_mean=float(np.mean(np.array(cd)[m])),
                   Cl_amp=float(np.abs(np.array(cl)[m]).max()))
    return res, dict(t=tp, v=np.array(vp), cl=np.array(cl), cd=np.array(cd)), \
        (np.array(snaps) if snaps else None, np.array(snap_t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=30)
    ap.add_argument("--Re", type=float, default=100)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--snap-every", type=int, default=0)
    ap.add_argument("--out", default="cyl.json")
    ap.add_argument("--save-signal", default=None)
    ap.add_argument("--save-snaps", default=None)
    a = ap.parse_args()

    crop = None
    if a.snap_every:
        D = a.D
        crop = (6 * D, 22 * D, 6 * D, 14 * D)      # wake window around cylinder
    res, sig, (snaps, snap_t) = run(D=a.D, Re=a.Re, steps=a.steps,
                                    device=a.device, snap_every=a.snap_every,
                                    snap_crop=crop)
    print("\n" + json.dumps(res, indent=1))
    if "St" in res:
        print(f"\nSt = {res['St']:.5f}   (Williamson 1989 correlation at "
              f"Re=100: 0.1643)")
        print(f"Cd = {res['Cd_mean']:.4f}  (literature 1.32-1.35)")
    json.dump(res, open(a.out, "w"), indent=1)
    if a.save_signal:
        np.savez_compressed(a.save_signal, **sig)
    if a.save_snaps and snaps is not None:
        np.savez_compressed(a.save_snaps, snaps=snaps, t=snap_t,
                            T_lattice=res.get("T_lattice", np.nan))
        print(f"saved {snaps.shape} snapshots")


if __name__ == "__main__":
    main()
