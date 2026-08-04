"""
Tier-2: latent surrogate of the cylinder wake, and its period error delta.

The Tier-0 result was obtained with the phase handed to the model as a
coordinate. Here phase has to come out of flow snapshots, which is the part
that was flagged as the real risk.

Pipeline:
  1. POD (method of snapshots) on wake velocity fields -> coefficients a(t)
  2. a(t) traces a closed orbit; modes 1-2 are the shedding pair, so
     theta = atan2(a2, a1) is a usable phase without being told what phase is
  3. train two latent ODEs on a(t):
       free      adot = f(a)                            (standard practice)
       anchored  polar split on the dominant pair with rho* and omega explicit
                 and supervised against the amplitude and shedding frequency
                 measured from the data
  4. measure delta = (T_model - T_reference)/T_reference

Time is rescaled so one shedding period is 2*pi, matching the Tier-0 setup.
"""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import hilbert

torch.set_default_dtype(torch.float64)
J2 = torch.tensor([[0.0, -1.0], [1.0, 0.0]])


# ------------------------------------------------------------------ utilities
def period_from_signal(t, s, edge_frac=0.1):
    """
    Period by linear fit to the unwrapped phase of the analytic signal.
    Robust where zero-crossing counting is not: additive noise puts several
    spurious crossings around each true one and collapses the estimate.
    """
    s = np.asarray(s, dtype=float)
    s = s - s.mean()
    ph = np.unwrap(np.angle(hilbert(s)))
    k = max(1, int(edge_frac * len(s)))          # drop Hilbert edge transients
    slope = np.polyfit(t[k:-k], ph[k:-k], 1)[0]
    return float(2 * math.pi / abs(slope))


def pod(snaps, r):
    """
    Method of snapshots. snaps: [n, 2, nx, ny] -> coefficients [n, r],
    spatial modes [r, dof], mean [dof], energy fractions [n].
    Uses the n x n Gram matrix because n << dof here.
    """
    n = snaps.shape[0]
    X = snaps.reshape(n, -1).astype(np.float64)
    mean = X.mean(0)
    X = X - mean
    # Under np.seterr the BLAS matmul kernel raises overflow/underflow flags
    # from its padding lanes. G comes out finite and correct; the flags are an
    # artifact of the vectorised kernel, not of the data.
    with np.errstate(over="ignore", under="ignore", invalid="ignore",
                     divide="ignore"):
        G = X @ X.T
    w, V = np.linalg.eigh(G)                     # ascending
    w, V = w[::-1], V[:, ::-1]
    w = np.clip(w, 0, None)                      # eigh leaves ~-1e-10 dust
    energy = w / w.sum()
    a = V[:, :r] * np.sqrt(w[:r])                # temporal coefficients
    modes = (X.T @ V[:, :r]) / np.sqrt(np.maximum(w[:r], 1e-30))
    return a, modes.T, mean, energy


# -------------------------------------------------------------------- models
def _mlp(d_in, d_out, width, depth):
    L, d = [], d_in
    for _ in range(depth):
        L += [nn.Linear(d, width), nn.Tanh()]; d = width
    return nn.Sequential(*L, nn.Linear(d, d_out))


class FreeLatent(nn.Module):
    """adot = f(a). What the field does."""
    def __init__(self, r, width=96, depth=3):
        super().__init__()
        self.net = _mlp(r, r, width, depth)

    def forward(self, a):
        return self.net(a)


class AnchoredLatent(nn.Module):
    """
    Polar split on the dominant POD pair, rest of the modes free:

        rho' = (rho* - rho) p(a),   p > 0
        th'  = omega + (rho - rho*) k(a)
        v'   = g(a)                       (modes 3..r, slaved to the phase)

    rho* and omega are plain parameters. They are the two quantities that set
    long-horizon error, and both are measurable from data (oscillation
    amplitude and shedding frequency, i.e. Strouhal).
    """
    def __init__(self, r, width=96, depth=3, omega_init=1.0, rho_init=1.0):
        super().__init__()
        self.r = r
        self.p = _mlp(r, 1, width, depth)
        self.k = _mlp(r, 1, width, depth)
        self.g = _mlp(r, r - 2, width, depth) if r > 2 else None
        self.omega = nn.Parameter(torch.tensor(float(omega_init)))
        self.log_rho = nn.Parameter(torch.tensor(math.log(float(rho_init))))

    @property
    def rho_star(self):
        return self.log_rho.exp()

    def forward(self, a):
        u = a[..., :2]
        rho = u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        rs = self.rho_star
        rhodot = (rs - rho) * (nn.functional.softplus(self.p(a)) + 1e-6)
        thdot = self.omega + (rho - rs) * self.k(a)
        udot = rhodot * (u / rho) + thdot * (u @ J2.T)
        if self.g is None:
            return udot
        return torch.cat([udot, self.g(a)], dim=-1)


class PhaseAnchoredLatent(nn.Module):
    """
    Phase carried as its own coordinate, so the period is exact by
    CONSTRUCTION rather than by penalty:

        state      (phi, s),  phi on the circle, s the deviation from the cycle
        phi' = omega                       (uniform, and omega is not learned)
        s'   = g(s, phi)
        a    = Z(phi) + s                  Z a learned closed loop

    The rollout period is exactly 2*pi/omega. Nothing the optimiser does to Z
    or g can change it, so delta equals the error in the measured frequency and
    nothing else.

    This supersedes AnchoredLatent, which set theta' = omega pointwise on the
    dominant POD pair. That is a false constraint for a real wake: the physical
    phase advances ~0.5% non-uniformly around a shedding cycle, and forcing it
    uniform makes the fit trade period accuracy against field accuracy. Here Z
    absorbs the non-uniformity, because the orbit is a general closed loop in
    the latent space rather than a circle.
    """
    def __init__(self, r, n_harm=6, width=96, depth=3, omega=1.0,
                 learn_omega=False):
        super().__init__()
        self.r, self.n_harm = r, n_harm
        h = torch.arange(n_harm + 1, dtype=torch.get_default_dtype())
        self.register_buffer("harm", h)
        self.Zc = nn.Parameter(torch.zeros(n_harm + 1, r))
        self.Zs = nn.Parameter(torch.zeros(n_harm + 1, r))
        if learn_omega:
            self.omega = nn.Parameter(torch.tensor(float(omega)))
        else:
            self.register_buffer("omega", torch.tensor(float(omega)))
        # transverse dynamics see (s, cos phi, sin phi)
        self.g = _mlp(r + 2, r, width, depth)

    def Z(self, phi):
        """Closed loop in latent space: truncated Fourier series in phase."""
        ang = phi.unsqueeze(-1) * self.harm            # [..., n_harm+1]
        return (ang.cos() @ self.Zc) + (ang.sin() @ self.Zs)

    def decode(self, phi, s):
        return self.Z(phi) + s

    def forward(self, state):
        """state = [phi, s] concatenated: [..., 1+r]"""
        phi, s = state[..., 0], state[..., 1:]
        sdot = self.g(torch.cat([s, phi.cos().unsqueeze(-1),
                                 phi.sin().unsqueeze(-1)], dim=-1))
        phidot = self.omega.expand(phi.shape).unsqueeze(-1)
        return torch.cat([phidot, sdot], dim=-1)

    def init_state(self, a, phi0):
        """Deviation of an observed point from the loop at that phase."""
        return torch.cat([phi0.unsqueeze(-1), a - self.Z(phi0)], dim=-1)


def rk4(f, x, dt, n, noise=0.0):
    out = [x]
    for _ in range(n):
        k1 = f(x); k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2); k4 = f(x + dt * k3)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if noise:
            x = x + noise * torch.randn_like(x)
        out.append(x)
    return torch.stack(out)


# ------------------------------------------------------------------ training
def make_segments(a, seg_len, stride=1):
    """Overlapping windows of the coefficient trajectory, plus start indices."""
    idx = list(range(0, len(a) - seg_len - 1, stride))
    segs = [a[i:i + seg_len + 1] for i in idx]
    return torch.tensor(np.stack(segs)), torch.tensor(idx, dtype=torch.long)


def train(model, segs, dt, epochs, lr, sup=None, w_sup=1e2, aug_noise=0.0,
          batch=64, seed=0, verbose=False):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n_steps = segs.shape[1] - 1
    last = float("nan")
    for ep in range(epochs):
        idx = torch.randperm(len(segs))[:batch]
        b = segs[idx]
        x0 = b[:, 0]
        if aug_noise:
            # the data lies on the attractor; a little jitter is the only
            # off-cycle information either model gets, and both get it equally
            x0 = x0 + aug_noise * torch.randn_like(x0)
        opt.zero_grad()
        pred = rk4(model, x0, dt, n_steps)
        loss = ((pred - b.transpose(0, 1)) ** 2).mean()
        last = loss.item()
        if sup is not None:
            w_t, r_t = sup
            loss = loss + w_sup * ((model.omega - w_t) ** 2
                                   + (model.rho_star - r_t) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); sched.step()
        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            print(f"    ep {ep:>5} mse {last:.3e}", flush=True)
    return last


def train_phase(model, segs, idx, dt, epochs, lr, aug_noise=0.0, batch=64,
                seed=0):
    """
    Training for the phase-anchored model.

    The data is one uniformly sampled trajectory and time was rescaled so a
    period is 2*pi, so the phase at sample i is exactly omega * i * dt. Nothing
    has to be inferred: Z is fit as a Fourier series in that phase and g learns
    the approach to the loop.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n_steps = segs.shape[1] - 1
    om = float(model.omega)
    last = float("nan")
    for _ in range(epochs):
        sel = torch.randperm(len(segs))[:batch]
        b, i0 = segs[sel], idx[sel]
        phi0 = om * i0.to(b.dtype) * dt
        a0 = b[:, 0]
        if aug_noise:
            a0 = a0 + aug_noise * torch.randn_like(a0)
        opt.zero_grad()
        st = model.init_state(a0, phi0)
        traj = rk4(model, st, dt, n_steps)
        pred = model.decode(traj[..., 0], traj[..., 1:])
        loss = ((pred - b.transpose(0, 1)) ** 2).mean()
        last = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); sched.step()
    return last


@torch.no_grad()
def rollout_period_phase(model, a0, dt, n_periods):
    """
    Free rollout of the phase-anchored model. The period should come out at
    exactly 2*pi/omega; measuring it anyway is the check that nothing in the
    decode path reintroduces a frequency error.
    """
    n = int(n_periods * 2 * math.pi / dt)
    st = model.init_state(a0.unsqueeze(0), torch.zeros(1, dtype=a0.dtype))
    traj = rk4(model, st, dt, n)
    dec = model.decode(traj[..., 0], traj[..., 1:])[:, 0, :].numpy()
    if not np.isfinite(dec).all():
        return None, dec
    t = np.arange(len(dec)) * dt
    half = len(dec) // 2
    return period_from_signal(t[half:], dec[half:, 0]), dec


@torch.no_grad()
def rollout_period(model, a0, dt, n_periods, T_model_units=2 * math.pi):
    """Long free rollout, then period of the dominant coefficient."""
    n = int(n_periods * T_model_units / dt)
    tr = rk4(model, a0.unsqueeze(0), dt, n)[:, 0, :].numpy()
    if not np.isfinite(tr).all():
        return None, tr
    t = np.arange(len(tr)) * dt
    half = len(tr) // 2                       # let transients settle
    return period_from_signal(t[half:], tr[half:, 0]), tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", default="data/D16_snaps.npz")
    ap.add_argument("--modes", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--seg-len", type=int, default=25)
    ap.add_argument("--periods", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--aug-noise", type=float, default=1e-3)
    ap.add_argument("--w-sup", type=float, default=1e2)
    ap.add_argument("--harmonics", type=int, default=6,
                    help="Fourier harmonics in the phase-anchored loop Z(phi)")
    ap.add_argument("--free-widths", type=int, nargs="+",
                    default=[96, 168, 256],
                    help="widths for the free baseline; 168 is roughly "
                         "parameter-matched to the anchored model")
    ap.add_argument("--out", default="cyl_latent.json")
    a = ap.parse_args()

    d = np.load(a.snaps)
    snaps, t_lat = d["snaps"], d["t"].astype(float)
    T_lattice = float(d["T_lattice"])
    print(f"snapshots {snaps.shape}  T_lattice={T_lattice:.2f}  "
          f"St={float(d['St']):.5f}")

    # ---- POD
    coef, modes, mean, energy = pod(snaps, a.modes)
    print(f"POD energy: mode1-2 {energy[:2].sum()*100:.2f}%  "
          f"first {a.modes} {energy[:a.modes].sum()*100:.3f}%")
    print("  per-mode %: " + " ".join(f"{e*100:.2f}" for e in energy[:a.modes]))

    # ---- reference period, measured from the data in snapshot units
    dt_snap = float(np.median(np.diff(t_lat)))
    t_snap = (t_lat - t_lat[0]) / dt_snap
    T_ref_snap = period_from_signal(t_snap, coef[:, 0])
    print(f"reference period: {T_ref_snap:.4f} snapshots "
          f"= {T_ref_snap*dt_snap:.2f} lattice steps "
          f"(probe gave {T_lattice:.2f}, "
          f"{abs(T_ref_snap*dt_snap-T_lattice)/T_lattice*100:.3f}% apart)")

    # ---- rescale: one period = 2*pi, so omega = 1 (matches Tier 0)
    dt = 2 * math.pi / T_ref_snap
    scale = coef[:, 0].std()
    A = coef / scale
    rho_meas = float(np.median(np.linalg.norm(A[:, :2], axis=1)))
    print(f"dt={dt:.5f} model units/snapshot, rho*_measured={rho_meas:.4f}")

    segs, seg_idx = make_segments(A, a.seg_len)
    print(f"{len(segs)} training segments of {a.seg_len} steps "
          f"({a.seg_len*dt/(2*math.pi):.2f} periods each)\n")

    sup = (torch.tensor(1.0), torch.tensor(rho_meas))   # omega=1 by construction

    # The anchored model carries three MLPs to the free model's one, so a
    # width-matched comparison would hand it ~3x the parameters. Sweep the free
    # model's width past that point instead, which also re-tests whether extra
    # capacity buys any accuracy in delta.
    configs = ([("free", w) for w in a.free_widths]
               + [("anchored+sup", 96), ("phase-anchored", 96)])
    rows = []
    for kind, width in configs:
        for seed in range(a.seeds):
            t0 = time.time()
            torch.manual_seed(seed)
            a0 = torch.tensor(A[0])
            if kind == "free":
                model = FreeLatent(a.modes, width=width)
                nparam = sum(p.numel() for p in model.parameters())
                mse = train(model, segs, dt, a.epochs, 3e-3, sup=None,
                            aug_noise=a.aug_noise, seed=seed)
                T_mod, tr = rollout_period(model, a0, dt, a.periods)
            elif kind == "anchored+sup":
                model = AnchoredLatent(a.modes, width=width, omega_init=1.0,
                                       rho_init=rho_meas)
                nparam = sum(p.numel() for p in model.parameters())
                mse = train(model, segs, dt, a.epochs, 3e-3, sup=sup,
                            w_sup=a.w_sup, aug_noise=a.aug_noise, seed=seed)
                T_mod, tr = rollout_period(model, a0, dt, a.periods)
            else:
                # omega is measured, not learned: period is 2*pi/omega exactly
                model = PhaseAnchoredLatent(a.modes, n_harm=a.harmonics,
                                            width=width, omega=1.0)
                nparam = sum(p.numel() for p in model.parameters())
                mse = train_phase(model, segs, seg_idx, dt, a.epochs, 3e-3,
                                  aug_noise=a.aug_noise, seed=seed)
                T_mod, tr = rollout_period_phase(model, a0, dt, a.periods)
            if T_mod is None:
                print(f"  {kind:13s} w={width:<4d} s={seed} mse={mse:.2e}"
                      f"  DIVERGED", flush=True)
                rows.append(dict(kind=kind, width=width, nparam=nparam,
                                 seed=seed, train_mse=mse, delta=None,
                                 diverged=True))
                continue
            delta = (T_mod - 2 * math.pi) / (2 * math.pi)
            horizon = 1.0 / (4 * abs(delta)) if delta else float("inf")
            rows.append(dict(kind=kind, width=width, nparam=nparam, seed=seed,
                             train_mse=mse, T_model=T_mod, delta=delta,
                             usable_horizon_periods=horizon,
                             amp_final=float(np.linalg.norm(tr[-1, :2])),
                             rho_star_meas=rho_meas, diverged=False,
                             secs=time.time() - t0))
            print(f"  {kind:13s} w={width:<4d} s={seed} np={nparam:>6d} "
                  f"mse={mse:.2e} delta={delta:+.3e} "
                  f"horizon={horizon:>8.0f}P "
                  f"amp_end={np.linalg.norm(tr[-1,:2]):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print()
    for kind, width in configs:
        ds = [abs(r["delta"]) for r in rows if r["kind"] == kind
              and r["width"] == width and r["delta"] is not None]
        ms = [r["train_mse"] for r in rows if r["kind"] == kind
              and r["width"] == width]
        if ds:
            print(f"  {kind:13s} w={width:<4d}: median |delta| = "
                  f"{np.median(ds):.3e}  horizon {1/(4*np.median(ds)):>8.0f}P  "
                  f"med mse {np.median(ms):.2e}  ({len(ds)}/{a.seeds} conv)")
    json.dump(dict(rows=rows, T_ref_snap=T_ref_snap, T_lattice=T_lattice,
                   dt=dt, energy=energy[:a.modes].tolist(),
                   modes=a.modes), open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
