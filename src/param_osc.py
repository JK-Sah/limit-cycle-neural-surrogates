"""
Parametric oscillators: does delta blow up when the frequency has to be
INTERPOLATED across a parameter rather than memorised as one scalar?

Everything so far fits a single operating point, where the model only has to
find one number. A parametric operator -- the thing the FSI proposal actually
wants -- must produce omega(mu) at parameter values it never saw. That is the
first place delta has a structural reason to be large.

System: van der Pol,  x' = y,  y' = eps (1 - x^2) y - x.
Its period runs from ~6.38 at eps=0.5 to ~8.86 at eps=3, a 40% swing, and the
dependence is nonlinear, so omega(eps) is a genuine function to learn.

Splits:
  train  eps = 0.5, 0.9, 1.3, 1.7, 2.1, 2.5
  interp eps = 0.7, 1.1, 1.5, 1.9, 2.3   (inside the training range)
  extrap eps = 2.9, 3.3                  (outside it)

Two models, both parameter-conditioned:
  free            adot = f(a, eps)
  phase-anchored  phi' = omega_t(eps),  a = Z(phi, eps) + s,  s' = g(s, phi, eps)
                  with omega_t supervised at the TRAINING eps only, so delta at
                  held-out eps is exactly the interpolation error of a smooth
                  scalar function fitted to exact measurements.

The hypothesis is that the free model's frequency, being an uncontrolled
by-product of field matching, generalises much worse than a directly
interpolated omega(eps).
"""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.signal import hilbert

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- ground truth
def vdp_np(t, z, eps):
    x, y = z
    return [y, eps * (1 - x * x) * y - x]


def vdp_period(eps, n_skip=60, n_meas=60):
    """
    Period by Poincare section x=0 with y>0, after discarding a transient.
    Tight tolerances: this is the reference every delta is measured against.
    """
    T_guess = 6.4 + 0.85 * (eps - 0.5)
    t_end = (n_skip + n_meas + 2) * T_guess
    sol = solve_ivp(vdp_np, (0, t_end), [2.0, 0.0], args=(eps,),
                    rtol=1e-12, atol=1e-12, dense_output=True, max_step=0.05)
    ts = np.linspace(0, t_end, int(t_end / 0.002))
    z = sol.sol(ts)
    x, y = z[0], z[1]
    cr = []
    for i in range(len(ts) - 1):
        if x[i] < 0 <= x[i + 1] and y[i] > 0:
            cr.append(ts[i] + (ts[i + 1] - ts[i]) * (-x[i] / (x[i + 1] - x[i])))
    cr = np.array(cr)
    cr = cr[len(cr) // 3:]                       # drop transient
    return float(np.mean(np.diff(cr))), float(np.std(np.diff(cr))), sol


def cycle_trajectory(eps, n_periods, dt):
    """
    One trajectory starting exactly on the limit cycle at a Poincare crossing,
    so phase is known: phi(t) = omega * t.
    """
    T, _, _ = vdp_period(eps)
    sol = solve_ivp(vdp_np, (0, 400 * 1.0 + n_periods * T + 10), [2.0, 0.0],
                    args=(eps,), rtol=1e-12, atol=1e-12, dense_output=True,
                    max_step=0.05)
    # find a late crossing to start from
    ts = np.linspace(0, 60 * T, int(60 * T / 0.002))
    z = sol.sol(ts)
    x, y = z[0], z[1]
    t0 = None
    for i in range(len(ts) - 1):
        if x[i] < 0 <= x[i + 1] and y[i] > 0 and ts[i] > 40 * T:
            t0 = ts[i] + (ts[i + 1] - ts[i]) * (-x[i] / (x[i + 1] - x[i]))
            break
    tq = t0 + np.arange(0, n_periods * T, dt)
    return sol.sol(tq).T, T


# --------------------------------------------------------------------- models
def _mlp(d_in, d_out, width, depth, act=nn.Tanh):
    L, d = [], d_in
    for _ in range(depth):
        L += [nn.Linear(d, width), act()]; d = width
    return nn.Sequential(*L, nn.Linear(d, d_out))


class FreeParametric(nn.Module):
    """adot = f(a, eps). The standard parameter-conditioned neural ODE."""
    def __init__(self, dim=2, width=96, depth=3):
        super().__init__()
        self.net = _mlp(dim + 1, dim, width, depth)

    def forward(self, a, eps):
        return self.net(torch.cat([a, eps], dim=-1))


class PhaseAnchoredParametric(nn.Module):
    """
    phi' = omega(eps)          omega a learned scalar function, supervised
    a    = Z(phi, eps) + s     Z a Fourier loop with eps-dependent coefficients
    s'   = g(s, phi, eps)

    At a held-out eps the rollout period is exactly 2*pi/omega(eps), so delta is
    the interpolation error of omega -- a smooth 1-D function fitted to exact
    measurements -- rather than an emergent property of the fit.
    """
    def __init__(self, dim=2, n_harm=8, width=96, depth=3, w_head=64):
        super().__init__()
        self.dim, self.n_harm = dim, n_harm
        self.register_buffer("harm", torch.arange(n_harm + 1,
                                                  dtype=torch.get_default_dtype()))
        self.omega_net = _mlp(1, 1, w_head, 2)
        self.coef_net = _mlp(1, 2 * (n_harm + 1) * dim, width, depth)
        self.g = _mlp(dim + 3, dim, width, depth)

    def omega(self, eps):
        return nn.functional.softplus(self.omega_net(eps)) + 1e-3

    def Z(self, phi, eps):
        c = self.coef_net(eps)
        b = c.shape[:-1]
        c = c.view(*b, 2, self.n_harm + 1, self.dim)
        ang = phi.unsqueeze(-1) * self.harm                      # [..., H+1]
        cos, sin = ang.cos().unsqueeze(-1), ang.sin().unsqueeze(-1)
        return (cos * c[..., 0, :, :]).sum(-2) + (sin * c[..., 1, :, :]).sum(-2)

    def decode(self, phi, s, eps):
        return self.Z(phi, eps) + s

    def forward(self, state, eps):
        phi, s = state[..., :1], state[..., 1:]
        sdot = self.g(torch.cat([s, phi.cos(), phi.sin(), eps], dim=-1))
        return torch.cat([self.omega(eps), sdot], dim=-1)

    def init_state(self, a, phi0, eps):
        return torch.cat([phi0, a - self.Z(phi0.squeeze(-1), eps)], dim=-1)


def rk4(f, x, eps, dt, n):
    out = [x]
    for _ in range(n):
        k1 = f(x, eps); k2 = f(x + 0.5 * dt * k1, eps)
        k3 = f(x + 0.5 * dt * k2, eps); k4 = f(x + dt * k3, eps)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(x)
    return torch.stack(out)


# ------------------------------------------------------------------ utilities
def period_from_signal(t, s, edge_frac=0.1):
    s = np.asarray(s, float); s = s - s.mean()
    ph = np.unwrap(np.angle(hilbert(s)))
    k = max(1, int(edge_frac * len(s)))
    return float(2 * math.pi / abs(np.polyfit(t[k:-k], ph[k:-k], 1)[0]))


@torch.no_grad()
def rollout_delta(model, kind, eps_val, T_true, dt, n_periods, scale, a0):
    """Free rollout at one eps; returns delta against the true period."""
    e = torch.tensor([[eps_val]])
    n = int(n_periods * T_true / dt)
    if kind == "free":
        tr = rk4(model, a0.unsqueeze(0), e, dt, n)[:, 0, :].numpy()
    else:
        st = model.init_state(a0.unsqueeze(0), torch.zeros(1, 1), e)
        traj = rk4(model, st, e, dt, n)
        tr = model.decode(traj[..., 0], traj[..., 1:], e)[:, 0, :].numpy()
    if not np.isfinite(tr).all():
        return None, None
    t = np.arange(len(tr)) * dt
    half = len(tr) // 2
    T_mod = period_from_signal(t[half:], tr[half:, 0])
    return (T_mod - T_true) / T_true, T_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--periods", type=int, default=100)
    ap.add_argument("--seg-len", type=int, default=30)
    ap.add_argument("--dt", type=float, default=0.12)
    ap.add_argument("--data-periods", type=int, default=30)
    ap.add_argument("--w-sup", type=float, default=1e3)
    ap.add_argument("--harmonics", type=int, default=8)
    ap.add_argument("--free-widths", type=int, nargs="+", default=[96, 150, 224],
                    help="150 is roughly parameter-matched to the anchored model")
    ap.add_argument("--seed-list", type=int, nargs="+", default=None,
                    help="explicit seeds (for Slurm arrays); overrides --seeds")
    ap.add_argument("--out", default="param_osc.json")
    a = ap.parse_args()

    train_eps = [0.5, 0.9, 1.3, 1.7, 2.1, 2.5]
    interp_eps = [0.7, 1.1, 1.5, 1.9, 2.3]
    extrap_eps = [2.9, 3.3]
    all_eps = train_eps + interp_eps + extrap_eps

    print("measuring reference periods (tight-tolerance Poincare)...")
    T = {}
    for e in all_eps:
        t, sd, _ = vdp_period(e)
        T[e] = t
        tag = ("train" if e in train_eps else
               "interp" if e in interp_eps else "extrap")
        print(f"  eps={e:.1f} [{tag:>6}]  T={t:.8f}  jitter={sd/t:.2e}")

    print("\ngenerating on-cycle trajectories...")
    data = {}
    for e in train_eps:
        z, _ = cycle_trajectory(e, a.data_periods, a.dt)
        data[e] = z
    scale = float(np.abs(np.concatenate(list(data.values()))).max())
    print(f"  {a.data_periods} periods each at dt={a.dt}, state scale {scale:.4f}")

    # segments, tagged with eps and with the phase at the segment start
    segs, seg_eps, seg_phi0 = [], [], []
    for e in train_eps:
        z = data[e] / scale
        om = 2 * math.pi / T[e]
        for i in range(len(z) - a.seg_len - 1):
            segs.append(z[i:i + a.seg_len + 1])
            seg_eps.append(e)
            seg_phi0.append(om * i * a.dt)
    segs = torch.tensor(np.stack(segs))
    seg_eps = torch.tensor(seg_eps).unsqueeze(-1)
    seg_phi0 = torch.tensor(seg_phi0).unsqueeze(-1)
    print(f"  {len(segs)} segments of {a.seg_len} steps")

    om_true = torch.tensor([[2 * math.pi / T[e]] for e in train_eps])
    eps_sup = torch.tensor([[e] for e in train_eps])

    seeds = a.seed_list if a.seed_list else list(range(a.seeds))
    configs = ([("free", w) for w in a.free_widths]
               + [("phase-anchored", 96)])
    rows = []
    for kind, width in configs:
        for seed in seeds:
            t0 = time.time()
            torch.manual_seed(seed)
            model = (FreeParametric(width=width) if kind == "free"
                     else PhaseAnchoredParametric(n_harm=a.harmonics,
                                                  width=width))
            npar = sum(p.numel() for p in model.parameters())
            opt = torch.optim.Adam(model.parameters(), lr=3e-3)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
            for ep in range(a.epochs):
                idx = torch.randperm(len(segs))[:128]
                b, be, bp = segs[idx], seg_eps[idx], seg_phi0[idx]
                opt.zero_grad()
                if kind == "free":
                    pred = rk4(model, b[:, 0], be, a.dt, a.seg_len)
                else:
                    st = model.init_state(b[:, 0], bp, be)
                    tj = rk4(model, st, be, a.dt, a.seg_len)
                    pred = model.decode(tj[..., 0], tj[..., 1:], be)
                loss = ((pred - b.transpose(0, 1)) ** 2).mean()
                mse = loss.item()
                if kind != "free":
                    # supervise omega at the TRAINING eps only
                    loss = loss + a.w_sup * ((model.omega(eps_sup)
                                              - om_true) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step(); sch.step()

            a0 = torch.tensor(data[train_eps[0]][0] / scale)
            for e in all_eps:
                tag = ("train" if e in train_eps else
                       "interp" if e in interp_eps else "extrap")
                z0 = torch.tensor(cycle_trajectory(e, 1, a.dt)[0][0] / scale) \
                    if e not in data else torch.tensor(data[e][0] / scale)
                d, T_mod = rollout_delta(model, kind, e, T[e], a.dt,
                                         a.periods, scale, z0)
                rows.append(dict(kind=kind, width=width, seed=seed, eps=e,
                                 split=tag, nparam=npar, train_mse=mse,
                                 delta=d, T_true=T[e], T_model=T_mod))
            ds = {t: [abs(r["delta"]) for r in rows if r["kind"] == kind
                      and r["width"] == width and r["seed"] == seed
                      and r["split"] == t and r["delta"] is not None]
                  for t in ["train", "interp", "extrap"]}
            fmt = lambda t: (f"{np.median(ds[t]):.2e}" if ds[t] else "  n/a  ")
            print(f"  {kind:15s} w={width:<4d} s={seed} np={npar:>6d} "
                  f"mse={mse:.2e} train={fmt('train')} "
                  f"interp={fmt('interp')} extrap={fmt('extrap')} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print()
    hdr = "{:>15} {:>5} {:>8} {:>12} {:>12} {:>10}".format(
        "model", "width", "split", "med |delta|", "worst", "horizon P")
    print(hdr); print("-" * len(hdr))
    for kind, width in configs:
        for tag in ["train", "interp", "extrap"]:
            ds = [abs(r["delta"]) for r in rows if r["kind"] == kind
                  and r["width"] == width and r["split"] == tag
                  and r["delta"] is not None]
            if ds:
                m = np.median(ds)
                print("{:>15} {:>5} {:>8} {:>12.3e} {:>12.3e} {:>10.0f}".format(
                    kind, width, tag, m, max(ds), 1 / (4 * m)))
    json.dump(dict(rows=rows, T=T, train_eps=train_eps,
                   interp_eps=interp_eps, extrap_eps=extrap_eps),
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
