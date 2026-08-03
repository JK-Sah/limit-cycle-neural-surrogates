"""
Tier-0 falsification test: is long-horizon drift in learned surrogates of
limit-cycle systems dominated by SECULAR PHASE ERROR rather than amplitude error?

Ground truth: Stuart-Landau (Hopf normal form), where everything is analytic.
    r' = mu*r - r^3
    th' = omega - beta*r^2
  limit cycle:  r* = sqrt(mu),  Omega = omega - beta*mu,  T = 2*pi/Omega
  Floquet multipliers: lam_phase = 1 (exact, time-translation invariance)
                       lam_trans = exp(-2*mu*T)

Predictions under test:
  P1  phase error grows LINEARLY in t (unbounded); amplitude error stays BOUNDED.
  P2  scaling (more data / wider net) shrinks the frequency error delta but does
      not change the linear-in-t character; time-to-decorrelation ~ 1/delta only.
  P3  stability regularizers that act on the transverse direction (Lipschitz /
      energy-amplitude penalties) leave the phase drift essentially unchanged.
"""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev

torch.set_default_dtype(torch.float64)
DEV = "cpu"  # float64 + tiny nets: CPU is fastest and fully deterministic

# ---------------------------------------------------------------- ground truth
class SL:
    """Stuart-Landau in Cartesian coords, with analytic cycle properties."""
    def __init__(self, mu=0.5, omega=1.0, beta=-0.5):
        self.mu, self.omega, self.beta = mu, omega, beta
        self.r_star = math.sqrt(mu)
        self.Omega = omega - beta * mu
        self.T = 2 * math.pi / abs(self.Omega)
        self.lam_trans = math.exp(-2 * mu * self.T)

    def f_np(self, x):
        x = np.atleast_2d(x)
        r2 = (x ** 2).sum(-1, keepdims=True)
        a = self.mu - r2                       # radial gain
        b = self.omega - self.beta * r2        # angular rate
        return np.concatenate([a * x[:, :1] - b * x[:, 1:],
                               b * x[:, :1] + a * x[:, 1:]], -1)

    def rollout(self, x0, t):
        """Dense analytic-quality reference via tight-tolerance RK (scipy)."""
        from scipy.integrate import solve_ivp
        sol = solve_ivp(lambda _t, y: self.f_np(y).ravel(), (t[0], t[-1]), x0,
                        t_eval=t, rtol=1e-12, atol=1e-12, method="DOP853")
        return sol.y.T


# ---------------------------------------------------------------------- model
class VF(nn.Module):
    """Learned autonomous vector field f_phi(x)."""
    def __init__(self, width=64, depth=3, dim=2):
        super().__init__()
        L, d = [], dim
        for _ in range(depth):
            L += [nn.Linear(d, width), nn.Tanh()]
            d = width
        L += [nn.Linear(d, dim)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


def rk4(f, x, dt, n, noise=0.0):
    """
    Fixed-step RK4 rollout; returns [n+1, ..., dim] including x0.
    noise>0 injects N(0,noise) into the state each step -- this is the
    pushforward / noise-injection trick (Brandstetter et al.) that the PDE
    surrogate literature uses to buy long-rollout stability.
    """
    out = [x]
    for _ in range(n):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if noise:
            x = x + noise * torch.randn_like(x)
        out.append(x)
    return torch.stack(out)


# ------------------------------------------------------------------- training
def make_data(sys, n_traj, seg_len, dt, seed):
    """Short trajectory segments from ICs spread on and off the cycle."""
    rng = np.random.default_rng(seed)
    t = np.arange(seg_len + 1) * dt
    X = []
    for _ in range(n_traj):
        # radii from 0.3 to 1.7 of r*: model must see approach to the cycle
        r = sys.r_star * rng.uniform(0.3, 1.7)
        th = rng.uniform(0, 2 * math.pi)
        x0 = np.array([r * math.cos(th), r * math.sin(th)])
        X.append(sys.rollout(x0, t))
    return torch.tensor(np.stack(X))          # [n_traj, seg_len+1, 2]


def train(model, data, dt, epochs, lr, reg=None, reg_w=0.0, seed=0, log=None):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n_steps = data.shape[1] - 1
    x0, target = data[:, 0], data.transpose(0, 1)      # [S+1, N, 2]
    for ep in range(epochs):
        opt.zero_grad()
        pred = rk4(model, x0, dt, n_steps,
                   noise=reg_w if reg == "noise" else 0.0)
        loss = ((pred - target) ** 2).mean()
        base = loss.item()
        if reg == "lipschitz":
            # penalize Frobenius norm of d f / d x  (the standard spectral fix)
            xs = data.reshape(-1, 2).detach().requires_grad_(True)
            fx = model(xs)
            J = torch.autograd.grad(fx.sum(), xs, create_graph=True)[0]
            loss = loss + reg_w * (J ** 2).mean()
        elif reg == "energy":
            # penalize drift of the amplitude invariant along a rollout
            r2 = (pred ** 2).sum(-1)
            loss = loss + reg_w * ((r2[1:] - r2[:-1]) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); sched.step()
        if log is not None and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            log.append((ep, base))
    return model


# ---------------------------------------------------------------- diagnostics
@torch.no_grad()
def learned_period(model, sys, dt=1e-3, n_cyc=60):
    """Poincare section y=0, x>0; interpolated crossing times."""
    x = torch.tensor([[sys.r_star, 0.0]])
    n = int(n_cyc * sys.T / dt)
    traj = rk4(model, x, dt, n)[:, 0, :].numpy()
    cross = []
    for i in range(len(traj) - 1):
        y0, y1 = traj[i, 1], traj[i + 1, 1]
        if y0 < 0 <= y1 and traj[i, 0] > 0:
            frac = -y0 / (y1 - y0)             # linear interp to y=0
            cross.append((i + frac) * dt)
    if len(cross) < 12:
        return None, None
    cross = np.array(cross[len(cross) // 3:])  # drop transient
    return float(np.mean(np.diff(cross))), float(np.std(np.diff(cross)))


def _jac(model, x):
    """Jacobian df/dx at a single point x:[dim] -> [dim,dim]."""
    return jacrev(lambda z: model(z.unsqueeze(0)).squeeze(0))(x)


def floquet(model, sys, That, dt=2e-3):
    """
    Monodromy matrix via RK4 on the joint system
        x' = f(x),   Phi' = J(x) Phi,   Phi(0) = I
    integrated over exactly one learned period. lam_1 must come out = 1
    (time-translation invariance) -- that is the correctness check, and it is
    also precisely why the phase direction has no restoring force.
    """
    if That is None:
        return None
    with torch.no_grad():                       # settle onto the learned cycle
        x = rk4(model, torch.tensor([[sys.r_star, 0.0]]), dt,
                int(40 * sys.T / dt))[-1, 0]

    def joint(state):
        x_, Phi_ = state
        with torch.no_grad():
            fx = model(x_.unsqueeze(0)).squeeze(0)
        return fx, _jac(model, x_) @ Phi_

    Phi = torch.eye(2)
    n = int(round(That / dt))
    h = That / n                                # land exactly on one period
    for _ in range(n):
        k1 = joint((x, Phi))
        k2 = joint((x + 0.5 * h * k1[0], Phi + 0.5 * h * k1[1]))
        k3 = joint((x + 0.5 * h * k2[0], Phi + 0.5 * h * k2[1]))
        k4 = joint((x + h * k3[0], Phi + h * k3[1]))
        x = x + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        Phi = Phi + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    ev = np.linalg.eigvals(Phi.detach().numpy())
    return sorted(np.abs(ev))[::-1]            # descending |lambda|


@torch.no_grad()
def drift_decomposition(model, sys, n_periods=200, dt=2e-3):
    """Split rollout error into phase (secular?) and amplitude (bounded?)."""
    t_end = n_periods * sys.T
    n = int(t_end / dt)
    t = np.arange(n + 1) * dt
    x0 = np.array([sys.r_star, 0.0])
    ref = sys.rollout(x0, t)
    pred = rk4(model, torch.tensor(np.array([x0])), dt, n)[:, 0, :].numpy()

    th_r = np.unwrap(np.arctan2(ref[:, 1], ref[:, 0]))
    th_p = np.unwrap(np.arctan2(pred[:, 1], pred[:, 0]))
    r_r = np.linalg.norm(ref, axis=1)
    r_p = np.linalg.norm(pred, axis=1)

    dphase = th_p - th_r
    damp = r_p - r_r
    l2 = np.linalg.norm(pred - ref, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-30)

    # slope of phase error over the last 80% (secular rate)
    m = t > 0.2 * t_end
    slope = float(np.polyfit(t[m], dphase[m], 1)[0])
    lin_resid = float(np.std(dphase[m] - np.polyval(np.polyfit(t[m], dphase[m], 1), t[m])))
    return dict(t=t, dphase=dphase, damp=damp, l2=l2,
                phase_slope=slope, phase_lin_resid=lin_resid,
                amp_max=float(np.abs(damp[m]).max()),
                amp_final=float(np.abs(damp[-1])),
                phase_final=float(abs(dphase[-1])),
                l2_final=float(l2[-1]))


def decorrelation_time(res, sys, thresh=math.pi / 2):
    """First time |phase error| exceeds thresh -> prediction is decorrelated."""
    idx = np.where(np.abs(res["dphase"]) > thresh)[0]
    return float(res["t"][idx[0]] / sys.T) if len(idx) else float("inf")


# ------------------------------------------------------------------ one trial
def run_trial(sys, n_traj, width, seed, epochs, seg_len=40, dt=0.05,
              reg=None, reg_w=0.0, n_periods=200, verbose=True):
    t0 = time.time()
    data = make_data(sys, n_traj, seg_len, dt, seed)
    model = VF(width=width).to(DEV)
    log = []
    train(model, data, dt, epochs, 3e-3, reg, reg_w, seed, log)
    That, Tstd = learned_period(model, sys)
    fl = floquet(model, sys, That)
    res = drift_decomposition(model, sys, n_periods)
    delta = (That - sys.T) / sys.T if That else None
    # Floquet prediction: dtheta/dt = 2*pi/That - 2*pi/T  (pure kinematics of a
    # frequency error on a neutrally-stable phase direction)
    slope_pred = (2 * math.pi / That - 2 * math.pi / sys.T) if That else None
    out = dict(n_traj=n_traj, width=width, seed=seed, reg=reg or "none",
               reg_w=reg_w, train_mse=log[-1][1],
               T_true=sys.T, T_learned=That, T_jitter=Tstd,
               delta_T=delta,
               floquet_phase=fl[0] if fl else None,
               floquet_trans=fl[1] if fl else None,
               floquet_trans_true=sys.lam_trans,
               phase_slope=res["phase_slope"], phase_slope_pred=slope_pred,
               phase_slope_ratio=(res["phase_slope"] / slope_pred
                                  if slope_pred else None),
               phase_lin_resid=res["phase_lin_resid"],
               phase_final=res["phase_final"], amp_max=res["amp_max"],
               amp_final=res["amp_final"], l2_final=res["l2_final"],
               decorr_periods=decorrelation_time(res, sys),
               secs=time.time() - t0)
    if verbose:
        g = lambda k, f, w=0: (format(out[k], f) if out[k] is not None
                               else "n/a".rjust(w))
        print(f"  n={n_traj:4d} w={width:4d} s={seed} reg={out['reg']:9s} "
              f"mse={out['train_mse']:.2e} dT/T={g('delta_T','+.2e',9)} "
              f"|lam1|={g('floquet_phase','.8f',10)} "
              f"|lam2|={g('floquet_trans','.2e',8)} "
              f"slope/pred={g('phase_slope_ratio','.3f',6)} "
              f"phase_end={out['phase_final']:6.2f} amp_end={out['amp_final']:.2e} "
              f"decorr={out['decorr_periods']:.0f}T ({out['secs']:.0f}s)", flush=True)
    return out, res


N_SEEDS = 5


def build_grid(mode):
    """Flat list of trial configs, addressable by Slurm array index."""
    g = []
    if mode in ("scaling", "all"):
        # P2: does more data / capacity remove the secular term?
        for n_traj in [16, 64, 256, 1024]:
            for width in [32, 128]:
                for s in range(N_SEEDS):
                    g.append(dict(exp="scaling", n_traj=n_traj, width=width,
                                  seed=s, reg=None, reg_w=0.0))
    if mode in ("regularizers", "all"):
        # P3: do the field's stability fixes touch the phase drift?
        for reg, w in [(None, 0.0),
                       ("lipschitz", 1e-3), ("lipschitz", 1e-2),
                       ("energy", 1e-2), ("energy", 1e-1),
                       ("noise", 1e-3), ("noise", 1e-2)]:
            for s in range(N_SEEDS):
                g.append(dict(exp="regularizers", n_traj=64, width=64,
                              seed=s, reg=reg, reg_w=w))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot",
                    choices=["pilot", "scaling", "regularizers", "all"])
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--periods", type=int, default=200)
    ap.add_argument("--task-id", type=int, default=None,
                    help="run only this index of the grid (Slurm array)")
    ap.add_argument("--count", action="store_true", help="print grid size, exit")
    ap.add_argument("--out", default="t0_results.json")
    a = ap.parse_args()

    if a.count:
        print(len(build_grid(a.mode))); return

    sys_ = SL()
    print(f"Stuart-Landau: r*={sys_.r_star:.4f}  Omega={sys_.Omega:.4f}  "
          f"T={sys_.T:.6f}  lam_trans_true={sys_.lam_trans:.3e}")
    print(f"(transverse direction is attracting by {1/sys_.lam_trans:.0f}x per "
          f"period; phase direction is exactly neutral)\n", flush=True)

    if a.mode == "pilot":
        cfgs = [dict(exp="pilot", n_traj=64, width=64, seed=s, reg=None,
                     reg_w=0.0) for s in range(3)]
    else:
        cfgs = build_grid(a.mode)
    if a.task_id is not None:
        cfgs = [cfgs[a.task_id]]
        print(f"task {a.task_id}: {cfgs[0]}", flush=True)

    rows = []
    for c in cfgs:
        r, _ = run_trial(sys_, c["n_traj"], c["width"], c["seed"], a.epochs,
                         reg=c["reg"], reg_w=c["reg_w"], n_periods=a.periods)
        r["exp"] = c["exp"]
        rows.append(r)

    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote {a.out}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
