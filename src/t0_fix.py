"""
Tier-0 constructive test: can the secular phase drift be removed structurally?

Diagnosis being acted on: in a black-box neural ODE the limit-cycle period is an
EMERGENT property. Nothing in a short-horizon L2 loss supervises it directly, so
the frequency error delta is whatever the fit happens to leave behind -- and
delta alone sets the long-horizon error via dtheta/dt = Omega_hat - Omega.

Four models, increasing structure:
  free          plain MLP vector field                    (what the field does)
  struct        r' = h(r), th' = w + k(r)                 (phase split out, but
                                                           k(r*) is free so
                                                           supervising w does
                                                           nothing -- ablation)
  anchored      r' = (r*-r)p(r), th' = w + (r-r*)k(r)     (cycle by construction,
                                                           invariants explicit)
  anchored+sup  anchored, with r* and w supervised against the amplitude and
                frequency measured from the data          (the proposed fix)

Prediction: delta_free ~ delta_struct ~ delta_anchored >> delta_anchored+sup,
and usable horizon scales as 1/delta. The fix costs two scalars and one loss
term; the structure alone buys almost nothing.
"""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import hilbert

from t0_falsify import (SL, VF, rk4, make_data, learned_period, floquet,
                        drift_decomposition, decorrelation_time)

torch.set_default_dtype(torch.float64)
J2 = torch.tensor([[0.0, -1.0], [1.0, 0.0]])


def _mlp(width, depth, d_in=1):
    L, d = [], d_in
    for _ in range(depth):
        L += [nn.Linear(d, width), nn.Tanh()]; d = width
    return nn.Sequential(*L, nn.Linear(d, 1))


class StructuredVF(nn.Module):
    """
    Naive polar split:  r' = h(r),  th' = w + k(r).
    NOTE this is the ablation that does NOT work: the angular rate on the cycle
    is w + k(r*), and k is free, so supervising w alone constrains nothing.
    Kept deliberately -- structure by itself is not the fix.
    """
    def __init__(self, width=64, depth=2, omega_init=1.0):
        super().__init__()
        self.h, self.k = _mlp(width, depth), _mlp(width, depth)
        self.omega = nn.Parameter(torch.tensor(float(omega_init)))

    def forward(self, x):
        r = x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return self.h(r) * (x / r) + (self.omega + self.k(r)) * (x @ J2.T)


class AnchoredVF(nn.Module):
    """
    Limit cycle by construction, with BOTH invariants explicit:

        r'  = (r* - r) * p(r),      p > 0   -> attracting cycle exactly at r*
        th' = w + (r - r*) * k(r)           -> angular rate exactly w on it

    r* and w are plain parameters, so the two quantities that govern
    long-horizon error are the two things you can supervise from data
    (oscillation amplitude and frequency -- in FSI, amplitude and Strouhal).
    The transverse rate p stays free: it is the direction that was never the
    problem.
    """
    def __init__(self, width=64, depth=2, omega_init=1.0, r_init=1.0):
        super().__init__()
        self.p, self.k = _mlp(width, depth), _mlp(width, depth)
        self.omega = nn.Parameter(torch.tensor(float(omega_init)))
        self.log_r = nn.Parameter(torch.tensor(math.log(float(r_init))))

    @property
    def r_star(self):
        return self.log_r.exp()

    def forward(self, x):
        r = x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        rs = self.r_star
        rdot = (rs - r) * (nn.functional.softplus(self.p(r)) + 1e-6)
        thdot = self.omega + (r - rs) * self.k(r)
        return rdot * (x / r) + thdot * (x @ J2.T)


def measured_invariants(sys, dt=1e-3, n_cyc=40, noise=0.0, seed=0):
    """
    What a practitioner can measure from observed data alone: settle onto the
    cycle, then estimate period and amplitude from the observed trajectory.
    `noise` emulates finite sensor/solver precision.

    The period comes from a linear fit to the unwrapped phase of the analytic
    signal. Naive zero-crossing detection must NOT be used here: additive noise
    puts several spurious crossings around each true one, which inflates the
    crossing count and collapses the estimated period (at noise=1e-3 it reports
    a 25% period error that is entirely an artifact of the estimator).
    Amplitude uses a median for the same robustness reason.

    Returns (T_measured, r_measured).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(n_cyc * sys.T / dt)) * dt
    traj = sys.rollout(np.array([sys.r_star, 0.0]), t)
    if noise:
        traj = traj + rng.normal(0, noise, traj.shape)

    k = max(1, len(t) // 10)                    # drop Hilbert edge transients
    ph = np.unwrap(np.angle(hilbert(traj[:, 1] - traj[:, 1].mean())))
    slope = np.polyfit(t[k:-k], ph[k:-k], 1)[0]
    T = float(2 * math.pi / abs(slope))
    r = float(np.median(np.linalg.norm(traj[len(traj) // 2:], axis=1)))
    return T, r


def train(model, data, dt, epochs, lr, sup=None, w_sup=1.0, seed=0):
    """sup = (omega_target, r_target) or None."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n_steps = data.shape[1] - 1
    x0, target = data[:, 0], data.transpose(0, 1)
    last = float("nan")
    for _ in range(epochs):
        opt.zero_grad()
        pred = rk4(model, x0, dt, n_steps)
        loss = ((pred - target) ** 2).mean()
        last = loss.item()
        if sup is not None:
            # the entire fix: two scalars, directly supervised
            w_t, r_t = sup
            loss = loss + w_sup * ((model.omega - w_t) ** 2
                                   + (model.r_star - r_t) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); sched.step()
    return last


def run(kind, sys, n_traj, width, seed, epochs, dt=0.05, seg_len=40,
        n_periods=200, obs_noise=0.0, w_sup=1.0):
    t0 = time.time()
    data = make_data(sys, n_traj, seg_len, dt, seed)
    sup, T_meas = None, None
    if kind == "free":
        model = VF(width=width)
    elif kind == "struct":
        model = StructuredVF(width=width)
    else:                                        # anchored / anchored+sup
        model = AnchoredVF(width=width)
        if kind == "anchored+sup":
            T_meas, r_meas = measured_invariants(sys, noise=obs_noise, seed=seed)
            sup = (torch.tensor(2 * math.pi / T_meas), torch.tensor(r_meas))
    mse = train(model, data, dt, epochs, 3e-3, sup, w_sup=w_sup, seed=seed)

    That, _ = learned_period(model, sys)
    fl = floquet(model, sys, That)
    res = drift_decomposition(model, sys, n_periods)
    delta = (That - sys.T) / sys.T if That else None
    out = dict(kind=kind, n_traj=n_traj, width=width, seed=seed, train_mse=mse,
               w_sup=w_sup, obs_noise=obs_noise,
               T_true=sys.T, T_measured_from_data=T_meas, T_learned=That,
               delta_T=delta,
               floquet_phase=fl[0] if fl else None,
               floquet_trans=fl[1] if fl else None,
               phase_final=res["phase_final"], amp_max=res["amp_max"],
               l2_final=res["l2_final"],
               decorr_periods=decorrelation_time(res, sys),
               secs=time.time() - t0)
    g = lambda k, f, w=0: (format(out[k], f) if out[k] is not None
                           else "n/a".rjust(w))
    print(f"  {kind:11s} s={seed} mse={mse:.2e} dT/T={g('delta_T','+.3e',10)} "
          f"|lam2|={g('floquet_trans','.2e',8)} "
          f"phase_end={out['phase_final']:8.2f} amp_max={out['amp_max']:.2e} "
          f"decorr={out['decorr_periods']:7.1f}T ({out['secs']:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--periods", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-traj", type=int, default=64)
    ap.add_argument("--obs-noise", type=float, default=0.0,
                    help="observation noise when measuring the period from data")
    ap.add_argument("--task-id", type=int, default=None,
                    help="run only this index of the grid (Slurm array)")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--out", default="t0_fix_results.json")
    a = ap.parse_args()

    KINDS = ["free", "struct", "anchored", "anchored+sup"]
    grid = [(k, s) for k in KINDS for s in range(a.seeds)]
    if a.count:
        print(len(grid)); return

    sys_ = SL()
    Tm, rm = measured_invariants(sys_, noise=a.obs_noise)
    print(f"Stuart-Landau: T_true={sys_.T:.8f}  r*_true={sys_.r_star:.8f}")
    print(f"measurable from data (noise={a.obs_noise}): "
          f"T={Tm:.8f} (err {(Tm-sys_.T)/sys_.T:+.2e})  "
          f"r*={rm:.8f} (err {(rm-sys_.r_star)/sys_.r_star:+.2e})\n")

    todo = [grid[a.task_id]] if a.task_id is not None else grid
    if a.task_id is not None:
        print(f"task {a.task_id}: kind={todo[0][0]} seed={todo[0][1]}", flush=True)

    rows = []
    for kind, s in todo:
        rows.append(run(kind, sys_, a.n_traj, 64, s, a.epochs,
                        n_periods=a.periods, obs_noise=a.obs_noise))
    if a.task_id is None:
        for kind in KINDS:
            d = [abs(r["delta_T"]) for r in rows
                 if r["kind"] == kind and r["delta_T"] is not None]
            if d:
                print(f"  -> {kind}: median |dT/T| = {np.median(d):.3e}")
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
