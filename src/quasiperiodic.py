"""
Where does the single-period argument stop holding?

Everything so far assumes a limit cycle: one neutral Floquet direction, one
period, one scalar delta. A quasi-periodic attractor has two incommensurate
frequencies, two neutral directions and no single period, so the diagnostic as
stated cannot apply. This draws the boundary rather than asserting it.

System: two Stuart-Landau oscillators with DIFFUSIVE coupling eps*(other-self),
which is what produces Arnold tongues. A driving term eps*other does not lock in
this range: the frequency ratio slides from 1.618 to 1.05 without ever pinning.

The ON-CYCLE frequencies are set to the golden ratio, the hardest to lock. Note
that Omega = w - beta*mu rather than w, so w2 must be solved for; setting
w2/w1 = golden instead puts Omega2/Omega1 at 1.4944, essentially 3:2, placing
the system on a low-order resonance.

Sweeping eps carries the system across the locking transition, measured by the
rate at which the two oscillators' phases separate:

    eps    0.0    0.1    0.2    0.3  |  0.4    0.5    0.8
    drift  0.77   0.75   0.66   0.49 |  0.00   0.00   0.00
           <------ quasi-periodic ---|--- phase-locked --->

so the boundary sits between 0.3 and 0.4, and both regimes are available in one
system with one code path.

Predictions:

  1. In the locked regime the single-scalar law holds as before.
  2. In the quasi-periodic regime it does not, and the error is governed by two
     frequency errors, generalising to
         err = sqrt( sum_k w_k * 4 sin^2(pi delta_k N_k) )
     with w_k the energy fraction of component k.
  3. The usable horizon is set by the FASTEST-slipping component, so a
     diagnostic that measures only the dominant frequency overestimates it
     whenever the weaker component has the larger delta. That is the practical
     warning this section exists to deliver.
"""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from period import period_from_signal

torch.set_default_dtype(torch.float64)
GOLDEN = (1 + math.sqrt(5)) / 2


class CoupledSL:
    """Two Stuart-Landau oscillators with linear coupling. State is 4-D real."""

    def __init__(self, mu=0.5, w1=1.0, w2=None, beta=-0.5, eps=0.0):
        self.mu, self.w1 = mu, w1
        # The frequency ON the cycle is Omega = w - beta*r^2 = w - beta*mu, not
        # w. Setting w2/w1 = golden puts Omega2/Omega1 at 1.4944, essentially
        # 3:2 -- the system lands ON a low-order resonance, the opposite of what
        # the golden ratio was chosen for. Solve for w2 instead.
        self.w2 = (w2 if w2 is not None else
                   beta * mu + GOLDEN * (w1 - beta * mu))
        self.beta, self.eps = beta, eps

    def f(self, t, y):
        x1, y1, x2, y2 = y
        r1, r2 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2
        a1, b1 = self.mu - r1, self.w1 - self.beta * r1
        a2, b2 = self.mu - r2, self.w2 - self.beta * r2
        e = self.eps                    # diffusive coupling -> Arnold tongues
        return [a1 * x1 - b1 * y1 + e * (x2 - x1),
                b1 * x1 + a1 * y1 + e * (y2 - y1),
                a2 * x2 - b2 * y2 + e * (x1 - x2),
                b2 * x2 + a2 * y2 + e * (y1 - y2)]

    def phase_drift(self, traj, t):
        """
        Rate at which the two oscillators' phases separate. Exactly zero when
        phase-locked and O(1) on a torus. This is the reliable regime test:
        a ratio-versus-rational test flags the golden ratio as 8:5 at any
        useful tolerance, and a ratio-drift test cannot distinguish "pinned by
        locking" from "pinned because uncoupled".
        """
        p1 = np.unwrap(np.arctan2(traj[:, 1], traj[:, 0]))
        p2 = np.unwrap(np.arctan2(traj[:, 3], traj[:, 2]))
        d = p2 - p1
        h = len(d) // 2
        return float((d[-1] - d[h]) / (t[-1] - t[h]))

    def rollout(self, y0, t):
        s = solve_ivp(self.f, (t[0], t[-1]), y0, t_eval=t,
                      rtol=1e-12, atol=1e-12, method="DOP853")
        return s.y.T


# ------------------------------------------------------- frequency extraction
def peak_frequencies(t, s, n=2, min_sep=0.15):
    """Dominant spectral peaks, parabolically refined, with real separation."""
    dt = float(np.median(np.diff(t)))
    x = np.asarray(s, float) - np.mean(s)
    P = np.abs(np.fft.rfft(x)) ** 2
    fr = np.fft.rfftfreq(len(x), d=dt) * 2 * math.pi
    P[0] = 0.0
    out = []
    for k in np.argsort(P)[::-1]:
        if fr[k] <= 1e-9 or 0 >= k or k >= len(P) - 1:
            continue
        a, b, c = P[k - 1], P[k], P[k + 1]
        den = a - 2 * b + c
        w = fr[k] + (0.5 * (a - c) / den if abs(den) > 0 else 0) * (fr[1] - fr[0])
        if all(abs(w - u) > min_sep * min(w, u) for u in out):
            out.append(float(w))
        if len(out) == n:
            break
    return sorted(out)


def _design(t, w, orders):
    cols = [np.ones_like(t)]
    for (m, n) in orders:
        ang = (m * w[0] + n * w[1]) * t
        cols += [np.cos(ang), np.sin(ang)]
    return np.stack(cols, 1)


def two_freq_fit(t, s, w0, max_order=3):
    """
    Least squares over combination tones m*w1 + n*w2, optimised over (w1, w2).
    The natural generalisation of the single-fundamental Fourier fit: on a torus
    the spectrum lives on the lattice of integer combinations, not on harmonics
    of one frequency.
    """
    t = np.asarray(t, float); t = t - t.mean()
    s = np.asarray(s, float)
    orders = [(m, n) for m in range(0, max_order + 1)
              for n in range(-max_order, max_order + 1)
              if not (m == 0 and n <= 0)]

    def obj(w):
        if w[0] <= 0 or w[1] <= 0:
            return 1e9
        P = _design(t, w, orders)
        with np.errstate(all="ignore"):
            r = P @ np.linalg.lstsq(P, s, rcond=None)[0] - s
        return float(np.linalg.norm(r))

    r = minimize(obj, np.asarray(w0, float), method="Nelder-Mead",
                 options=dict(xatol=1e-12, fatol=1e-14, maxiter=4000))
    return np.sort(r.x), float(r.fun)


def component_weights(t, s, w, max_order=3):
    """Energy fraction carried by the w1 and w2 families."""
    t = np.asarray(t, float); t = t - t.mean()
    orders = [(m, n) for m in range(0, max_order + 1)
              for n in range(-max_order, max_order + 1)
              if not (m == 0 and n <= 0)]
    P = _design(t, w, orders)
    with np.errstate(all="ignore"):
        c = np.linalg.lstsq(P, np.asarray(s, float), rcond=None)[0]
    e1 = e2 = 0.0
    for i, (m, n) in enumerate(orders):
        p = c[1 + 2 * i] ** 2 + c[2 + 2 * i] ** 2
        if abs(m) >= abs(n):
            e1 += p
        else:
            e2 += p
    tot = e1 + e2 + 1e-30
    return e1 / tot, e2 / tot


def second_family_energy(t, s, n_harm=8, tol=0.04):
    """
    Energy outside the dominant frequency's own harmonic family, relative to
    the dominant peak. A torus carries a genuine second family; a locked state
    does not.
    """
    dt = float(np.median(np.diff(t)))
    x = np.asarray(s, float) - np.mean(s)
    P = np.abs(np.fft.rfft(x)) ** 2
    fr = np.fft.rfftfreq(len(x), d=dt) * 2 * math.pi
    k1 = int(np.argmax(P[1:])) + 1
    f1 = fr[k1]
    mask = np.ones_like(P, bool)
    for h in range(1, n_harm + 1):
        mask &= np.abs(fr - h * f1) > tol * f1
    mask[:2] = False
    k2 = int(np.argmax(np.where(mask, P, 0)))
    return float(P[k2] / (P[k1] + 1e-300)), float(f1), float(fr[k2])


# ---------------------------------------------------------------------- model
def _mlp(d, width, depth):
    L, x = [], d
    for _ in range(depth):
        L += [nn.Linear(x, width), nn.Tanh()]; x = width
    return nn.Sequential(*L, nn.Linear(x, d))


class VF(nn.Module):
    def __init__(self, d=4, width=96, depth=3):
        super().__init__()
        self.net = _mlp(d, width, depth)

    def forward(self, x):
        return self.net(x)


def rk4(f, x, dt, n):
    out = [x]
    for _ in range(n):
        k1 = f(x); k2 = f(x + .5 * dt * k1)
        k3 = f(x + .5 * dt * k2); k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(x)
    return torch.stack(out)


def train(model, segs, dt, epochs, lr=3e-3, batch=64, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = segs.shape[1] - 1
    last = float("nan")
    for _ in range(epochs):
        i = torch.randperm(len(segs))[:batch]
        b = segs[i]
        opt.zero_grad()
        pred = rk4(model, b[:, 0], dt, n)
        loss = ((pred - b.transpose(0, 1)) ** 2).mean()
        last = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); sch.step()
    return last


def predicted_err(deltas, weights, w_true, t):
    """sqrt( sum_k w_k * 4 sin^2(dtheta_k / 2) ),  dtheta_k = w_k_true * d_k * t"""
    tot = 0.0
    for d, wt, w in zip(deltas, weights, w_true):
        tot += wt * 4 * math.sin(0.5 * w * d * t) ** 2
    return math.sqrt(max(tot, 0.0))


def measure_state(t, obs, traj, sysm):
    """Regime, frequencies and component weights of one trajectory."""
    drift = sysm.phase_drift(traj, t)
    lk = abs(drift) < 1e-9
    e2, f1, f2 = second_family_energy(t, obs)
    if lk:
        T = period_from_signal(t, obs)
        return dict(locked=True, drift=drift, e2=e2,
                    w=[2 * math.pi / T], weights=[1.0])
    w0 = peak_frequencies(t, obs, 2)
    if len(w0) < 2:
        w0 = [f1, f2]
    w, _ = two_freq_fit(t, obs, w0)
    return dict(locked=False, drift=drift, e2=e2,
                w=list(map(float, w)),
                weights=list(map(float, component_weights(t, obs, w))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps-list", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.8])
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--seg-len", type=int, default=40)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--data-time", type=float, default=400.0)
    ap.add_argument("--eval-time", type=float, default=20000.0)
    ap.add_argument("--seed-list", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="quasiperiodic.json")
    a = ap.parse_args()

    rows = []
    for eps in a.eps_list:
        sysm = CoupledSL(eps=eps)
        y0 = sysm.rollout([0.7, 0.0, 0.7, 0.3], np.arange(0, 900, a.dt))[-1]
        t = np.arange(0, a.data_time, a.dt)
        traj = sysm.rollout(y0, t)
        obs = traj[:, 0] + traj[:, 2]
        st = measure_state(t, obs, traj, sysm)
        reg = "LOCKED" if st["locked"] else "quasi-periodic"
        print("\neps=%.2f  %s  drift=%.2e  w=%s  weights=%s"
              % (eps, reg, st["drift"], np.round(st["w"], 6),
                 np.round(st["weights"], 3)), flush=True)

        t_ev = np.arange(0, a.eval_time, a.dt)
        ref = sysm.rollout(y0, t_ev)

        scale = float(np.abs(traj).max())
        A = traj / scale
        segs = torch.tensor(np.stack(
            [A[i:i + a.seg_len + 1] for i in range(len(A) - a.seg_len - 1)]))

        for seed in a.seed_list:
            t0 = time.time()
            torch.manual_seed(seed)
            model = VF()
            mse = train(model, segs, a.dt, a.epochs, seed=seed)
            with torch.no_grad():
                tr = rk4(model, torch.tensor(A[0]).unsqueeze(0), a.dt,
                         len(t_ev) - 1)[:, 0, :].numpy() * scale
            good = bool(np.isfinite(tr).all() and np.abs(tr).max() < 1e3)
            row = dict(eps=eps, seed=seed, train_mse=mse, diverged=not good)
            row.update(st)
            if good:
                k = len(t)
                obs_p = tr[:, 0] + tr[:, 2]
                stp = measure_state(t_ev[:k], obs_p[:k], tr[:k], sysm)
                nw = min(len(st["w"]), len(stp["w"]))
                dl = [(stp["w"][i] - st["w"][i]) / st["w"][i]
                      for i in range(nw)]
                T_p = period_from_signal(t_ev[:k], obs_p[:k])
                T_t = period_from_signal(t[:k], obs[:k])
                d_single = (T_p - T_t) / T_t

                errs = []
                for frac in [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]:
                    j = int(frac * (len(t_ev) - 1))
                    lo = max(0, j - int(60 / a.dt))
                    e = (np.linalg.norm(tr[lo:j] - ref[lo:j]) /
                         (np.linalg.norm(ref[lo:j]) + 1e-30))
                    errs.append(dict(
                        t=float(t_ev[j]), err=float(e),
                        pred_multi=predicted_err(dl, st["weights"][:nw],
                                                 st["w"][:nw], t_ev[j]),
                        pred_single=predicted_err(
                            [d_single], [1.0], [2 * math.pi / T_t], t_ev[j])))
                row.update(w_pred=stp["w"], delta=dl,
                           delta_single=float(d_single), errs=errs)
                ds = " ".join("%+.2e" % x for x in dl)
                print("  seed=%d mse=%.2e delta=[%s] single=%+.2e (%.0fs)"
                      % (seed, mse, ds, d_single, time.time() - t0), flush=True)
            else:
                print("  seed=%d mse=%.2e DIVERGED" % (seed, mse), flush=True)
            rows.append(row)

    json.dump(rows, open(a.out, "w"), indent=1)
    print("\nwrote " + a.out)


if __name__ == "__main__":
    main()
