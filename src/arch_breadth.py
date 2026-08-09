"""
Is the period error specific to continuous-time neural ODEs, or does every
learned dynamics model have it?

So far only a tanh-MLP neural ODE has been tested. Two questions follow. Does
the pathology survive a change of architecture, and is it the model class or the
fitting procedure that costs the frequency?

An earlier claim that DMD beats the neural ODE by 37x is RETRACTED: it compared
DMD at 16 modes on one dataset against a neural ODE at 8 modes on another. In a
controlled comparison at 8 modes DMD is 11x worse, and at 16 modes it is far
better, so the answer depends on truncation. See the mode sweep.

Architectures, all on the same POD coefficients, same data, same delta
estimator, comparable parameter budgets:

  ode-mlp      continuous, RK4 rollout                    (the existing model)
  ode-deep     the same, deeper and wider
  ar-residual  discrete, a_{k+1} = a_k + f(a_k)     (what most of the field does)
  ar-direct    discrete, a_{k+1} = f(a_k)
  gru          recurrent latent stepper with hidden state
  linear-sgd   a_{k+1} = A a_k with A learned by Adam

The last one is the control that matters. It is exactly DMD's model class,
fitted by gradient descent instead of least squares. If it loses to DMD, the
frequency is being lost by the optimisation, not by the hypothesis class.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn

from param_wake import (TRAIN_RE, build_basis, project, period_from_signal)
from rom_wake import dmd_fit, dmd_period

torch.set_default_dtype(torch.float64)


def _mlp(d_in, d_out, width, depth):
    L, d = [], d_in
    for _ in range(depth):
        L += [nn.Linear(d, width), nn.Tanh()]; d = width
    return nn.Sequential(*L, nn.Linear(d, d_out))


class ODEMLP(nn.Module):
    kind = "continuous"

    def __init__(self, r, width=96, depth=3):
        super().__init__()
        self.net = _mlp(r, r, width, depth)

    def vf(self, a):
        return self.net(a)

    def step(self, a, dt):
        k1 = self.vf(a); k2 = self.vf(a + .5 * dt * k1)
        k3 = self.vf(a + .5 * dt * k2); k4 = self.vf(a + dt * k3)
        return a + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class ARResidual(nn.Module):
    kind = "discrete"

    def __init__(self, r, width=96, depth=3):
        super().__init__()
        self.net = _mlp(r, r, width, depth)

    def step(self, a, dt):
        return a + self.net(a)


class ARDirect(nn.Module):
    kind = "discrete"

    def __init__(self, r, width=96, depth=3):
        super().__init__()
        self.net = _mlp(r, r, width, depth)

    def step(self, a, dt):
        return self.net(a)


class GRULatent(nn.Module):
    kind = "recurrent"

    def __init__(self, r, width=96, depth=3):
        super().__init__()
        self.h = width
        self.cell = nn.GRUCell(r, width)
        self.out = nn.Linear(width, r)
        self._hid = None

    def reset(self, batch, device):
        self._hid = torch.zeros(batch, self.h, dtype=torch.get_default_dtype(),
                                device=device)

    def step(self, a, dt):
        self._hid = self.cell(a, self._hid)
        return a + self.out(self._hid)


class LinearSGD(nn.Module):
    """a_{k+1} = A a_k, A by Adam. DMD's hypothesis class, different fit."""
    kind = "discrete"

    def __init__(self, r, width=None, depth=None):
        super().__init__()
        self.A = nn.Parameter(torch.eye(r) + 1e-3 * torch.randn(r, r))

    def step(self, a, dt):
        return a @ self.A.T


ARCH = {"ode-mlp": (ODEMLP, dict(width=96, depth=3)),
        "ode-deep": (ODEMLP, dict(width=192, depth=4)),
        "ar-residual": (ARResidual, dict(width=96, depth=3)),
        "ar-direct": (ARDirect, dict(width=96, depth=3)),
        "gru": (GRULatent, dict(width=96)),
        "linear-sgd": (LinearSGD, dict())}


def rollout(model, a0, dt, n):
    if hasattr(model, "reset"):
        model.reset(a0.shape[0], a0.device)
    out = [a0]
    a = a0
    for _ in range(n):
        a = model.step(a, dt)
        out.append(a)
    return torch.stack(out)


def train(model, segs, dt, epochs, lr, batch=64, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = segs.shape[1] - 1
    last = float("nan")
    for _ in range(epochs):
        idx = torch.randperm(len(segs))[:batch]
        b = segs[idx]
        opt.zero_grad()
        pred = rollout(model, b[:, 0], dt, n)
        loss = ((pred - b.transpose(0, 1)) ** 2).mean()
        last = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); sch.step()
    return last


@torch.no_grad()
def measure_delta(model, a0, dt, T_true, n_periods):
    n = int(n_periods * T_true / dt)
    tr = rollout(model, a0.unsqueeze(0), dt, n)[:, 0, :].numpy()
    if not np.isfinite(tr).all() or np.abs(tr).max() > 1e6:
        return None
    t = np.arange(len(tr)) * dt
    h = len(tr) // 2
    return (period_from_signal(t[h:], tr[h:, 0]) - T_true) / T_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/resweep")
    ap.add_argument("--Re", type=int, default=100)
    ap.add_argument("--modes", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--seg-len", type=int, default=25)
    ap.add_argument("--periods", type=int, default=100)
    ap.add_argument("--seed-list", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="arch_breadth.json")
    a = ap.parse_args()

    snap = os.path.join(a.data, f"Re{a.Re}_snaps.npz")
    d = np.load(os.path.join(a.data, f"Re{a.Re}_signal.npz"))
    t, v = d["t"], d["v"]
    m = t > 0.5 * t.max()
    T_lat = period_from_signal(t[m], v[m])

    modes, mean, energy = build_basis([snap], a.modes, per_re_sub=600)
    coef, tsnap = project(snap, modes, mean)
    dt_lat = float(np.median(np.diff(tsnap)))
    T_snap = T_lat / dt_lat
    dt = 2 * math.pi / T_snap
    T_model = 2 * math.pi
    scale = float(np.abs(coef).max())
    A = coef / scale
    print(f"Re={a.Re}  T={T_lat:.3f} lattice steps  "
          f"POD first 2 {energy[:2].sum()*100:.2f}%, "
          f"first {a.modes} {energy[:a.modes].sum()*100:.3f}%")

    Am = dmd_fit(A)
    Td, mag = dmd_period(Am, dt, T_model)
    dmd_d = (Td - T_model) / T_model
    print(f"DMD reference: delta = {dmd_d:+.3e}  "
          f"({1/(4*abs(dmd_d)):.0f} P)  |lambda| = {mag:.8f}\n")

    segs = torch.tensor(np.stack(
        [A[i:i + a.seg_len + 1] for i in range(len(A) - a.seg_len - 1)]))
    print(f"{len(segs)} segments of {a.seg_len} steps\n")

    rows = [dict(arch="DMD (least squares)", seed=-1, nparam=a.modes ** 2,
                 train_mse=None, delta=float(dmd_d))]
    for name, (cls, kw) in ARCH.items():
        for seed in a.seed_list:
            t0 = time.time()
            torch.manual_seed(seed)
            model = cls(a.modes, **kw)
            npar = sum(p.numel() for p in model.parameters())
            mse = train(model, segs, dt, a.epochs, 3e-3, seed=seed)
            dl = measure_delta(model, torch.tensor(A[0]), dt, T_model,
                               a.periods)
            rows.append(dict(arch=name, seed=seed, nparam=npar,
                             train_mse=mse, delta=dl,
                             secs=time.time() - t0))
            print(f"  {name:12s} s={seed} np={npar:>7d} mse={mse:.2e} "
                  f"delta={'%+.3e' % dl if dl is not None else 'DIVERGED':>11} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print()
    hdr = "{:>20} {:>8} {:>11} {:>12} {:>11} {:>8}".format(
        "architecture", "params", "med mse", "med |delta|", "horizon", "stable")
    print(hdr); print("-" * len(hdr))
    order = ["DMD (least squares)"] + list(ARCH)
    for name in order:
        g = [r for r in rows if r["arch"] == name]
        ds = [abs(r["delta"]) for r in g if r["delta"] is not None]
        if not ds:
            print("{:>20} {:>8} {:>11} {:>12} {:>11} {:>8}".format(
                name, g[0]["nparam"], "-", "all diverged", "-",
                f"0/{len(g)}"))
            continue
        md = np.median(ds)
        ms = [r["train_mse"] for r in g if r["train_mse"] is not None]
        print("{:>20} {:>8} {:>11} {:>12.3e} {:>9.0f} P {:>4}/{:<3}".format(
            name, g[0]["nparam"],
            f"{np.median(ms):.2e}" if ms else "-",
            md, 1 / (4 * md), len(ds), len(g)))
    json.dump(dict(rows=rows, Re=a.Re, T_lattice=T_lat, dt=dt,
                   modes=a.modes,
                   energy_captured=float(energy[:a.modes].sum()),
                   energy_tail=float(energy[a.modes:a.modes + 4].sum()),
                   energy_spectrum=energy[:40].tolist()),
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
