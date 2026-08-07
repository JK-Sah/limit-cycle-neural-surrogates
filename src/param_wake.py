"""
Parametric wake: one latent operator across Reynolds number.

This joins the two halves of the argument. The parametric van der Pol study
showed delta inflating ~100x once omega has to be interpolated rather than
memorised, but on an ODE. The single-Re cylinder wake showed delta staying at
~2e-5, but at one operating point. The open question is whether a parametric
FLOW surrogate inherits the van der Pol failure.

Strouhal varies along a known curve here (Williamson 1989:
St = -3.3265/Re + 0.1816 + 1.6e-4 Re), so omega(Re) has an external reference
that the van der Pol sweep did not.

  train  Re = 80, 100, 120, 140, 160, 180
  interp Re = 90, 110, 130, 150, 170
  extrap Re = 200, 220

Method note: the POD basis is built from TRAINING Re only. Pooling all Re into
the basis would leak held-out information into the representation, and the
interpolation numbers would not mean what they claim.
"""
import argparse, glob, json, math, os, re, time
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import hilbert

torch.set_default_dtype(torch.float64)

TRAIN_RE = [80, 100, 120, 140, 160, 180]
INTERP_RE = [90, 110, 130, 150, 170]
EXTRAP_RE = [200, 220]


def st_williamson(Re):
    return -3.3265 / Re + 0.1816 + 1.6e-4 * Re


def period_from_signal(t, s, edge_frac=0.1):
    s = np.asarray(s, float); s = s - s.mean()
    ph = np.unwrap(np.angle(hilbert(s)))
    k = max(1, int(edge_frac * len(s)))
    return float(2 * math.pi / abs(np.polyfit(t[k:-k], ph[k:-k], 1)[0]))


# ------------------------------------------------------------------------ POD
def build_basis(paths, r, per_re_sub=300, seed=0):
    """
    Shared POD basis from a subsample of the TRAINING Reynolds numbers.

    Subsampling matters for cost, not accuracy: the Gram matrix is n x n, and
    pooling every snapshot from six Re would make n ~ 9000 and the matmul
    ~3e12 flops. A few hundred snapshots per Re span the same subspace.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for p in paths:
        d = np.load(p)
        s = d["snaps"]
        idx = rng.choice(len(s), size=min(per_re_sub, len(s)), replace=False)
        chunks.append(s[np.sort(idx)].reshape(len(idx), -1).astype(np.float64))
        del d
    X = np.concatenate(chunks, 0)
    mean = X.mean(0)
    X -= mean
    with np.errstate(over="ignore", under="ignore", invalid="ignore",
                     divide="ignore"):
        G = X @ X.T
    w, V = np.linalg.eigh(G)
    w, V = np.clip(w[::-1], 0, None), V[:, ::-1]
    modes = (X.T @ V[:, :r]) / np.sqrt(np.maximum(w[:r], 1e-30))   # [dof, r]
    return modes, mean, w / w.sum()


def project(path, modes, mean):
    d = np.load(path)
    s = d["snaps"].reshape(len(d["snaps"]), -1).astype(np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore",
                     divide="ignore"):
        c = (s - mean) @ modes          # BLAS padding-lane flags, not real
    return c, d["t"].astype(float)


# --------------------------------------------------------------------- models
def _mlp(d_in, d_out, width, depth):
    L, d = [], d_in
    for _ in range(depth):
        L += [nn.Linear(d, width), nn.Tanh()]; d = width
    return nn.Sequential(*L, nn.Linear(d, d_out))


class FreeParametric(nn.Module):
    def __init__(self, r, width=128, depth=3):
        super().__init__()
        self.net = _mlp(r + 1, r, width, depth)

    def forward(self, a, mu):
        return self.net(torch.cat([a, mu], -1))


def fit_omega(mode, mu_train, om_train, re_train):
    """
    Fit a CONSTRAINED interpolant to the measured training frequencies.

    This replaces supervising a free MLP at the sampled parameters. That version
    pinned omega where it was supervised and wiggled between: on the wake it gave
    median interpolation delta 1.96e-3 with a worst case of 1.42e-1, far worse
    than an unconstrained baseline. Nothing forced the interpolant to be smooth.

    Because the rollout period is exactly 2*pi/omega(mu), a frozen interpolant
    makes delta computable in closed form BEFORE any network is trained -- the
    frequency interpolant can be chosen by cross-validation on measured
    frequencies alone.
    """
    if mode == "poly4":
        return np.polyfit(mu_train, om_train, 4)
    if mode == "williamson":
        # St = a/Re + b + c Re (Williamson 1989); omega inherits the form
        A = np.stack([1.0 / re_train, np.ones_like(re_train), re_train], 1)
        return np.linalg.lstsq(A, om_train, rcond=None)[0]
    raise ValueError(mode)


class PhaseAnchoredParametric(nn.Module):
    """
    phi' = omega(mu),  a = Z(phi, mu) + s,  s' = g(s, phi, mu)

    omega is a FROZEN interpolant fitted to measured training frequencies, not a
    learned function. The rollout period is exactly 2*pi/omega(mu), so delta at
    a held-out parameter is exactly the interpolation error of that curve.

    omega_mode:
      poly4       degree-4 least squares in the normalised parameter
      williamson  a/Re + b + c Re, the physical form of the Strouhal curve
      mlp         the failed ablation: a free MLP supervised at training Re only
    """
    def __init__(self, r, n_harm=8, width=128, depth=3, w_head=64,
                 omega_mode="williamson", omega_fit=None):
        super().__init__()
        self.r, self.n_harm, self.omega_mode = r, n_harm, omega_mode
        self.register_buffer("harm", torch.arange(
            n_harm + 1, dtype=torch.get_default_dtype()))
        if omega_mode == "mlp":
            self.omega_net = _mlp(1, 1, w_head, 2)
        else:
            self.register_buffer("om_coef",
                                 torch.tensor(np.asarray(omega_fit)))
        self.coef_net = _mlp(1, 2 * (n_harm + 1) * r, width, depth)
        self.g = _mlp(r + 3, r, width, depth)

    def omega(self, mu):
        if self.omega_mode == "mlp":
            return nn.functional.softplus(self.omega_net(mu)) + 1e-3
        if self.omega_mode == "poly4":
            out = torch.zeros_like(mu)
            for c in self.om_coef:
                out = out * mu + c
            return out
        Re = 130.0 + 70.0 * mu                      # undo the normalisation
        a, b, c = self.om_coef
        return a / Re + b + c * Re

    def coefs(self, mu):
        """Fourier coefficients of the loop at this parameter."""
        c = self.coef_net(mu)
        return c.view(*c.shape[:-1], 2, self.n_harm + 1, self.r)

    def Z(self, phi, mu=None, c=None):
        if c is None:
            c = self.coefs(mu)
        ang = phi.unsqueeze(-1) * self.harm
        return ((ang.cos().unsqueeze(-1) * c[..., 0, :, :]).sum(-2)
                + (ang.sin().unsqueeze(-1) * c[..., 1, :, :]).sum(-2))

    def decode(self, phi, s, mu=None, c=None):
        return self.Z(phi, mu, c) + s

    def forward(self, state, mu):
        phi, s = state[..., :1], state[..., 1:]
        sdot = self.g(torch.cat([s, phi.cos(), phi.sin(), mu], -1))
        return torch.cat([self.omega(mu), sdot], -1)

    def init_state(self, a, phi0, mu):
        return torch.cat([phi0, a - self.Z(phi0.squeeze(-1), mu)], -1)


def rk4(f, x, mu, dt, n):
    out = [x]
    for _ in range(n):
        k1 = f(x, mu); k2 = f(x + .5 * dt * k1, mu)
        k3 = f(x + .5 * dt * k2, mu); k4 = f(x + dt * k3, mu)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(x)
    return torch.stack(out)


@torch.no_grad()
def rollout_delta(model, kind, mu, T_true_model, dt, n_periods, a0):
    m = torch.tensor([[mu]])
    n = int(n_periods * T_true_model / dt)
    if kind == "free":
        tr = rk4(model, a0.unsqueeze(0), m, dt, n)[:, 0, :].numpy()
    else:
        c = model.coefs(m)                  # mu is fixed along this rollout
        st = model.init_state(a0.unsqueeze(0), torch.zeros(1, 1), m)
        tj = rk4(model, st, m, dt, n)
        tr = model.decode(tj[..., 0], tj[..., 1:], c=c)[:, 0, :].numpy()
    if not np.isfinite(tr).all():
        return None, None
    t = np.arange(len(tr)) * dt
    h = len(tr) // 2
    T_mod = period_from_signal(t[h:], tr[h:, 0])
    return (T_mod - T_true_model) / T_true_model, T_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/resweep")
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--seg-len", type=int, default=25)
    ap.add_argument("--periods", type=int, default=100)
    ap.add_argument("--seed-list", type=int, nargs="+", default=[0])
    ap.add_argument("--free-widths", type=int, nargs="+", default=[128, 200, 288])
    ap.add_argument("--harmonics", type=int, default=8)
    ap.add_argument("--w-sup", type=float, default=1e3,
                    help="only used by the failed --omega-mode mlp ablation")
    ap.add_argument("--omega-modes", nargs="+",
                    default=["williamson", "poly4"],
                    choices=["williamson", "poly4", "mlp"])
    ap.add_argument("--out", default="param_wake.json")
    a = ap.parse_args()

    allre = TRAIN_RE + INTERP_RE + EXTRAP_RE
    pth = {R: os.path.join(a.data, f"Re{R}_snaps.npz") for R in allre}
    sig = {R: os.path.join(a.data, f"Re{R}_signal.npz") for R in allre}
    missing = [R for R in allre if not os.path.exists(pth[R])]
    if missing:
        raise SystemExit(f"missing snapshot files for Re={missing}")

    # reference periods, in lattice steps, from the wake probe
    T_lat, St_meas = {}, {}
    for R in allre:
        d = np.load(sig[R])
        t, v = d["t"], d["v"]
        m = t > 0.5 * t.max()
        T_lat[R] = period_from_signal(t[m], v[m])
        St_meas[R] = 24.0 / (0.1 * T_lat[R])
    print("reference periods (wake probe) vs Williamson correlation:")
    for R in allre:
        tag = ("train" if R in TRAIN_RE else
               "interp" if R in INTERP_RE else "extrap")
        print(f"  Re={R:>4} [{tag:>6}] T={T_lat[R]:9.3f}  St={St_meas[R]:.5f}"
              f"  ref={st_williamson(R):.5f}"
              f"  ({(St_meas[R]-st_williamson(R))/st_williamson(R)*100:+.2f}%)")

    print(f"\nbuilding shared POD basis from the {len(TRAIN_RE)} training Re "
          f"only (held-out Re must not enter the basis)...")
    modes, mean, energy = build_basis([pth[R] for R in TRAIN_RE], a.modes)
    print(f"  energy: first 2 {energy[:2].sum()*100:.2f}%, "
          f"first {a.modes} {energy[:a.modes].sum()*100:.3f}%")

    coef, tsnap = {}, {}
    for R in allre:
        coef[R], tsnap[R] = project(pth[R], modes, mean)
    dt_lat = float(np.median(np.diff(tsnap[TRAIN_RE[0]])))
    T_snap = {R: T_lat[R] / dt_lat for R in allre}       # period in snapshots

    # model time: one mid-range period is 2*pi, so omega ~ 1 across the sweep
    T_mid = float(np.median([T_snap[R] for R in TRAIN_RE]))
    dt = 2 * math.pi / T_mid
    T_model = {R: T_snap[R] * dt for R in allre}
    om = {R: 2 * math.pi / T_model[R] for R in allre}
    print(f"  dt={dt:.5f} model units/snapshot; omega spans "
          f"{min(om.values()):.4f} to {max(om.values()):.4f}")

    scale = float(np.abs(np.concatenate([coef[R] for R in TRAIN_RE])).max())
    mu = lambda R: (R - 130.0) / 70.0                     # normalised parameter

    segs, seg_mu, seg_phi0 = [], [], []
    for R in TRAIN_RE:
        A = coef[R] / scale
        for i in range(len(A) - a.seg_len - 1):
            segs.append(A[i:i + a.seg_len + 1])
            seg_mu.append(mu(R))
            seg_phi0.append(om[R] * i * dt)
    segs = torch.tensor(np.stack(segs))
    seg_mu = torch.tensor(seg_mu).unsqueeze(-1)
    seg_phi0 = torch.tensor(seg_phi0).unsqueeze(-1)
    print(f"  {len(segs)} segments of {a.seg_len} steps\n")

    mu_sup = torch.tensor([[mu(R)] for R in TRAIN_RE])
    om_sup = torch.tensor([[om[R]] for R in TRAIN_RE])

    # Fit the frozen frequency interpolants and report the delta each one
    # implies, analytically, before training anything. The anchored rollout
    # period is exactly 2*pi/omega(mu), so these are what the model must achieve.
    mu_tr = np.array([mu(R) for R in TRAIN_RE])
    om_tr = np.array([om[R] for R in TRAIN_RE])
    re_tr = np.array([float(R) for R in TRAIN_RE])
    fits, pred_delta = {}, {}
    print("frozen frequency interpolants, delta predicted in closed form:")
    for m_ in [x for x in a.omega_modes if x != "mlp"]:
        fits[m_] = fit_omega(m_, mu_tr, om_tr, re_tr)
        tmp = PhaseAnchoredParametric(a.modes, omega_mode=m_, omega_fit=fits[m_])
        with torch.no_grad():
            pr = {R: float(tmp.omega(torch.tensor([[mu(R)]]))) for R in allre}
        pred_delta[m_] = {R: (pr[R] - om[R]) / om[R] for R in allre}
        line = []
        for tag, grp in [("train", TRAIN_RE), ("interp", INTERP_RE),
                         ("extrap", EXTRAP_RE)]:
            d = np.median([abs(pred_delta[m_][R]) for R in grp])
            line.append(f"{tag} {d:.2e} ({1/(4*d):.0f}P)")
        print(f"  {m_:>11}: " + "  ".join(line))
    print()

    configs = ([("free", w) for w in a.free_widths]
               + [(f"anchored[{m_}]", 128) for m_ in a.omega_modes])
    rows = []
    for kind, width in configs:
        for seed in a.seed_list:
            t0 = time.time()
            torch.manual_seed(seed)
            if kind == "free":
                model = FreeParametric(a.modes, width=width)
                om_mode = None
            else:
                om_mode = kind[kind.index("[") + 1:-1]
                model = PhaseAnchoredParametric(
                    a.modes, n_harm=a.harmonics, width=width,
                    omega_mode=om_mode, omega_fit=fits.get(om_mode))
            npar = sum(p.numel() for p in model.parameters())
            opt = torch.optim.Adam(model.parameters(), lr=2e-3)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
            for _ in range(a.epochs):
                idx = torch.randperm(len(segs))[:128]
                b, bm, bp = segs[idx], seg_mu[idx], seg_phi0[idx]
                opt.zero_grad()
                if kind == "free":
                    pred = rk4(model, b[:, 0], bm, dt, a.seg_len)
                else:
                    st = model.init_state(b[:, 0], bp, bm)
                    tj = rk4(model, st, bm, dt, a.seg_len)
                    pred = model.decode(tj[..., 0], tj[..., 1:], bm)
                loss = ((pred - b.transpose(0, 1)) ** 2).mean()
                mse = loss.item()
                if om_mode == "mlp":
                    # only the failed ablation learns omega and needs this
                    loss = loss + a.w_sup * ((model.omega(mu_sup)
                                              - om_sup) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step(); sch.step()

            for R in allre:
                tag = ("train" if R in TRAIN_RE else
                       "interp" if R in INTERP_RE else "extrap")
                a0 = torch.tensor(coef[R][0] / scale)
                d, Tm = rollout_delta(model, kind, mu(R), T_model[R], dt,
                                      a.periods, a0)
                rows.append(dict(kind=kind, width=width, seed=seed, Re=R,
                                 split=tag, nparam=npar, train_mse=mse,
                                 delta=d, T_model_true=T_model[R], T_pred=Tm,
                                 delta_predicted=(
                                     pred_delta[om_mode][R]
                                     if om_mode in pred_delta else None)))
            cell = {}
            for tag in ["train", "interp", "extrap"]:
                ds = [abs(r["delta"]) for r in rows if r["kind"] == kind
                      and r["width"] == width and r["seed"] == seed
                      and r["split"] == tag and r["delta"] is not None]
                cell[tag] = f"{np.median(ds):.2e}" if ds else "diverged"
            print(f"  {kind:15s} w={width:<4d} s={seed} np={npar:>7d} "
                  f"mse={mse:.2e} train={cell['train']} "
                  f"interp={cell['interp']} extrap={cell['extrap']} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print()
    h = "{:>15} {:>5} {:>8} {:>11} {:>11} {:>11}".format(
        "model", "width", "params", "train", "interp", "extrap")
    print(h); print("-" * len(h))
    for kind, width in configs:
        g = [r for r in rows if r["kind"] == kind and r["width"] == width]
        c = []
        for tag in ["train", "interp", "extrap"]:
            ds = [abs(r["delta"]) for r in g
                  if r["split"] == tag and r["delta"] is not None]
            c.append(f"{np.median(ds):.2e}" if ds else "diverged")
        print("{:>15} {:>5} {:>8} {:>11} {:>11} {:>11}".format(
            kind, width, g[0]["nparam"], *c))

    json.dump(dict(rows=rows, T_lat=T_lat, St_meas=St_meas,
                   omega_fits={k: np.asarray(v).tolist()
                               for k, v in fits.items()},
                   T_model={str(k): v for k, v in T_model.items()},
                   energy=energy[:a.modes].tolist(), dt=dt,
                   train_re=TRAIN_RE, interp_re=INTERP_RE,
                   extrap_re=EXTRAP_RE), open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
