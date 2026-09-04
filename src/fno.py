"""
Fourier neural operator on the cylinder wake, measured with the same delta.

Why this case is necessary. Every architecture in the breadth comparison acts
on POD coefficients: neural ODEs, autoregressive MLPs, a GRU, a single matrix,
DMD, operator inference. None of them is a neural OPERATOR, and neural
operators are what the field actually deploys on PDEs. A claim that long-horizon
fidelity is governed by a spectral property the training loss cannot see is
aimed squarely at that community, so it has to be demonstrated on their model
rather than adjacent to it.

The comparison is arranged so only the model changes:

  - the operator is trained on FIELDS, u(x, y, t) -> u(x, y, t + dt), because
    that is what an FNO is. Restricting it to eight POD coefficients would test
    a different object and would hand it the low-dimensional structure that the
    other latent models are given on purpose.
  - delta is measured by rolling the operator out, projecting the rolled-out
    fields onto the SAME POD basis, and fitting the period of coefficient 1
    with the same Fourier estimator. Identical measurement path, so the number
    is directly comparable to the table it joins.
  - training uses a multi-step rollout loss, not one-step regression. Section 4
    of the draft shows the objective horizon matters more than the model class
    -- one-step least squares beat by its own hypothesis class fitted over 25
    steps -- so a one-step FNO would confound architecture with objective and
    the comparison would be worthless.

Both the direct and residual parameterisations are run, mirroring ar-direct and
ar-residual, since that split mattered for the autoregressive models.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn

from param_wake import build_basis, project, period_from_signal


def _cdtype():
    """Complex dtype matching the repo-wide real default.

    param_wake sets float64 globally, as every other model file does, because
    the deltas being resolved reach 1e-8. Hardcoding complex64 here would make
    the operator the only single-precision model in the comparison and put its
    round-off above the quantity being measured.
    """
    return {torch.float32: torch.complex64,
            torch.float64: torch.complex128}[torch.get_default_dtype()]


class SpectralConv2d(nn.Module):
    """Truncated-spectrum convolution: rfft2, keep low modes, mix, irfft2."""

    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        s = 1.0 / (in_ch * out_ch)
        cd = _cdtype()
        # two blocks: positive and negative first-axis frequencies
        self.w1 = nn.Parameter(s * torch.rand(in_ch, out_ch, modes1, modes2,
                                              dtype=cd))
        self.w2 = nn.Parameter(s * torch.rand(in_ch, out_ch, modes1, modes2,
                                              dtype=cd))

    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                          dtype=xf.dtype, device=x.device)
        m1 = min(self.modes1, H // 2)
        m2 = min(self.modes2, W // 2 + 1)
        out[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(H, W), norm="ortho")


class FNO2d(nn.Module):
    def __init__(self, width=32, modes=(16, 12), depth=4, ch=2,
                 residual=True, grid=True):
        super().__init__()
        self.residual, self.grid = residual, grid
        self.lift = nn.Conv2d(ch + (2 if grid else 0), width, 1)
        self.spec = nn.ModuleList(
            [SpectralConv2d(width, width, *modes) for _ in range(depth)])
        self.point = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, ch, 1))

    def _with_grid(self, x):
        if not self.grid:
            return x
        B, _, H, W = x.shape
        gx = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        gy = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, gx, gy], 1)

    def forward(self, x):
        h = self.lift(self._with_grid(x))
        for s, p in zip(self.spec, self.point):
            h = torch.nn.functional.gelu(s(h) + p(h))
        y = self.proj(h)
        return x + y if self.residual else y


def train(model, data, seg_len, epochs, lr, batch, seed, noise=0.0,
          log_every=50):
    """Multi-step rollout loss on field segments, matching the other models.

    `noise` switches on pushforward-style noise injection, one of the three
    stability methods section 2 shows acts on the transverse direction. Running
    it here is deliberate: if injecting noise rescues the rollout from
    divergence but leaves delta where it was, that is the paper's thesis
    demonstrated on the architecture the field actually deploys -- amplitude
    stabilised, phase untouched.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(data) - seg_len
    last = float("nan")
    for ep in range(epochs):
        idx = torch.randperm(n)[:batch]
        seg = torch.stack([data[i:i + seg_len + 1] for i in idx])  # [B,K+1,C,H,W]
        x = seg[:, 0]
        loss = 0.0
        for k in range(seg_len):
            if noise:
                x = x + noise * torch.randn_like(x)
            x = model(x)
            loss = loss + torch.mean((x - seg[:, k + 1]) ** 2)
        loss = loss / seg_len
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        last = loss.detach().item()
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"    ep {ep:>5}/{epochs}  loss {last:.4e}", flush=True)
    return last


@torch.no_grad()
def measure_delta(model, x0, n_steps, modes, mean, T_snap, scale):
    """Roll the operator out, project onto the POD basis, fit the period.

    Projection and period fitting are exactly the calls the latent models go
    through, so the resulting delta sits on the same axis as the rest of the
    table rather than merely resembling it.
    """
    x = x0.unsqueeze(0)
    coefs = np.empty((n_steps, modes.shape[1]))
    for k in range(n_steps):
        x = model(x)
        f = x[0].numpy().reshape(-1).astype(np.float64) * scale
        if not np.isfinite(f).all() or np.abs(f).max() > 1e8:
            return None, k
        coefs[k] = (f - mean) @ modes
    t = np.arange(n_steps, dtype=float)
    h = n_steps // 2
    T_meas = period_from_signal(t[h:], coefs[h:, 0])
    return (T_meas - T_snap) / T_snap, n_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/resweep")
    ap.add_argument("--Re", type=int, default=100)
    ap.add_argument("--modes-pod", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--fmodes", type=int, nargs=2, default=[16, 12])
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--seg-len", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--periods", type=int, default=100)
    ap.add_argument("--variant", choices=["residual", "direct"], default="residual")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="pushforward noise injection during rollout training")
    ap.add_argument("--seed-list", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="fno.json")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    snap = os.path.join(a.data, f"Re{a.Re}_snaps.npz")
    d = np.load(os.path.join(a.data, f"Re{a.Re}_signal.npz"))
    t, v = d["t"], d["v"]
    m = t > 0.5 * t.max()
    T_lat = period_from_signal(t[m], v[m])

    pod_modes, pod_mean, energy = build_basis([snap], a.modes_pod, per_re_sub=600)
    _, tsnap = project(snap, pod_modes, pod_mean)
    dt_lat = float(np.median(np.diff(tsnap)))
    T_snap = T_lat / dt_lat                      # period in snapshot units

    z = np.load(snap)
    fields = z["snaps"].astype(np.float32)       # [N, 2, H, W]
    scale = float(np.abs(fields).max())
    data = torch.from_numpy(fields / scale).to(torch.get_default_dtype())
    n_steps = int(a.periods * T_snap)

    print(f"Re={a.Re}  field {tuple(fields.shape[1:])}  N={len(fields)}  "
          f"T={T_snap:.2f} snaps/period  rollout={n_steps} steps  "
          f"POD {a.modes_pod} modes = {energy[:a.modes_pod].sum()*100:.3f}%",
          flush=True)

    rows = []
    for seed in a.seed_list:
        t0 = time.time()
        model = FNO2d(width=a.width, modes=tuple(a.fmodes), depth=a.depth,
                      residual=(a.variant == "residual"))
        npar = sum(p.numel() for p in model.parameters())
        print(f"  seed {seed}  variant={a.variant}  params={npar}", flush=True)
        mse = train(model, data, a.seg_len, a.epochs, a.lr, a.batch, seed,
                    noise=a.noise)
        dl, reached = measure_delta(model, data[0], n_steps, pod_modes,
                                    pod_mean, T_snap, scale)
        rows.append(dict(arch=f"fno-{a.variant}", seed=seed, nparam=npar,
                         noise=a.noise,
                         train_mse=mse, delta=dl, steps_reached=reached,
                         secs=time.time() - t0))
        hz = "DIVERGED" if dl is None else f"{1/(4*abs(dl)):.0f} P"
        print(f"  seed {seed}  mse={mse:.3e}  "
              f"delta={'DIVERGED' if dl is None else '%+.3e' % dl}  {hz}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    ds = [abs(r["delta"]) for r in rows if r["delta"] is not None]
    summary = dict(Re=a.Re, T_snap=T_snap, variant=a.variant,
                   med_delta=float(np.median(ds)) if ds else None,
                   spread=(max(ds) / min(ds)) if len(ds) > 1 and min(ds) > 0 else None,
                   n_stable=len(ds), n_seeds=len(rows))
    print("\n" + json.dumps(summary, indent=1))
    json.dump(dict(rows=rows, summary=summary, config=vars(a)),
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
