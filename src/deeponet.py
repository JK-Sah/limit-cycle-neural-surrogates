"""
DeepONet on the cylinder wake, measured with the same delta.

The companion to fno.py. Fourier neural operators and DeepONets are the two
architectures the operator-learning literature actually builds on, and they
factorise the problem differently: an FNO mixes information globally through a
truncated spectrum, while a DeepONet splits into a branch that reads the input
function and a trunk that supplies a spatial basis. If the period error is a
property of the problem rather than of a particular inductive bias, both should
show it, and showing it on only one of them would leave the obvious objection
open.

The trunk is worth a note. Because the query grid is fixed here, the trunk
evaluates to p spatial fields that the branch coefficients weight -- which is
to say the network learns its own basis and then predicts coefficients in it.
That is structurally close to what the POD-plus-latent-dynamics models in the
breadth table do by construction, with the basis learned instead of computed.
Whether a learned basis changes delta is exactly the sort of thing the
comparison should answer rather than assume.

Training and delta measurement are imported from fno.py rather than
reimplemented, so the objective horizon, the rollout length, the POD
projection and the period estimator are identical by construction and not by
inspection.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn

from param_wake import build_basis, project, period_from_signal
from fno import train, measure_delta


def fourier_features(xy, K):
    """Axis-aligned Fourier embedding of coordinates, plus the raw pair.

    Without this the trunk cannot do its job. A plain MLP on (x, y) has a strong
    spectral bias toward low frequencies, and the wake's modes are not low
    frequency: fitting a single shedding-like field
    sin(2*pi*6 x) cos(2*pi*2 y) directly reaches MSE 2.2e-1 against a target
    variance of 0.251, which is to say it explains almost none of it. The same
    MLP on these features reaches 2.9e-5. The first version of this file used
    raw coordinates and trained 4700x worse than the FNO on identical data --
    that gap was the trunk, not the architecture.

    Deterministic integer frequencies are used rather than random Gaussian
    features so the basis is reproducible across seeds and the only thing
    varying between runs is the initialisation.
    """
    ks = torch.arange(1, K + 1, dtype=xy.dtype, device=xy.device)
    a = 2 * math.pi * xy[:, 0:1] * ks
    b = 2 * math.pi * xy[:, 1:2] * ks
    return torch.cat([xy, torch.sin(a), torch.cos(a),
                      torch.sin(b), torch.cos(b)], 1)


class DeepONet(nn.Module):
    def __init__(self, H, W, ch=2, p=128, branch_width=256, trunk_width=256,
                 trunk_depth=4, n_freq=16, residual=True, field_bias=False):
        super().__init__()
        self.H, self.W, self.ch, self.p = H, W, ch, p
        self.n_freq = n_freq
        self.residual = residual

        # branch: strided conv encoder over the input field -> [B, ch, p]
        self.branch_conv = nn.Sequential(
            nn.Conv2d(ch, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(64, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
        )
        with torch.no_grad():
            n_flat = self.branch_conv(torch.zeros(1, ch, H, W)).numel()
        self.branch_head = nn.Sequential(
            nn.Flatten(), nn.Linear(n_flat, branch_width), nn.GELU(),
            nn.Linear(branch_width, ch * p))

        # trunk: MLP on Fourier-embedded coordinates -> p basis fields per channel
        layers, d = [], 2 + 4 * n_freq
        for _ in range(trunk_depth):
            layers += [nn.Linear(d, trunk_width), nn.GELU()]
            d = trunk_width
        layers += [nn.Linear(d, ch * p)]
        self.trunk = nn.Sequential(*layers)

        gx = torch.linspace(0, 1, H)
        gy = torch.linspace(0, 1, W)
        gg = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
        self.register_buffer("grid", gg)
        # A per-channel scalar is enough for the residual form, which only has
        # to represent an increment. The direct form has to reproduce the whole
        # field including the mean flow, and a scalar cannot carry that -- the
        # trunk basis would have to spend its capacity on a component that never
        # changes. `field_bias` gives it a learnable mean field instead, which
        # is what separates "cannot represent this" from "was not trained long
        # enough".
        self.bias = (nn.Parameter(torch.zeros(ch, H, W)) if field_bias
                     else nn.Parameter(torch.zeros(ch)))
        self.field_bias = field_bias

    def forward(self, x):
        B = x.shape[0]
        b = self.branch_head(self.branch_conv(x)).view(B, self.ch, self.p)
        t = self.trunk(fourier_features(self.grid.to(x.dtype), self.n_freq))
        t = t.view(self.H, self.W, self.ch, self.p).permute(2, 3, 0, 1)
        bias = (self.bias.unsqueeze(0) if self.field_bias
                else self.bias.view(1, -1, 1, 1))
        y = torch.einsum("bcp,cphw->bchw", b, t) + bias
        return x + y if self.residual else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/resweep")
    ap.add_argument("--Re", type=int, default=100)
    ap.add_argument("--modes-pod", type=int, default=8)
    ap.add_argument("--p", type=int, default=128)
    ap.add_argument("--branch-width", type=int, default=256)
    ap.add_argument("--trunk-width", type=int, default=256)
    ap.add_argument("--n-freq", type=int, default=16)
    ap.add_argument("--field-bias", action="store_true",
                    help="learnable mean field instead of a per-channel scalar")
    ap.add_argument("--trunk-depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seg-len", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--periods", type=int, default=100)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--variant", choices=["residual", "direct"], default="residual")
    ap.add_argument("--seed-list", type=int, nargs="+", default=[0])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="deeponet.json")
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
    T_snap = T_lat / dt_lat

    z = np.load(snap)
    fields = z["snaps"].astype(np.float32)
    scale = float(np.abs(fields).max())
    data = torch.from_numpy(fields / scale).to(torch.get_default_dtype())
    n_steps = int(a.periods * T_snap)
    H, W = fields.shape[2], fields.shape[3]

    print(f"Re={a.Re}  field {tuple(fields.shape[1:])}  N={len(fields)}  "
          f"T={T_snap:.2f} snaps/period  rollout={n_steps} steps", flush=True)

    rows = []
    for seed in a.seed_list:
        t0 = time.time()
        torch.manual_seed(seed)
        model = DeepONet(H, W, p=a.p, branch_width=a.branch_width,
                         trunk_width=a.trunk_width, trunk_depth=a.trunk_depth,
                         n_freq=a.n_freq, residual=(a.variant == "residual"),
                         field_bias=a.field_bias)
        npar = sum(q.numel() for q in model.parameters())
        print(f"  seed {seed}  variant={a.variant}  p={a.p}  params={npar}",
              flush=True)
        mse = train(model, data, a.seg_len, a.epochs, a.lr, a.batch, seed,
                    noise=a.noise)
        dl, reached = measure_delta(model, data[0], n_steps, pod_modes,
                                    pod_mean, T_snap, scale)
        rows.append(dict(arch=f"deeponet-{a.variant}", seed=seed, nparam=npar,
                         noise=a.noise, train_mse=mse, delta=dl,
                         steps_reached=reached, secs=time.time() - t0))
        hz = "DIVERGED" if dl is None else f"{1/(4*abs(dl)):.0f} P"
        print(f"  seed {seed}  mse={mse:.3e}  "
              f"delta={'DIVERGED' if dl is None else '%+.3e' % dl}  {hz}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    ds = [abs(r["delta"]) for r in rows if r["delta"] is not None]
    summary = dict(Re=a.Re, T_snap=T_snap, variant=a.variant,
                   med_delta=float(np.median(ds)) if ds else None,
                   n_stable=len(ds), n_seeds=len(rows))
    print("\n" + json.dumps(summary, indent=1))
    json.dump(dict(rows=rows, summary=summary, config=vars(a)),
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
