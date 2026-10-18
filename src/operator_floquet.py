"""
Floquet structure of the trained operators themselves.

The phase law assumes the surrogate HAS a stable limit cycle: one multiplier at
unity (the phase), the rest strictly inside the unit circle. The 400-period
rollouts of the segment-length sweep showed that residual-parameterised
operators often do not -- they collapse, diverge late, or keep an orbit whose
period drifts -- while every direct-parameterised operator kept a cycle that
passes the split-half test. The hypothesis tested here: a residual network
x + f(x) starts every multiplier at 1 and training only has to move the ones
the loss sees, so transverse contraction stays weak; a direct network has no
such bias. If so, the residual operators' second Lyapunov exponent sits near
zero and the direct ones' well below it.

For an invariant circle of a map, the analogue of the Floquet multipliers is
the set of Lyapunov exponents along it: the leading one is ~0 (phase), the
next ones give the transverse contraction. They are estimated by the standard
QR (Benettin) method: settle the operator onto its own attractor from the
first training snapshot, then carry k tangent vectors forward with the
operator's Jacobian, re-orthonormalising every step and accumulating the log
of the diagonal of R. Jacobian-vector products come from forward differences:
the model is float64, so eps = 1e-6 on unit tangent vectors is accurate to
~1e-9 relative, and all k + 1 states go through the network as one batch.

Exponents are reported per step and as multipliers per period,
exp(lambda * T_snap), on the same footing as the oscillator's |lambda_2|.

  python src/operator_floquet.py --task 0 --ntasks 14
"""
import argparse, glob, json, math, os, time
import numpy as np
import torch

import param_wake  # noqa: F401  (sets float64 as the default dtype)
from fno import FNO2d


def load_fields(cfg):
    d = cfg["data"]
    if "Re" in cfg and os.path.isdir(d):           # wake, via fno.py
        f = np.load(os.path.join(d, f"Re{cfg['Re']}_snaps.npz"))["snaps"]
    else:
        f = np.load(d)["snaps"]
    f = f.astype(np.float32)
    if f.ndim == 3:
        f = f[:, None]
    return f


@torch.no_grad()
def lyapunov(model, x, steps, k, eps, seed, burn=0):
    """QR estimate of the k leading Lyapunov exponents per step.

    The first tangent vector starts along the orbit (F(x) - x), the rest at
    random; `burn` steps are run before accumulating. Both matter in ~1e4
    dimensions: a random start has a component ~d^-1/2 along the neutral
    direction, and the log of that misalignment (~ -5) biases a 200-step
    estimate of the phase exponent by -0.02 per step (observed).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = x[0].numel()
    Q = torch.randn(d, k, generator=g, dtype=torch.float64).to(x.device)
    Q[:, 0] = (model(x) - x).reshape(-1)
    Q, _ = torch.linalg.qr(Q)
    shape = x.shape[1:]
    sums = torch.zeros(k, dtype=torch.float64, device=x.device)
    for n in range(burn + steps):
        batch = torch.cat([x, x + eps * Q.T.reshape(k, *shape)], 0)
        y = model(batch)
        if not torch.isfinite(y).all():
            return None, n
        V = ((y[1:] - y[:1]) / eps).reshape(k, d).T     # J Q, d x k
        Q, R = torch.linalg.qr(V)
        if n >= burn:
            sums += torch.log(torch.abs(torch.diagonal(R)))
        x = y[:1]
    return (sums / steps).cpu().numpy(), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/seglen_v2")
    ap.add_argument("--out", default="results/floquet_ops")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--settle", type=float, default=300.0, help="periods before measuring")
    ap.add_argument("--periods", type=float, default=30.0, help="periods of tangent dynamics")
    ap.add_argument("--burn", type=float, default=10.0,
                    help="periods of tangent dynamics discarded before accumulating")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None, help="substring filter on tags")
    a = ap.parse_args()
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)

    tags = sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{a.res}/*.json"))
    tags = [t for t in tags if "-unet" not in t and os.path.exists(f"{a.res}/models/{t}.pt")]
    if a.only:
        tags = [t for t in tags if a.only in t]
    mine = tags[a.task::a.ntasks]
    print(f"task {a.task}/{a.ntasks}: {len(mine)} of {len(tags)} models", flush=True)
    cache = {}
    for tag in mine:
        out = f"{a.out}/{tag}.json"
        if os.path.exists(out):
            continue
        t0 = time.time()
        j = json.load(open(f"{a.res}/{tag}.json"))
        cfg, T = j["config"], j["summary"]["T_snap"]
        key = (cfg["data"], cfg.get("Re"))
        if key not in cache:
            f = load_fields(cfg)
            cache[key] = (torch.from_numpy(f[:1] / float(np.abs(f).max())).to(dev),
                          f.shape[1])
        x0, ch = cache[key]
        model = FNO2d(width=cfg["width"], modes=tuple(cfg["fmodes"]), depth=cfg["depth"],
                      ch=ch, residual=(cfg["variant"] == "residual")).to(dev)
        model.load_state_dict(torch.load(f"{a.res}/models/{tag}.pt", map_location=dev))
        model.eval()
        x = x0.to(torch.float64)
        rec = dict(tag=tag, T_snap=T, variant=cfg["variant"], seg_len=cfg["seg_len"],
                   delta=j["rows"][0]["delta"], delta_late=j["rows"][0].get("delta_late"),
                   split_late=j["rows"][0].get("split_late"))
        with torch.no_grad():
            amp0 = float(x.norm())
            ok = True
            for n in range(int(a.settle * T)):
                x = model(x)
                if n % 200 == 0 and (not torch.isfinite(x).all() or x.abs().max() > 1e8):
                    ok = False; break
            ok = ok and bool(torch.isfinite(x).all())
        if not ok:
            rec.update(status="diverged while settling")
        else:
            lam, n = lyapunov(model, x, int(a.periods * T), a.k, a.eps, seed=0,
                              burn=int(a.burn * T))
            if lam is None:
                rec.update(status=f"diverged during tangent run at step {n}")
            else:
                rec.update(status="ok", lyap_per_step=lam.tolist(),
                           multipliers_per_period=np.exp(lam * T).tolist(),
                           state_norm_ratio=float(x.norm()) / amp0)
        rec["secs"] = time.time() - t0
        json.dump(rec, open(out, "w"), indent=1)
        m = rec.get("multipliers_per_period")
        print(f"  {tag:28s} {rec['status']:10s} "
              f"{'' if m is None else ' '.join(f'{v:.3g}' for v in m)}  [{rec['secs']:.0f}s]",
              flush=True)


if __name__ == "__main__":
    main()
