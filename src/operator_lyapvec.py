"""
Where the weakest transverse contraction of a trained operator lives.

The Discussion's account of why residual operators keep transverse multipliers
near one is that a field operator's state has ~1e4 dimensions and its attractor
a handful, so most transverse directions are never probed by the training
trajectories and a residual map x + F(x) stays near the identity there. That
predicts something checkable on the saved models: the near-neutral transverse
direction of a residual operator should lie mostly OUTSIDE the subspace the
training data span, and (for an FNO, whose spectral layers act only on the
retained low Fourier modes) largely at wavenumbers above the retained modes.
A direct operator's weakest transverse direction carries no such prediction.

For each model: settle 300 periods from the first training snapshot (as in
operator_floquet.py), run the QR iteration for 10 + 30 periods, and take the
second Lyapunov vector v2 (the first is the phase direction). Report
  in_pod     fraction of |v2|^2 inside the span of the leading POD modes of
             the normalised training fields (8 and 32 modes)
  high_k     fraction of |v2|^2 at Fourier wavenumbers outside the operator's
             retained modes
and the same for v1 as a sanity check: the phase direction is tangent to the
cycle, which lies in the data, so in_pod for v1 should be close to one.

  python src/operator_lyapvec.py --only K25-b8 --out results/lyapvec
"""
import argparse, glob, json, os, time
import numpy as np
import torch

import param_wake  # noqa: F401  (float64 default)
from fno import FNO2d
from operator_floquet import load_fields


@torch.no_grad()
def lyap_vectors(model, x, steps, k, eps, burn):
    g = torch.Generator(device="cpu").manual_seed(0)
    d = x[0].numel()
    Q = torch.randn(d, k, generator=g, dtype=torch.float64).to(x.device)
    Q[:, 0] = (model(x) - x).reshape(-1)
    Q, _ = torch.linalg.qr(Q)
    shape = x.shape[1:]
    for n in range(burn + steps):
        y = model(torch.cat([x, x + eps * Q.T.reshape(k, *shape)], 0))
        if not torch.isfinite(y).all():
            return None
        Q, _ = torch.linalg.qr(((y[1:] - y[:1]) / eps).reshape(k, d).T)
        x = y[:1]
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/seglen_v2")
    ap.add_argument("--out", default="results/lyapvec")
    ap.add_argument("--only", default="", help="comma-separated substrings of tags")
    ap.add_argument("--settle", type=float, default=300.0)
    ap.add_argument("--periods", type=float, default=30.0)
    ap.add_argument("--burn", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    keys = [s for s in a.only.split(",") if s]
    tags = sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{a.res}/*.json"))
    tags = [t for t in tags if "-unet" not in t and os.path.exists(f"{a.res}/models/{t}.pt")
            and (not keys or any(s in t for s in keys))]
    cache = {}
    for tag in tags:
        out = f"{a.out}/{tag}.json"
        if os.path.exists(out):
            continue
        t0 = time.time()
        j = json.load(open(f"{a.res}/{tag}.json"))
        cfg, T = j["config"], j["summary"]["T_snap"]
        key = (cfg["data"], cfg.get("Re"))
        if key not in cache:
            f = load_fields(cfg)
            f = f / float(np.abs(f).max())
            X = f.reshape(len(f), -1).astype(np.float64)
            Xc = X - X.mean(0)
            # POD of the normalised training fields: the directions the data span
            _, sv, Vt = np.linalg.svd(Xc[:: max(1, len(Xc) // 1500)], full_matrices=False)
            cache[key] = (torch.from_numpy(f[:1]).to(dev),
                          torch.from_numpy(Vt[:32].T.copy()).to(dev), f.shape[1:])
        x0, P, shp = cache[key]
        model = FNO2d(width=cfg["width"], modes=tuple(cfg["fmodes"]), depth=cfg["depth"],
                      ch=shp[0], residual=(cfg["variant"] == "residual")).to(dev)
        model.load_state_dict(torch.load(f"{a.res}/models/{tag}.pt", map_location=dev))
        model.eval()
        rec = dict(tag=tag, variant=cfg["variant"], noise=cfg.get("noise", 0.0))
        x = x0.to(torch.float64)
        ok = True
        with torch.no_grad():
            for n in range(int(a.settle * T)):
                x = model(x)
                if n % 200 == 0 and (not torch.isfinite(x).all() or x.abs().max() > 1e8):
                    ok = False; break
        Q = lyap_vectors(model, x, int(a.periods * T), 2, 1e-6, int(a.burn * T)) if ok else None
        if Q is None:
            rec["status"] = "diverged"
        else:
            m1, m2 = cfg["fmodes"]
            def frac(v):
                v = v / v.norm()
                pod8 = float((P[:, :8].T @ v).pow(2).sum())
                pod32 = float((P[:, :32].T @ v).pow(2).sum())
                F = torch.fft.rfft2(v.reshape(*shp), norm="ortho")
                H, W = shp[-2], shp[-1]
                ky = torch.fft.fftfreq(H, d=1.0 / H, device=v.device).abs().view(H, 1)
                kx = torch.arange(W // 2 + 1, device=v.device).view(1, -1)
                outside = ((ky >= m1) | (kx >= m2)).to(v.dtype)
                e = F.abs().pow(2)
                # an rfft2 stores each kx > 0 once for the pair +-kx; count it twice
                wgt = torch.full((H, W // 2 + 1), 2.0, dtype=e.dtype, device=e.device)
                wgt[:, 0] = 1.0
                if W % 2 == 0:
                    wgt[:, -1] = 1.0
                hk = float((e * wgt * outside).sum() / (e * wgt).sum())
                return dict(in_pod8=pod8, in_pod32=pod32, high_k=hk)
            rec.update(status="ok", v1=frac(Q[:, 0]), v2=frac(Q[:, 1]))
        rec["secs"] = time.time() - t0
        json.dump(rec, open(out, "w"), indent=1)
        if rec["status"] == "ok":
            print(f"  {tag:30s} v2: pod8 {rec['v2']['in_pod8']:.3f} pod32 {rec['v2']['in_pod32']:.3f} "
                  f"high-k {rec['v2']['high_k']:.3f} | v1 pod8 {rec['v1']['in_pod8']:.3f} "
                  f"[{rec['secs']:.0f}s]", flush=True)
        else:
            print(f"  {tag:30s} {rec['status']}", flush=True)


if __name__ == "__main__":
    main()
