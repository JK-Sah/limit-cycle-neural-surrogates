"""
Fourier neural operator on the periodic Navier-Stokes attractor, measured with
the same delta.

This is the case the study has been missing. Every other system it measures is
either a two-dimensional ODE or a cylinder wake compressed to eight POD
coefficients; the objection that follows is that the period error might be an
artefact of low-dimensional latent dynamics rather than a property of the
problem. Here the operator acts on a 128 x 128 vorticity field of the
two-dimensional Navier-Stokes equations, in the forcing geometry of the Fourier
neural operator benchmark itself, at nu = 1.7e-3 -- the one viscosity where a
transition sweep found a well-posed limit cycle (split-half 4.8e-9 over 51
periods).

The benchmark's own settings, nu = 1e-3, 1e-4 and 1e-5, all sit past that
transition: a split-half refit there disagrees with itself by 1e-3 to 1e-1, so a
delta quoted on them would measure nothing. That is worth stating as a property
of the benchmark rather than a limitation here.

Training and measurement are imported from fno.py, so the objective horizon, the
rollout length, the POD projection and the period estimator are identical to the
wake experiments by construction rather than by inspection.
"""
import argparse, json, math, os, time
import numpy as np
import torch

from period import period_from_signal
from fno import FNO2d, train, measure_delta, whitener, late_delta
from unet import UNet2d


def pod_basis(fields, r):
    """Method of snapshots on the vorticity fields. Returns modes, mean, energy."""
    X = fields.reshape(len(fields), -1).astype(np.float64)
    mean = X.mean(0)
    Xc = X - mean
    G = Xc @ Xc.T
    w, V = np.linalg.eigh(G)
    w, V = np.clip(w[::-1], 0, None), V[:, ::-1]
    modes = (Xc.T @ V[:, :r]) / np.sqrt(np.maximum(w[:r], 1e-30))
    return modes, mean, w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/kolm/nu1.7e-3_snaps.npz")
    ap.add_argument("--period", type=float, default=29.376736396871546)
    ap.add_argument("--modes-pod", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--fmodes", type=int, nargs=2, default=[16, 16])
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seg-len", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--periods", type=int, default=100)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--variant", choices=["residual", "direct"], default="residual")
    ap.add_argument("--phase-weight", type=float, default=0.0,
                    help="weight on the squared relative period error of the "
                         "training segment; 0 reproduces the earlier runs")
    ap.add_argument("--phase-warmup", type=float, default=0.5,
                    help="fraction of epochs before the phase term switches on")
    ap.add_argument("--seed-list", type=int, nargs="+", default=[0])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--arch", choices=["fno", "unet"], default="fno")
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda; float64 throughout either way")
    ap.add_argument("--seed-init", action="store_true",
                    help="seed the weight initialisation as well as the segment "
                         "order. Runs before this flag drew initial weights from "
                         "torch's per-process random default seed, so the seed "
                         "fixed the data order only and no run is re-runnable "
                         "bit for bit.")
    ap.add_argument("--zero-init-out", action="store_true",
                    help="U-Net only: zero the output convolution so the residual "
                         "network starts at the identity map")
    ap.add_argument("--ckpt", action="store_true",
                    help="recompute activations per step in the backward pass "
                         "(same arithmetic, needed for long segments in 40 GB)")
    ap.add_argument("--periods-long", type=float, default=0,
                    help="continue the rollout to this many periods and fit delta "
                         "on its second half with a split-half test; delta itself "
                         "is still fitted on the first --periods, as before")
    ap.add_argument("--save-model", default=None,
                    help="path for the trained state_dict, so a delta can be "
                         "re-measured later without retraining")
    ap.add_argument("--save-coefs", default=None,
                    help="npz path for the rolled-out POD coefficients, so delta "
                         "can be re-measured on any window afterwards")
    ap.add_argument("--out", default="fno_kolm.json")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device(a.device)

    z = np.load(a.data)
    fields = z["snaps"].astype(np.float32)          # [Nt, H, W] or [Nt, C, H, W]
    if fields.ndim == 3:
        fields = fields[:, None]
    ch = fields.shape[1]
    tsnap = z["t"].astype(float)
    dt_snap = float(np.median(np.diff(tsnap)))
    T_snap = a.period / dt_snap                      # period in snapshot units

    pm, pmean, energy = pod_basis(fields, a.modes_pod)
    scale = float(np.abs(fields).max())
    data = torch.from_numpy(fields / scale).to(
        torch.get_default_dtype()).to(dev)
    n_steps = int(a.periods * T_snap)
    H, W = fields.shape[2], fields.shape[3]

    print(f"{os.path.basename(a.data)}  field {ch}x{H}x{W}  N={len(fields)}  "
          f"T={T_snap:.2f} snaps/period  rollout={n_steps} steps  "
          f"POD {a.modes_pod} modes = {energy[:a.modes_pod].sum()*100:.3f}% energy",
          flush=True)

    dt_ = torch.get_default_dtype()
    train_coefs = (data.reshape(len(data), -1).cpu().numpy().astype(np.float64)
                   * scale - pmean) @ pm
    pod_t = (torch.from_numpy(pm).to(dt_).to(dev),
             torch.from_numpy(pmean).to(dt_).to(dev),
             T_snap, scale, torch.from_numpy(whitener(train_coefs)).to(dt_).to(dev))

    rows = []
    for seed in a.seed_list:
        t0 = time.time()
        if a.seed_init:
            torch.manual_seed(seed)
        if a.arch == "fno":
            model = FNO2d(width=a.width, modes=tuple(a.fmodes), depth=a.depth,
                          ch=ch, residual=(a.variant == "residual"))
        else:
            model = UNet2d(ch=ch, residual=(a.variant == "residual"),
                           zero_out=a.zero_init_out)
        model = model.to(dev)
        npar = sum(q.numel() for q in model.parameters())
        print(f"  seed {seed}  arch={a.arch}  variant={a.variant}  "
              f"noise={a.noise}  phase_weight={a.phase_weight}  params={npar}  "
              f"device={dev}", flush=True)
        mse = train(model, data, a.seg_len, a.epochs, a.lr, a.batch, seed,
                    noise=a.noise, phase=a.phase_weight, pod=pod_t,
                    phase_warmup=a.phase_warmup, ckpt=a.ckpt)
        n_extra = max(0, int(a.periods_long * T_snap) - n_steps)
        dl, reached, coefs = measure_delta(model, data[0], n_steps, pm, pmean,
                                           T_snap, scale, return_coefs=True,
                                           n_extra=n_extra)
        late = late_delta(coefs, T_snap, a.periods_long) if (a.periods_long and dl is not None) else None
        if a.save_model:
            torch.save(model.state_dict(), a.save_model)
        if a.save_coefs:
            np.savez_compressed(a.save_coefs if len(a.seed_list) == 1
                                else a.save_coefs.replace(".npz", f"_s{seed}.npz"),
                                coefs=coefs, T_snap=T_snap, seed=seed)
        tag = f"{a.arch}-kolm-{a.variant}" + ("-phase" if a.phase_weight else "")
        rows.append(dict(arch=tag, seed=seed, nparam=npar,
                         noise=a.noise, phase_weight=a.phase_weight,
                         train_mse=mse, delta=dl,
                         steps_reached=reached, secs=time.time() - t0,
                         device=(torch.cuda.get_device_name(dev)
                                 if dev.type == "cuda" else "cpu"),
                         init_seeded=a.seed_init, **(late or {})))
        hz = "DIVERGED" if dl is None else f"{1/(4*abs(dl)):.0f} P"
        print(f"  seed {seed}  mse={mse:.3e}  "
              f"delta={'DIVERGED' if dl is None else '%+.3e' % dl}  {hz}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    ds = [abs(r["delta"]) for r in rows if r["delta"] is not None]
    summary = dict(data=os.path.basename(a.data),
                   nu=1.7e-3 if "kolm" in a.data else None,
                   T_snap=T_snap, variant=a.variant, noise=a.noise,
                   phase_weight=a.phase_weight,
                   med_delta=float(np.median(ds)) if ds else None,
                   n_stable=len(ds), n_seeds=len(rows),
                   pod_energy=float(energy[:a.modes_pod].sum()))
    print("\n" + json.dumps(summary, indent=1))
    json.dump(dict(rows=rows, summary=summary, config=vars(a)),
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
