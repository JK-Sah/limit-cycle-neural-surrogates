import json, glob
import numpy as np

rows = []
for f in sorted(glob.glob("results/parosc/*.json")):
    rows += json.load(open(f))["rows"]
meta = json.load(open(sorted(glob.glob("results/parosc/*.json"))[0]))
T = {float(k): v for k, v in meta["T"].items()}
n_seeds = len({r["seed"] for r in rows})
print("=== PARAMETRIC VAN DER POL ({} seeds, 6000 epochs) ===".format(n_seeds))

configs = sorted({(r["kind"], r["width"]) for r in rows},
                 key=lambda t: (t[0] != "free", t[1]))
h = "{:>15} {:>5} {:>7} {:>10} {:>11} {:>11} {:>11}".format(
    "model", "width", "params", "med mse", "train", "interp", "extrap")
print(h); print("-" * len(h))
for kind, w in configs:
    g = [r for r in rows if r["kind"] == kind and r["width"] == w]
    if not g:
        continue
    cell = {}
    for tag in ["train", "interp", "extrap"]:
        ds = [abs(r["delta"]) for r in g
              if r["split"] == tag and r["delta"] is not None]
        nd = len([r for r in g if r["split"] == tag and r["delta"] is None])
        cell[tag] = ("{:.2e}".format(np.median(ds)) if ds else "diverged")
        if nd:
            cell[tag] += "*"
    print("{:>15} {:>5} {:>7} {:>10.2e} {:>11} {:>11} {:>11}".format(
        kind, w, g[0]["nparam"], np.median([r["train_mse"] for r in g]),
        cell["train"], cell["interp"], cell["extrap"]))
print("  (* = some seeds produced a non-finite rollout)")

print("\n--- ratio, free(best) / phase-anchored ---")
best = {}
for tag in ["train", "interp", "extrap"]:
    fr = []
    for kind, w in configs:
        if kind != "free":
            continue
        ds = [abs(r["delta"]) for r in rows if r["kind"] == kind
              and r["width"] == w and r["split"] == tag
              and r["delta"] is not None]
        if ds:
            fr.append(np.median(ds))
    pa = [abs(r["delta"]) for r in rows if r["kind"] == "phase-anchored"
          and r["split"] == tag and r["delta"] is not None]
    if fr and pa:
        best[tag] = (min(fr), np.median(pa))
        print("  {:>7}: free {:.2e}   anchored {:.2e}   ratio {:>8.1f}x".format(
            tag, min(fr), np.median(pa), min(fr) / np.median(pa)))

print("\n--- per-eps median |delta| (phase-anchored) ---")
for e in sorted(T):
    g = [abs(r["delta"]) for r in rows if r["kind"] == "phase-anchored"
         and abs(r["eps"] - e) < 1e-9 and r["delta"] is not None]
    tag = ("train" if e in meta["train_eps"] else
           "interp" if e in meta["interp_eps"] else "extrap")
    if g:
        print("  eps={:.1f} [{:>6}] T={:.5f}  med |delta| = {:.3e}".format(
            e, tag, T[e], np.median(g)))

print("\n=== DOMAIN STUDY (D=24, Re=100, 5% blockage) ===")
h = "{:>6} {:>6} {:>12} {:>9} {:>10}".format(
    "nx/D", "xc/D", "grid", "St", "vs .1643")
print(h); print("-" * len(h))
dr = [json.load(open(f)) for f in glob.glob("results/domain/*.json")]
for d in sorted([r for r in dr if "St" in r],
                key=lambda d: (d["nx"], d["xc"] if "xc" in d else 0)):
    print("{:>6.0f} {:>6} {:>12} {:>9.5f} {:>+9.2f}%".format(
        d["nx"] / d["D"], "-", "{}x{}".format(d["nx"], d["ny"]),
        d["St"], (d["St"] - 0.1643) / 0.1643 * 100))
