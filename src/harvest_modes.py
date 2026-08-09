import json, glob, re
import numpy as np
runs = {}
for f in sorted(glob.glob("results/modesweep/modes*_seed*.json")):
    d = json.load(open(f))
    runs.setdefault(d["modes"], []).append(d)
if not runs:
    raise SystemExit("no results yet")
mode_list = sorted(runs)
arch_order = ["DMD (least squares)", "linear-sgd", "ode-mlp", "ode-deep",
              "ar-residual", "ar-direct", "gru"]

print("=== delta vs POD truncation (Re=100, 3 seeds each) ===\n")
print("POD energy captured by the retained subspace:")
for m in mode_list:
    e = runs[m][0].get("energy_captured")
    tail = runs[m][0].get("energy_tail")
    if e is not None:
        print(f"  {m:>3} modes: {e*100:9.5f}%   next 4 modes carry {tail*100:.5f}%")

print("\nmedian |delta| by architecture and truncation:")
hdr = "{:>20}".format("architecture") + "".join(f"{m:>12}" for m in mode_list)
print(hdr); print("-" * len(hdr))
for arch in arch_order:
    cells = []
    for m in mode_list:
        ds = []
        for d in runs[m]:
            ds += [abs(r["delta"]) for r in d["rows"]
                   if r["arch"] == arch and r["delta"] is not None]
        cells.append(f"{np.median(ds):.2e}" if ds else "  diverged")
    print("{:>20}".format(arch) + "".join(f"{c:>12}" for c in cells))

print("\nusable horizon (periods):")
print(hdr); print("-" * len(hdr))
for arch in arch_order:
    cells = []
    for m in mode_list:
        ds = []
        for d in runs[m]:
            ds += [abs(r["delta"]) for r in d["rows"]
                   if r["arch"] == arch and r["delta"] is not None]
        cells.append(f"{1/(4*np.median(ds)):.0f}" if ds else "0")
    print("{:>20}".format(arch) + "".join(f"{c:>12}" for c in cells))

print("\n--- does delta track the unresolved energy? ---")
for arch in ["DMD (least squares)", "ode-mlp", "linear-sgd"]:
    xs, ys = [], []
    for m in mode_list:
        e = runs[m][0].get("energy_captured")
        ds = []
        for d in runs[m]:
            ds += [abs(r["delta"]) for r in d["rows"]
                   if r["arch"] == arch and r["delta"] is not None]
        if ds and e is not None:
            xs.append(1.0 - e); ys.append(np.median(ds))
    if len(xs) > 2:
        rho = np.corrcoef(np.log10(np.maximum(xs, 1e-12)), np.log10(ys))[0, 1]
        print(f"  {arch:22s} corr(log unresolved energy, log |delta|) = {rho:+.3f}")
