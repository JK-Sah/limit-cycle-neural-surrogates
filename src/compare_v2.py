import json, glob
import numpy as np

def load(pat, key="arch"):
    rows = []
    for f in sorted(glob.glob(pat)):
        rows += json.load(open(f))["rows"]
    return rows

old = load("results/arch/*.json")
new = load("results/arch_v2/*.json")
order = ["DMD (least squares)", "linear-sgd", "ode-mlp", "ode-deep",
         "ar-residual", "ar-direct", "gru"]
print("=== ARCHITECTURE at Re=100, 8 modes: old estimator vs corrected ===\n")
h = "{:>20} {:>13} {:>13} {:>9} {:>12} {:>12}".format(
    "model", "old |delta|", "new |delta|", "ratio", "old horizon", "new horizon")
print(h); print("-" * len(h))
for a in order:
    o = [abs(r["delta"]) for r in old if r["arch"] == a and r["delta"] is not None]
    n = [abs(r["delta"]) for r in new if r["arch"] == a and r["delta"] is not None]
    if not o or not n:
        continue
    mo, mn = np.median(o), np.median(n)
    print("{:>20} {:>13.3e} {:>13.3e} {:>8.2f}x {:>10.0f} P {:>10.0f} P".format(
        a, mo, mn, mn / mo, 1/(4*mo), 1/(4*mn)))

print("\n=== MODE SWEEP: corrected horizons (periods) ===\n")
runs = {}
for f in sorted(glob.glob("results/modesweep_v2/modes*_seed*.json")):
    d = json.load(open(f)); runs.setdefault(d["modes"], []).append(d)
ml = sorted(runs)
hdr = "{:>20}".format("architecture") + "".join(f"{m:>10}" for m in ml)
print(hdr); print("-" * len(hdr))
for a in order:
    cells = []
    for m in ml:
        ds = []
        for d in runs[m]:
            ds += [abs(r["delta"]) for r in d["rows"]
                   if r["arch"] == a and r["delta"] is not None]
        cells.append(f"{1/(4*np.median(ds)):.0f}" if ds else "div")
    print("{:>20}".format(a) + "".join(f"{c:>10}" for c in cells))

print("\n=== does DMD still look flat across truncation? ===")
for a in ["DMD (least squares)", "ode-mlp"]:
    vals = []
    for m in ml:
        ds = []
        for d in runs[m]:
            ds += [abs(r["delta"]) for r in d["rows"]
                   if r["arch"] == a and r["delta"] is not None]
        if ds:
            vals.append(np.median(ds))
    if vals:
        print(f"  {a:22s} min {min(vals):.2e}  max {max(vals):.2e}  "
              f"spread {max(vals)/min(vals):.1f}x")
