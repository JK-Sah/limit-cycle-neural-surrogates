import json, glob, math
import numpy as np

print("=== CYLINDER RESOLUTION STUDY (Re=100) ===")
hdr = "{:>4} {:>11} {:>9} {:>10} {:>8} {:>10} {:>7}".format(
    "D", "grid", "St", "vs .1643", "Cd", "T_std/T", "cycles")
print(hdr); print("-" * len(hdr))
rows = []
for f in glob.glob("results/cyl/D*.json"):
    d = json.load(open(f))
    if "St" in d:
        rows.append(d)
for d in sorted(rows, key=lambda d: d["D"]):
    print("{:>4} {:>11} {:>9.5f} {:>+9.2f}% {:>8.4f} {:>10.2e} {:>7}".format(
        d["D"], "{}x{}".format(d["nx"], d["ny"]), d["St"],
        (d["St"] - 0.1643) / 0.1643 * 100, d["Cd_mean"],
        d["T_std"] / d["T_lattice"], d["n_cycles"]))

print("\n=== ANCHOR SWEEP (supervision weight x observation noise) ===")
ar = []
for f in glob.glob("results/anchor/*.json"):
    ar += json.load(open(f))
hdr = "{:>8} {:>10} {:>12} {:>12} {:>10} {:>12}".format(
    "w_sup", "obs_noise", "meas floor", "median |d|", "d/floor", "horizon P")
print(hdr); print("-" * len(hdr))
keys = sorted({(r["w_sup"], r["obs_noise"]) for r in ar})
for w, n in keys:
    g = [r for r in ar if r["w_sup"] == w and r["obs_noise"] == n
         and r.get("delta_T") is not None]
    if not g:
        continue
    d = np.median([abs(r["delta_T"]) for r in g])
    fl = np.median([r.get("measurement_floor", float("nan")) for r in g])
    print("{:>8.0e} {:>10.0e} {:>12.2e} {:>12.2e} {:>10.1f} {:>12.0f}".format(
        w, n, fl, d, d / fl if fl else float("nan"), 1 / (4 * d)))
