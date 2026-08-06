import json, glob, os, re
import numpy as np
print("=== DOMAIN STUDY (D=24, Re=100, blockage 5%) ===")
h = "{:>7} {:>7} {:>7} {:>12} {:>9} {:>10}".format(
    "nx/D", "up/D", "down/D", "grid", "St", "vs .1643")
print(h); print("-" * len(h))
out = []
for f in glob.glob("results/domain/*.json"):
    d = json.load(open(f))
    if "St" not in d: continue
    m = re.search(r"nx(\d+)_xc(\d+)", os.path.basename(f))
    nx, xc = int(m.group(1)), int(m.group(2))
    out.append((nx, xc, d))
for nx, xc, d in sorted(out, key=lambda t: (t[0] - t[1])):
    print("{:>7} {:>7} {:>7} {:>12} {:>9.5f} {:>+9.2f}%".format(
        nx, xc, nx - xc, "{}x{}".format(d["nx"], d["ny"]), d["St"],
        (d["St"] - 0.1643) / 0.1643 * 100))
print("\n  downstream length is what moves it: {:.2f}% at {}D "
      "-> {:.2f}% at {}D".format(
          (sorted(out, key=lambda t: t[0]-t[1])[0][2]["St"]-0.1643)/0.1643*100,
          sorted(out, key=lambda t: t[0]-t[1])[0][0]-sorted(out, key=lambda t: t[0]-t[1])[0][1],
          (sorted(out, key=lambda t: t[0]-t[1])[-1][2]["St"]-0.1643)/0.1643*100,
          sorted(out, key=lambda t: t[0]-t[1])[-1][0]-sorted(out, key=lambda t: t[0]-t[1])[-1][1]))

print("\n=== PARAMETRIC vdP: usable horizons (periods to pi/2 phase error) ===")
rows = []
for f in sorted(glob.glob("results/parosc/*.json")):
    rows += json.load(open(f))["rows"]
h = "{:>15} {:>5} {:>8} {:>12} {:>12} {:>12}".format(
    "model", "width", "params", "train", "interp", "extrap")
print(h); print("-" * len(h))
for kind, w in sorted({(r["kind"], r["width"]) for r in rows},
                      key=lambda t: (t[0] != "free", t[1])):
    g = [r for r in rows if r["kind"] == kind and r["width"] == w]
    c = []
    for tag in ["train", "interp", "extrap"]:
        ds = [abs(r["delta"]) for r in g
              if r["split"] == tag and r["delta"] is not None]
        c.append("{:.0f} P".format(1/(4*np.median(ds))) if ds else "n/a")
    print("{:>15} {:>5} {:>8} {:>12} {:>12} {:>12}".format(
        kind, w, g[0]["nparam"], *c))
