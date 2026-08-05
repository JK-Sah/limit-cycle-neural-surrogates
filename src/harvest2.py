import json, glob
import numpy as np

print("=== BLOCKAGE STUDY (D=24, Re=100) ===")
h = "{:>6} {:>10} {:>12} {:>9} {:>10} {:>11}".format(
    "ny/D", "blockage", "grid", "St", "vs .1643", "jitter")
print(h); print("-" * len(h))
rows = [json.load(open(f)) for f in glob.glob("results/blockage/*.json")]
rows = [r for r in rows if "St" in r]
for d in sorted(rows, key=lambda d: d["blockage"], reverse=True):
    print("{:>6.0f} {:>9.2f}% {:>12} {:>9.5f} {:>+9.2f}% {:>11.1e}".format(
        1 / d["blockage"], d["blockage"] * 100,
        "{}x{}".format(d["nx"], d["ny"]), d["St"],
        (d["St"] - 0.1643) / 0.1643 * 100, d["T_std"] / d["T_lattice"]))
if len(rows) >= 2:
    s = sorted(rows, key=lambda d: d["blockage"])
    print("\n  trend as the domain opens: St {:.5f} -> {:.5f} "
          "as blockage {:.1f}% -> {:.1f}%".format(
              s[-1]["St"], s[0]["St"], s[-1]["blockage"] * 100,
              s[0]["blockage"] * 100))

print("\n=== CYLINDER LATENT SURROGATE (D=16, 8 POD modes, 4000 epochs) ===")
try:
    d = json.load(open("results/cyl_latent_D16.json"))
except FileNotFoundError:
    print("  (job still running)"); raise SystemExit
rs = d["rows"]
h = "{:>15} {:>6} {:>8} {:>11} {:>12} {:>12} {:>11}".format(
    "model", "width", "params", "med mse", "med |delta|", "delta spread",
    "horizon P")
print(h); print("-" * len(h))
for kind, width in sorted({(r["kind"], r["width"]) for r in rs},
                          key=lambda t: (t[0], t[1])):
    g = [r for r in rs if r["kind"] == kind and r["width"] == width]
    ds = [abs(r["delta"]) for r in g if r.get("delta") is not None]
    if not ds:
        print("{:>15} {:>6} all diverged".format(kind, width)); continue
    m = np.median(ds)
    print("{:>15} {:>6} {:>8} {:>11.2e} {:>12.2e} {:>12.1f}x {:>11.0f}".format(
        kind, width, g[0]["nparam"],
        np.median([r["train_mse"] for r in g]), m,
        max(ds) / min(ds), 1 / (4 * m)))
