import json, glob, math
import numpy as np

rows = []
for f in sorted(glob.glob("results/downstream/*.json")):
    rows += json.load(open(f))["rows"]
order = ["exact DMD", "linear-sgd", "ode-mlp", "ode-deep", "ar-residual",
         "ar-direct", "gru"]
Ns = sorted({r["N"] for r in rows})
live = [r for r in rows if not r.get("diverged") and "err_inst" in r]


def med_delta(a):
    d = [abs(r["delta"]) for r in live
         if r["arch"] == a and r.get("delta") is not None]
    return np.median(d) if d else None


print("=== PHASE-INVARIANT statistics (normalised by fluctuation scale) ===\n")
for key, lab in [("err_mean", "time-mean"), ("err_rms", "RMS fluct"),
                 ("err_cov", "covariance")]:
    hdr = "{:>13} {:>10}".format(lab, "delta") + "".join(f"{n:>9}" for n in Ns)
    print(hdr); print("-" * len(hdr))
    for a in order:
        g = [r for r in live if r["arch"] == a]
        if not g:
            continue
        dm = med_delta(a)
        cells = []
        for n in Ns:
            v = [r[key] for r in g if r["N"] == n and key in r]
            cells.append(f"{np.median(v):.1e}" if v else "    -")
        print("{:>13} {:>10}".format(a, f"{dm:.1e}" if dm else "-")
              + "".join(f"{c:>9}" for c in cells))
    print()

print("=== PHASE-RESOLVED: instantaneous field error ===\n")
hdr = "{:>13} {:>10}".format("model", "delta") + "".join(f"{n:>8}" for n in Ns)
print(hdr); print("-" * len(hdr))
for a in order:
    g = [r for r in live if r["arch"] == a]
    if not g:
        continue
    dm = med_delta(a)
    cells = []
    for n in Ns:
        v = [r["err_inst"] for r in g if r["N"] == n]
        cells.append(f"{np.median(v):.3f}" if v else "     -")
    print("{:>13} {:>10}".format(a, f"{dm:.1e}" if dm else "-")
          + "".join(f"{c:>8}" for c in cells))

print("\n=== does it follow 2|sin(pi delta N)| ? ===")
print("    ratio observed/predicted, predictions above 0.05 only\n")
hdr = "{:>13} {:>9} {:>9} {:>9} {:>6}".format(
    "model", "median", "p10", "p90", "n")
print(hdr); print("-" * len(hdr))
allr = []
for a in order:
    rr = [r["err_inst"] / r["err_inst_pred"] for r in live
          if r["arch"] == a and r.get("err_inst_pred") is not None
          and r["err_inst_pred"] > 0.05]
    if rr:
        allr += rr
        print("{:>13} {:>9.3f} {:>9.3f} {:>9.3f} {:>6}".format(
            a, np.median(rr), np.percentile(rr, 10),
            np.percentile(rr, 90), len(rr)))
if allr:
    print("-" * len(hdr))
    print("{:>13} {:>9.3f} {:>9.3f} {:>9.3f} {:>6}".format(
        "ALL", np.median(allr), np.percentile(allr, 10),
        np.percentile(allr, 90), len(allr)))

print("\n=== usable horizon: predicted 1/(4d) vs where error first hits sqrt(2) ===\n")
print("{:>13} {:>11} {:>13} {:>11}".format(
    "model", "delta", "1/(4 delta)", "observed"))
print("-" * 51)
for a in order:
    g = sorted([r for r in live if r["arch"] == a], key=lambda r: r["N"])
    if not g:
        continue
    dm = med_delta(a)
    if not dm:
        continue
    hit = [r["N"] for r in g if r["err_inst"] >= math.sqrt(2)]
    print("{:>13} {:>11.2e} {:>13.0f} {:>11}".format(
        a, dm, 1 / (4 * dm), str(min(hit)) if hit else ">30000"))

print("\n=== the practical split, at the longest horizon reached ===")
nmax = max(Ns)
inv, res = [], []
for a in order:
    g = [r for r in live if r["arch"] == a and r["N"] == nmax]
    for r in g:
        inv += [r.get("err_rms"), r.get("err_cov")]
        res.append(r.get("err_inst"))
inv = [x for x in inv if x is not None]
res = [x for x in res if x is not None]
if inv and res:
    print(f"  phase-invariant  median {np.median(inv):.2e}   worst {max(inv):.2e}")
    print(f"  phase-resolved   median {np.median(res):.2e}   worst {max(res):.2e}")
    print(f"  ratio at N={nmax}: {np.median(res)/np.median(inv):.0f}x")
