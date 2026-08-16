import json, glob, math
import numpy as np

rows = []
for f in sorted(glob.glob("results/quasi/*.json")):
    rows += json.load(open(f))
live = [r for r in rows if not r.get("diverged") and "delta" in r]
eps_all = sorted({r["eps"] for r in rows})

print("=== regime map, and what each diagnostic reports ===")
print("  For LOCKED truth the single-period measurement is the correct one; for")
print("  torus truth the two-frequency one is. Reported on their own terms: the")
print("  predicted trajectory is not exactly locked even when the truth is, so a")
print("  shared classifier would compare mismatched quantities.\n")
h = "{:>6} {:>15} {:>10} {:>24} {:>14}".format(
    "eps", "regime", "drift", "delta (proper)", "single-period")
print(h); print("-" * len(h))
summ = {}
for e in eps_all:
    g = [r for r in live if r["eps"] == e]
    if not g:
        continue
    lk = g[0]["locked"]
    ds = float(np.median([abs(r["delta_single"]) for r in g]))
    if lk:
        proper, cell = ds, "%.2e  (= single)" % ds
    else:
        d1 = float(np.median([abs(r["delta"][0]) for r in g]))
        d2s = [abs(r["delta"][1]) for r in g if len(r["delta"]) > 1]
        d2 = float(np.median(d2s)) if d2s else float("nan")
        proper = max(d1, d2)
        cell = "%.2e, %.2e" % (d1, d2)
    summ[e] = dict(locked=lk, proper=proper, single=ds, n=len(g))
    print("{:>6.2f} {:>15} {:>10.1e} {:>24} {:>14.2e}".format(
        e, "LOCKED" if lk else "quasi-periodic", g[0]["drift"], cell, ds))

print("\n=== the boundary: how wrong is a single-period diagnostic? ===\n")
print("{:>6} {:>15} {:>20}".format("eps", "regime", "single / proper"))
print("-" * 43)
for e in eps_all:
    s = summ.get(e)
    if not s:
        continue
    r = s["single"] / s["proper"] if s["proper"] > 0 else float("nan")
    print("{:>6.2f} {:>15} {:>19.1f}x".format(
        e, "LOCKED" if s["locked"] else "quasi-periodic", r))

print("\n=== error law: ratio observed/predicted ===")
print("  single = one frequency (the assumption in the rest of the paper)")
print("  multi  = both frequency families\n")
h = "{:>6} {:>15} {:>16} {:>16}".format("eps", "regime", "single", "multi")
print(h); print("-" * len(h))
for e in eps_all:
    g = [r for r in live if r["eps"] == e]
    if not g:
        continue
    rs, rm = [], []
    for r in g:
        for x in r.get("errs", []):
            if x["err"] < 0.02:
                continue
            if x.get("pred_single", 0) > 0.05:
                rs.append(x["err"] / x["pred_single"])
            if x.get("pred_multi", 0) > 0.05:
                rm.append(x["err"] / x["pred_multi"])
    f = lambda v: "%.2f (n=%d)" % (np.median(v), len(v)) if v else "-"
    print("{:>6.2f} {:>15} {:>16} {:>16}".format(
        e, "LOCKED" if g[0]["locked"] else "quasi-periodic", f(rs), f(rm)))

print("\n=== practical warning: horizon from the dominant frequency alone ===\n")
h = "{:>6} {:>13} {:>13} {:>14}".format(
    "eps", "1/(4 d_dom)", "1/(4 d_max)", "overestimate")
print(h); print("-" * len(h))
for e in eps_all:
    g = [r for r in live if r["eps"] == e and not r["locked"]
         and len(r.get("delta", [])) > 1]
    if not g:
        continue
    dom, mx = [], []
    for r in g:
        d = [abs(x) for x in r["delta"]]
        w = r["weights"][:len(d)]
        dom.append(d[int(np.argmax(w))])
        mx.append(max(d))
    hd, hm = 1 / (4 * np.median(dom)), 1 / (4 * np.median(mx))
    print("{:>6.2f} {:>13.0f} {:>13.0f} {:>13.1f}x".format(e, hd, hm, hd / hm))
