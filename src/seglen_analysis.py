"""
Segment-length sweep: per-arm table, two-arm tests, dose-response, and the
ten-period delta, all from results/seglen.

Every two-arm comparison uses the test the paper applies to the remedies it
rejects: an exact one-sided permutation test on the difference in median
log10|delta|, enumerating every relabelling. Holding the recommended
intervention to the same bar as the failed ones is the point of using it.

The ten-period check re-measures delta on the first ten periods of each saved
rollout with the same estimator, and compares it with the delta the sweep
reports, which comes from the second half of a 100-period rollout.

  python src/seglen_analysis.py [--results results] [--json out.json]
"""
import argparse, glob, itertools, json, math, os
import numpy as np
from scipy.stats import spearmanr

from period import period_from_signal


SPLIT_OK = 1e-7     # the split-half threshold the paper applies to data


def load(res, sub="seglen_v2", metric="delta_late"):
    """One row per run. `delta` is the metric chosen (falling back to the
    100-period delta where a run has no late window); `wellposed` applies the
    data threshold to the surrogate's own late record."""
    rows = []
    for f in sorted(glob.glob(f"{res}/{sub}/*.json")):
        tag = os.path.basename(f)[:-5]
        sysname, arch, k, b = tag.split("_s")[0].split("-")
        j = json.load(open(f))
        for r in j["rows"]:
            d = r.get(metric, r["delta"]) if r["delta"] is not None else None
            sp = r.get("split_late")
            rows.append(dict(tag=tag, sys=sysname, arch=arch, K=int(k[1:]),
                             batch=int(b[1:]), seed=r["seed"], delta=d,
                             delta100=r["delta"], split=sp,
                             wellposed=(sp is not None and sp < SPLIT_OK),
                             mse=r["train_mse"], T=j["summary"]["T_snap"],
                             coefs=f"{res}/{sub}/coefs/{tag}.npz"))
    return rows


def perm_p(worse, better):
    """P(median log|d| of `worse` exceeds that of `better` by >= observed)."""
    a, b = np.log10(worse), np.log10(better)
    x = np.r_[a, b]; n = len(a)
    obs = np.median(a) - np.median(b)
    hit = tot = 0
    for idx in itertools.combinations(range(len(x)), n):
        m = np.zeros(len(x), bool); m[list(idx)] = True
        tot += 1
        hit += (np.median(x[m]) - np.median(x[~m])) >= obs - 1e-12
    return hit / tot


def arm(rows, sysname, arch, K, batch=8):
    return [r for r in rows if (r["sys"], r["arch"], r["K"], r["batch"])
            == (sysname, arch, K, batch)]


def ok(g):
    return [abs(r["delta"]) for r in g if r["delta"] is not None]


def delta_window(path, n_periods, T):
    """delta from the first n_periods of a saved rollout, same estimator."""
    if not os.path.exists(path):
        return None
    c = np.load(path)["coefs"]
    n = int(round(n_periods * T))
    if len(c) < n:
        return None
    t = np.arange(n, dtype=float)
    Tm = period_from_signal(t, c[:n, 0])
    return (Tm - T) / T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--sub", default="seglen_v2")
    ap.add_argument("--metric", default="delta_late", choices=["delta_late", "delta"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = load(a.results, a.sub, a.metric)
    out = dict(arms=[], tests=[], slopes=[], ten_period=None)

    print(f"{len(rows)} runs\n")
    print(f"metric: {a.metric}; well-posed = late split-half < {SPLIT_OK:g}\n")
    print(f"{'system':7} {'arch':6} {'K':>3} {'K/T':>5} {'b':>3} {'stable':>7} {'wellp':>5} "
          f"{'med MSE':>9} {'med|d|':>9} {'spread':>8} {'horizon':>8} {'med split':>9}")
    for key in sorted({(r["sys"], r["arch"], r["K"], r["batch"]) for r in rows}):
        g = arm(rows, *key)
        d = ok(g)
        T = g[0]["T"]
        mse = np.median([r["mse"] for r in g if r["delta"] is not None] or [np.nan])
        md = float(np.median(d)) if d else None
        sp = max(d) / min(d) if len(d) > 1 else None
        nw = sum(r["wellposed"] for r in g)
        spl = [r["split"] for r in g if r["split"] is not None]
        print(f"{key[0]:7} {key[1]:6} {key[2]:>3} {key[2]/T:>5.2f} {key[3]:>3} "
              f"{len(d)}/{len(g):<5} {nw:>5} {mse:>9.2e} "
              f"{md if md is not None else float('nan'):>9.2e} "
              f"{sp if sp else float('nan'):>7.0f}x "
              f"{(1/(4*md)) if md else float('nan'):>8.0f} "
              f"{np.median(spl) if spl else float('nan'):>9.1e}")
        out["arms"].append(dict(sys=key[0], arch=key[1], K=key[2], batch=key[3],
                                K_over_T=key[2] / T, n=len(g), n_stable=len(d),
                                med_mse=float(mse), med_delta=md, spread=sp,
                                n_wellposed=int(nw),
                                deltas=[r["delta"] for r in g],
                                splits=[r["split"] for r in g]))

    print("\ntwo-arm tests (one-sided exact permutation, median log10|delta|)")
    pairs = [
        ("NS FNO: K=25 worse than K=40", ("ns", "fno", 25, 8), ("ns", "fno", 40, 8)),
        ("NS FNO, transitions matched: K=25 b13 worse than K=40 b8",
         ("ns", "fno", 25, 13), ("ns", "fno", 40, 8)),
        ("NS FNO, transitions matched: K=25 b8 worse than K=40 b5",
         ("ns", "fno", 25, 8), ("ns", "fno", 40, 5)),
        ("NS FNO: K=25 b13 vs K=25 b8 (more data, same K; worse = b8)",
         ("ns", "fno", 25, 8), ("ns", "fno", 25, 13)),
        ("NS U-Net (zero-init): K=25 worse than K=40", ("ns", "unetz", 25, 8),
         ("ns", "unetz", 40, 8)),
        ("NS: residual K=25 worse than direct K=25", ("ns", "fno", 25, 8),
         ("ns", "fnodir", 25, 8)),
        ("NS: direct K=25 worse than residual K=40", ("ns", "fnodir", 25, 8),
         ("ns", "fno", 40, 8)),
        ("NS: noise K=25 worse than residual K=25", ("ns", "fnonz", 25, 8),
         ("ns", "fno", 25, 8)),
        ("NS: residual K=25 worse than noise K=25", ("ns", "fno", 25, 8),
         ("ns", "fnonz", 25, 8)),
        ("wake: residual K=24 worse than direct K=24", ("wake", "fno", 24, 8),
         ("wake", "fnodir", 24, 8)),
        ("wake: noise K=24 worse than residual K=24", ("wake", "fnonz", 24, 8),
         ("wake", "fno", 24, 8)),
        ("wake: residual K=24 worse than noise K=24", ("wake", "fno", 24, 8),
         ("wake", "fnonz", 24, 8)),
        ("NS FNO: K=35 worse than K=40", ("ns", "fno", 35, 8), ("ns", "fno", 40, 8)),
        ("NS FNO: K=40 worse than K=80", ("ns", "fno", 40, 8), ("ns", "fno", 80, 8)),
        ("wake FNO: K=24 worse than K=32", ("wake", "fno", 24, 8), ("wake", "fno", 32, 8)),
        ("wake FNO: K=16 worse than K=32", ("wake", "fno", 16, 8), ("wake", "fno", 32, 8)),
        ("wake FNO: K=32 worse than K=64", ("wake", "fno", 32, 8), ("wake", "fno", 64, 8)),
    ]
    # the cardiac K are set from its measured period, so pick them by fraction
    cg = [r for r in rows if r["sys"] == "cardiac" and r["arch"] == "fno"]
    if cg:
        T = cg[0]["T"]; Ks = sorted({r["K"] for r in cg})
        near = lambda f: min(Ks, key=lambda K: abs(K / T - f))
        for f_lo, f_hi in ((0.625, 1.0), (0.75, 1.0), (0.25, 1.0), (1.0, 1.5)):
            k_lo, k_hi = near(f_lo), near(f_hi)
            if k_lo != k_hi:
                pairs.append((f"cardiac FNO: K={k_lo} ({k_lo/T:.2f}T) worse than "
                              f"K={k_hi} ({k_hi/T:.2f}T)",
                              ("cardiac", "fno", k_lo, 8), ("cardiac", "fno", k_hi, 8)))
    for name, w, b in pairs:
        dw, db = ok(arm(rows, *w)), ok(arm(rows, *b))
        if len(dw) < 2 or len(db) < 2:
            print(f"  {name}: not enough runs ({len(dw)}, {len(db)})")
            continue
        p = perm_p(dw, db)
        r = np.median(dw) / np.median(db)
        sep = min(dw) / max(db)
        print(f"  {name}: ratio {r:.1f}x  p = {p:.3f}  (n = {len(dw)}, {len(db)}; "
              f"min/max = {sep:.2f})")
        out["tests"].append(dict(name=name, ratio=float(r), p=p, n=[len(dw), len(db)],
                                 min_over_max=float(sep)))

    print("\nwell-posed fraction (late split-half < %g), Fisher exact one-sided" % SPLIT_OK)
    from scipy.stats import fisher_exact
    for sysname in ("ns", "wake", "cardiac"):
        g = [r for r in rows if r["sys"] == sysname and r["arch"] == "fno" and r["batch"] == 8]
        if not g:
            continue
        T = g[0]["T"]; Ks = sorted({r["K"] for r in g})
        Kc = min(Ks, key=lambda K: abs(K - T))
        line = "  ".join(f"K={K}: {sum(r['wellposed'] for r in g if r['K'] == K)}/"
                         f"{sum(1 for r in g if r['K'] == K)}" for K in Ks)
        print(f"  {sysname:7} {line}")
        below = [r for r in g if r["K"] < Kc]; above = [r for r in g if r["K"] >= Kc]
        if below and above:
            tab = [[sum(r["wellposed"] for r in above), sum(not r["wellposed"] for r in above)],
                   [sum(r["wellposed"] for r in below), sum(not r["wellposed"] for r in below)]]
            print(f"          K >= T vs K < T: {tab[0][0]}/{sum(tab[0])} vs {tab[1][0]}/{sum(tab[1])}"
                  f"  p = {fisher_exact(tab, alternative='greater')[1]:.3g}")

    print("\ndose-response: slope of log10 median|delta| against log10 K, and")
    print("Spearman over seeds, below and above one period (FNO, batch 8)")
    for sysname in ("ns", "wake", "cardiac"):
        g = [r for r in rows if r["sys"] == sysname and r["arch"] == "fno"
             and r["batch"] == 8 and r["delta"] is not None]
        if not g:
            continue
        T = g[0]["T"]
        # split at the swept K nearest one period (32 on the wake is 0.995 T)
        Kc = min({r["K"] for r in g}, key=lambda K: abs(K - T))
        for side, sel in (("K <= T", lambda K: K <= Kc),
                          ("K >= T", lambda K: K >= Kc)):
            Ks = sorted({r["K"] for r in g if sel(r["K"])})
            if len(Ks) < 2:
                continue
            med = [np.median([abs(r["delta"]) for r in g if r["K"] == k]) for k in Ks]
            slope = np.polyfit(np.log10(Ks), np.log10(med), 1)[0]
            pts = [(r["K"], abs(r["delta"])) for r in g if sel(r["K"])]
            rho, p = spearmanr([k for k, _ in pts], np.log10([d for _, d in pts]))
            print(f"  {sysname:4} {side}: K = {Ks}  slope {slope:+.2f}  "
                  f"Spearman rho {rho:+.2f} (p = {p:.2g}, n = {len(pts)})")
            out["slopes"].append(dict(sys=sysname, side=side, Ks=Ks, slope=float(slope),
                                      rho=float(rho), p=float(p), n=len(pts)))

    print("\nten-period delta against the reported (second half of 100 periods)")
    pairs10 = []
    for r in rows:
        if r["delta"] is None:
            continue
        d10 = delta_window(r["coefs"], 10, r["T"])
        if d10 is not None:
            pairs10.append((r["delta"], d10))
    if pairs10:
        d100, d10 = map(np.array, zip(*pairs10))
        rel = np.abs(d10 - d100) / np.abs(d100)
        same_sign = np.mean(np.sign(d10) == np.sign(d100))
        rho, p = spearmanr(np.log10(np.abs(d100)), np.log10(np.abs(d10)))
        hz = np.abs(np.log10(np.abs(d10) / np.abs(d100)))
        print(f"  n = {len(rel)}  median |d10 - d|/|d| = {np.median(rel):.2f}  "
              f"p90 = {np.percentile(rel, 90):.2f}  same sign {same_sign:.0%}  "
              f"Spearman(log) {rho:.2f}  horizon within 2x: {np.mean(hz < math.log10(2)):.0%}")
        out["ten_period"] = dict(n=len(rel), median_rel=float(np.median(rel)),
                                 p90_rel=float(np.percentile(rel, 90)),
                                 same_sign=float(same_sign), rho=float(rho),
                                 within_2x=float(np.mean(hz < math.log10(2))))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
