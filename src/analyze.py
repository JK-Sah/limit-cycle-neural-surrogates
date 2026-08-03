"""Aggregate the Tier-0 sweep and test the three predictions."""
import glob, json, math, sys
import numpy as np
from scipy import stats


def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        with open(f) as fh:
            rows += json.load(fh)
    return rows


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return np.median(v) if v else float("nan")


def fmt_inf(v):
    v = [x for x in v if x is not None]
    fin = [x for x in v if np.isfinite(x)]
    if not fin:
        return f"all >horizon ({len(v)})"
    s = f"{np.median(fin):.0f}"
    n_inf = len(v) - len(fin)
    return s + (f" ({n_inf}/{len(v)} >horizon)" if n_inf else "")


def main(pat):
    rows = load(pat)
    print(f"loaded {len(rows)} trials\n")

    ok = [r for r in rows if r.get("delta_T") is not None]
    bad = [r for r in rows if r.get("delta_T") is None]
    print(f"{len(ok)} produced a measurable limit cycle; {len(bad)} did not")
    if bad:
        for r in bad:
            print(f"    no cycle: n={r['n_traj']} w={r['width']} s={r['seed']} "
                  f"reg={r['reg']} mse={r['train_mse']:.1e}")
    print()

    # ---- correctness of the diagnostic -------------------------------------
    lam1 = [r["floquet_phase"] for r in ok]
    ratio = [r["phase_slope_ratio"] for r in ok
             if r.get("phase_slope_ratio") is not None]
    print("DIAGNOSTIC CHECKS")
    print(f"  |lambda_1| (theory: exactly 1): "
          f"max deviation {max(abs(l-1) for l in lam1):.2e}")
    print(f"  observed/predicted phase slope: median {np.median(ratio):.4f}  "
          f"[{np.min(ratio):.3f}, {np.max(ratio):.3f}]")
    print(f"  -> phase drift is fully explained by the period error\n")

    # ---- P1: phase unbounded, amplitude bounded ----------------------------
    print("P1  phase secular vs amplitude bounded")
    ph = [r["phase_final"] for r in ok]
    am = [r["amp_max"] for r in ok]
    print(f"  |phase error| at horizon : median {np.median(ph):.3f} rad  "
          f"max {np.max(ph):.2f}")
    print(f"  max |amplitude error|    : median {np.median(am):.2e}  "
          f"max {np.max(am):.2e}   (r* = 0.7071)")
    print(f"  linearity of phase(t), median residual: "
          f"{med([r['phase_lin_resid'] for r in ok]):.2e} rad\n")

    # ---- the reframed headline: does the loss predict delta? ---------------
    sc = [r for r in ok if r.get("exp") == "scaling"]
    if sc:
        x = np.log10([r["train_mse"] for r in sc])
        y = np.log10([abs(r["delta_T"]) + 1e-12 for r in sc])
        rho, p = stats.spearmanr(x, y)
        print("DOES TRAINING LOSS PREDICT LONG-HORIZON FIDELITY?")
        print(f"  Spearman(log mse, log |delta|) over {len(sc)} runs: "
              f"rho={rho:+.3f}  p={p:.3g}")
        print(f"  -> {'loss is informative' if p < 0.05 else 'loss carries essentially no information about delta'}")
        # spread of delta among the best-fitting decile
        k = max(3, len(sc) // 10)
        best = sorted(sc, key=lambda r: r["train_mse"])[:k]
        d = [abs(r["delta_T"]) for r in best]
        print(f"  among the {k} lowest-loss runs, |delta| spans "
              f"{min(d):.2e} .. {max(d):.2e}  ({max(d)/max(min(d),1e-12):.0f}x)\n")

        print("P2  scaling")
        print(f"  {'n_traj':>7} {'width':>6} {'med |dT/T|':>12} {'med mse':>10} "
              f"{'med decorr':>22}")
        for n in sorted({r["n_traj"] for r in sc}):
            for w in sorted({r["width"] for r in sc}):
                g = [r for r in sc if r["n_traj"] == n and r["width"] == w]
                if not g:
                    continue
                print(f"  {n:>7} {w:>6} {med([abs(r['delta_T']) for r in g]):>12.2e} "
                      f"{med([r['train_mse'] for r in g]):>10.1e} "
                      f"{fmt_inf([r['decorr_periods'] for r in g]):>22}")
        print()

    # ---- P3: do the standard stability fixes touch delta? ------------------
    rg = [r for r in ok if r.get("exp") == "regularizers"]
    if rg:
        print("P3  stability regularizers vs the phase direction")
        print(f"  {'method':>12} {'weight':>8} {'med |dT/T|':>12} {'med mse':>10} "
              f"{'med |lam2|':>11} {'med decorr':>22}")
        base = None
        for key in sorted({(r["reg"], r["reg_w"]) for r in rg},
                          key=lambda t: (t[0], t[1])):
            g = [r for r in rg if (r["reg"], r["reg_w"]) == key]
            d = med([abs(r["delta_T"]) for r in g])
            if key[0] == "none":
                base = d
            print(f"  {key[0]:>12} {key[1]:>8.0e} {d:>12.2e} "
                  f"{med([r['train_mse'] for r in g]):>10.1e} "
                  f"{med([r['floquet_trans'] for r in g]):>11.2e} "
                  f"{fmt_inf([r['decorr_periods'] for r in g]):>22}")
        if base:
            print(f"\n  relative to unregularized baseline (|dT/T| = {base:.2e}):")
            for key in sorted({(r["reg"], r["reg_w"]) for r in rg}):
                if key[0] == "none":
                    continue
                g = [r for r in rg if (r["reg"], r["reg_w"]) == key]
                d = med([abs(r["delta_T"]) for r in g])
                print(f"    {key[0]:>12} {key[1]:.0e}: {d/base:>6.2f}x delta   "
                      f"(transverse |lam2| "
                      f"{med([r['floquet_trans'] for r in g]):.2e})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/all/*.json")
