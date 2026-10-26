"""
One table for every operator arm: what its 400-period rollouts do, what delta
they give, and the Floquet multipliers of the trained models.

Outcome of each run, from its saved rollout (results/seglen_v2) --
  diverged      left the bounded region within the first 100 periods
  late-div      survived 100 periods, diverged before 400
  collapsed     leading-pair amplitude over periods 390-400 below 80% of that
                over periods 0-10 (or above 125%: "grown"), or the late orbit no
                longer oscillates near the true period (|delta_late| > 0.5) --
                a model that falls onto a fixed point within its first period
                keeps a constant amplitude and is caught only by the second test
  cycle         amplitude held, but the late record fails the split-half test
  cycle-wp      amplitude held and late split-half < 1e-7, the threshold the
                paper applies to data
The 20% amplitude tolerance was chosen after inspection; retention counts at
10% and 50% are printed alongside so the choice can be seen not to matter.

Multipliers come from results/floquet_ops (src/operator_floquet.py): sorted,
the leading one is the phase direction (1 for a limit cycle) and the next is
the weakest transverse contraction per period.

  python src/operator_summary.py [--json out.json]
"""
import argparse, collections, glob, json, os
import numpy as np


def outcome(r, coefs, T, tol=0.2):
    if r["delta"] is None:
        return "diverged"
    if len(coefs) / T < 399:
        return "late-div"
    amp = lambda a, b: np.sqrt(np.mean(coefs[int(a * T):int(b * T), 0] ** 2
                                       + coefs[int(a * T):int(b * T), 1] ** 2))
    ratio = amp(390, 400) / amp(0, 10)
    dl = r.get("delta_late")
    if ratio < 1 - tol or (dl is not None and abs(dl) > 0.5):
        return "collapsed"
    if ratio > 1 / (1 - tol):
        return "grown"
    return "cycle-wp" if (r.get("split_late") or 1) < 1e-7 else "cycle"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/seglen_v2")
    ap.add_argument("--floq", default="results/floquet_ops")
    ap.add_argument("--json", default=None)
    ap.add_argument("--markdown", default=None,
                    help="write the Extended Data table (FNO arms) to this path")
    a = ap.parse_args()
    arms = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{a.res}/*.json")):
        tag = os.path.basename(f)[:-5]
        arm = tag.split("_s")[0]
        j = json.load(open(f)); r = j["rows"][0]; T = j["summary"]["T_snap"]
        cf = f"{a.res}/coefs/{tag}.npz"
        c = np.load(cf)["coefs"] if os.path.exists(cf) else np.zeros((0, 8))
        fl = f"{a.floq}/{tag}.json"
        m = None
        if os.path.exists(fl):
            q = json.load(open(fl))
            # a Jacobian that maps every tangent vector to zero (exponents of
            # -inf) is numerically degenerate, not a measurement
            if q.get("status") == "ok" and np.all(np.isfinite(q["lyap_per_step"])):
                m = sorted(q["multipliers_per_period"], reverse=True)
        arms[arm].append(dict(tag=tag, T=T, K=j["config"]["seg_len"],
                              out=outcome(r, c, T),
                              keep10=outcome(r, c, T, 0.1) in ("cycle", "cycle-wp"),
                              keep50=outcome(r, c, T, 0.5) in ("cycle", "cycle-wp"),
                              d100=r["delta"], dlate=r.get("delta_late"),
                              split=r.get("split_late"), mse=r["train_mse"], mult=m))
    order = lambda arm: (arm.split("-")[0], arm.split("-")[1], int(arm.split("-K")[1].split("-")[0]),
                         arm.split("-b")[1])
    hdr = (f"{'arm':22} {'K/T':>5} {'div':>3} {'ldiv':>4} {'coll':>4} {'cyc':>3} {'wp':>3} "
           f"{'keep10/20/50':>12} {'med|d100|':>9} {'med|dlate| wp':>13} {'spread':>7} "
           f"{'m1':>6} {'m2':>8} {'n_m':>3}")
    print(hdr); print("-" * len(hdr))
    out = []
    for arm in sorted(arms, key=order):
        g = arms[arm]; T = g[0]["T"]
        cnt = collections.Counter(x["out"] for x in g)
        keep20 = cnt["cycle"] + cnt["cycle-wp"]
        d100 = [abs(x["d100"]) for x in g if x["d100"] is not None]
        dwp = [abs(x["dlate"]) for x in g if x["out"] == "cycle-wp"]
        ms = [x["mult"] for x in g if x["mult"]]
        m1 = float(np.median([m[0] for m in ms])) if ms else float("nan")
        m2 = float(np.median([m[1] for m in ms])) if ms else float("nan")
        sp = max(d100) / min(d100) if len(d100) > 1 else float("nan")
        print(f"{arm:22} {g[0]['K']/T:>5.2f} {cnt['diverged']:>3} {cnt['late-div']:>4} "
              f"{cnt['collapsed'] + cnt['grown']:>4} {cnt['cycle']:>3} {cnt['cycle-wp']:>3} "
              f"{sum(x['keep10'] for x in g):>4}/{keep20}/{sum(x['keep50'] for x in g):<4} "
              f"{np.median(d100) if d100 else float('nan'):>9.2e} "
              f"{np.median(dwp) if dwp else float('nan'):>13.2e} {sp:>6.0f}x "
              f"{m1:>6.3f} {m2:>8.2e} {len(ms):>3}")
        out.append(dict(arm=arm, K_over_T=g[0]["K"] / T, n=len(g), outcomes=dict(cnt),
                        keep10=sum(x["keep10"] for x in g), keep20=keep20,
                        keep50=sum(x["keep50"] for x in g),
                        med_d100=float(np.median(d100)) if d100 else None,
                        med_dlate_wp=float(np.median(dwp)) if dwp else None,
                        spread=sp, m1=m1, m2=m2, n_mult=len(ms),
                        mults=[x["mult"] for x in g]))
    # pooled residual vs direct
    print()
    for lab, sel in (("residual", lambda arm: "-fno-" in arm), ("direct", lambda arm: "-fnodir-" in arm),
                     ("noise", lambda arm: "-fnonz-" in arm), ("noise10", lambda arm: "-fnonz10-" in arm),
                     ("noise30", lambda arm: "-fnonz30-" in arm)):
        g = [x for arm in arms if sel(arm) for x in arms[arm]]
        ms = [x["mult"] for x in g if x["mult"]]
        wp = sum(x["out"] == "cycle-wp" for x in g)
        contract = sum(m[1] < 0.5 for m in ms)
        print(f"  {lab:8}: well-posed cycle {wp}/{len(g)}; models with multipliers {len(ms)}; "
              f"m2 < 0.5 in {contract}/{len(ms)}; median m2 "
              f"{np.median([m[1] for m in ms]) if ms else float('nan'):.3g}; "
              f"median m1 {np.median([m[0] for m in ms]) if ms else float('nan'):.3g}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    if a.markdown:
        name = {"ns": "Navier–Stokes", "wake": "wake", "cardiac": "cardiac"}
        par = {"fno": "residual", "fnodir": "direct", "fnonz": "residual + noise 10⁻³",
               "fnonz10": "residual + noise 10⁻²", "fnonz30": "residual + noise 3 × 10⁻²"}
        sci = lambda v: "—" if v is None or not np.isfinite(v) else (
            f"{v:.2g}" if 0.01 <= v < 10 else
            "{:.2f} × 10{}".format(v / 10 ** int(np.floor(np.log10(v))),
                                  "".join("⁻⁰¹²³⁴⁵⁶⁷⁸⁹"[0 if c == "-" else int(c) + 1]
                                          for c in str(int(np.floor(np.log10(v)))))))
        lines = ["| system | parameterisation | K/T | batch | diverged | collapsed or wrong rhythm "
                 "| cycle, not well-posed | well-posed cycle | median train MSE "
                 "| median \\|δ\\| (runs keeping a cycle) | transverse multiplier |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        keyorder = lambda arm: ({"wake": 0, "ns": 1, "cardiac": 2}[arm.split("-")[0]],
                                {"fno": 0, "fnonz": 1, "fnonz10": 2, "fnonz30": 3,
                                 "fnodir": 4}.get(arm.split("-")[1], 5),
                                int(arm.split("-K")[1].split("-")[0]), arm.split("-b")[1])
        for arm in sorted((x for x in arms if not x.split("-")[1].startswith("unet")), key=keyorder):
            g = arms[arm]; sysn, ar = arm.split("-")[0], arm.split("-")[1]
            cnt = collections.Counter(x["out"] for x in g)
            kept = [abs(x["dlate"]) for x in g if x["out"] in ("cycle", "cycle-wp")]
            ms = [x["mult"][1] for x in g if x["mult"]]
            mse = np.median([x["mse"] for x in g])
            lines.append(f"| {name[sysn]} | {par[ar]} | {g[0]['K'] / g[0]['T']:.2f} | "
                         f"{arm.split('-b')[1]} | {cnt['diverged'] + cnt['late-div']} | "
                         f"{cnt['collapsed'] + cnt['grown']} | {cnt['cycle']} | {cnt['cycle-wp']} | "
                         f"{sci(mse)} | {sci(float(np.median(kept))) if kept else '—'} | "
                         f"{sci(float(np.median(ms))) if ms else '—'} |")
        open(a.markdown, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
