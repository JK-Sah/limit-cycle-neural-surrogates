"""
Is the split-half well-posedness test robust to the length of the record?

Section 9 rests on a ratio: the single-period fit disagrees with itself by
3.8e-1 on a quasi-periodic record and by 1.8e-9 once the oscillators lock, and
that separation is quoted in the abstract. The original check
(`quasi_estimator.py`) printed the number and saved nothing, and it used one
record length, so two things were unknown: whether the result is reproducible
from stored output, and whether the separation is a property of the attractor
or of the 400-time-unit window it was measured in.

The system is deterministic -- an ODE integrated from a fixed initial condition,
no training, no seed -- so there is no seed noise to average over here. The
free parameter is the observation window, and that is what this sweeps. A
diagnostic whose verdict flips with record length would be no diagnostic at all,
since a practitioner applying it has whatever record they happen to have.
"""
import argparse, json, math, sys
import numpy as np

sys.path.insert(0, "src")
from quasiperiodic import CoupledSL
from period import period_from_signal


def split_half(eps, T_record, dt=0.05, T_transient=900.0):
    S = CoupledSL(eps=eps)
    y = S.rollout([0.7, 0.0, 0.7, 0.3], np.arange(0, T_transient, dt))[-1]
    t = np.arange(0, T_record, dt)
    tr = S.rollout(y, t)
    obs = tr[:, 0] + tr[:, 2]
    T = period_from_signal(t, obs)
    h = len(t) // 2
    Ta = period_from_signal(t[:h], obs[:h])
    Tb = period_from_signal(t[h:], obs[h:])
    return dict(eps=eps, T_record=T_record, T_fit=float(T),
                T_first_half=float(Ta), T_second_half=float(Tb),
                instability=float(abs(Ta - Tb) / T),
                periods_in_record=float(T_record / T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.8])
    ap.add_argument("--records", type=float, nargs="+",
                    default=[200, 400, 800, 1600])
    ap.add_argument("--out", default="quasi_split.json")
    a = ap.parse_args()

    rows = []
    print("%6s %10s %10s %14s %12s" %
          ("eps", "record", "periods", "instability", "regime"))
    print("-" * 56)
    for eps in a.eps:
        for R in a.records:
            r = split_half(eps, R)
            rows.append(r)
            print("%6.2f %10.0f %10.1f %14.2e %12s" %
                  (eps, R, r["periods_in_record"], r["instability"],
                   "locked" if eps >= 0.4 else "quasi-periodic"))
        print()

    # the claim under test: does the quasi/locked separation hold at every
    # record length, or only at the one originally used?
    print("separation between regimes, by record length:")
    print("%10s %14s %14s %10s" % ("record", "max quasi", "max locked", "ratio"))
    for R in a.records:
        q = [r["instability"] for r in rows if r["T_record"] == R and r["eps"] < 0.4]
        l = [r["instability"] for r in rows if r["T_record"] == R and r["eps"] >= 0.4]
        if q and l:
            print("%10.0f %14.2e %14.2e %9.1e" %
                  (R, max(q), max(l), max(q) / max(l)))
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
