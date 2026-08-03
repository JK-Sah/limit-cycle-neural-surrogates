"""
How far does anchoring get you, and what sets the floor?

If the diagnosis is right, the frequency error of an anchored model is limited by
how well the frequency can be MEASURED from the available data -- not by the fit.
So delta should fall as the supervision weight rises, and plateau at a floor set
by observation noise.

Grid: supervision weight x observation noise x seed.
"""
import argparse, json
import numpy as np
from t0_falsify import SL
from t0_fix import run, measured_invariants

W_SUP = [1.0, 1e2, 1e4]
NOISE = [0.0, 1e-3, 1e-2]
SEEDS = [0, 1]
GRID = [(w, n, s) for w in W_SUP for n in NOISE for s in SEEDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--periods", type=int, default=200)
    ap.add_argument("--out", default="anchor_sweep.json")
    a = ap.parse_args()
    if a.count:
        print(len(GRID)); return

    s = SL()
    todo = [GRID[a.task_id]] if a.task_id is not None else GRID
    rows = []
    for w, noise, seed in todo:
        Tm, rm = measured_invariants(s, noise=noise, seed=seed)
        floor = abs(Tm - s.T) / s.T
        print(f"w_sup={w:.0e} obs_noise={noise:.0e} seed={seed} "
              f"| measurement floor |dT/T| = {floor:.3e}", flush=True)
        r = run("anchored+sup", s, 64, 64, seed, a.epochs,
                n_periods=a.periods, obs_noise=noise, w_sup=w)
        r["measurement_floor"] = floor
        rows.append(r)
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
