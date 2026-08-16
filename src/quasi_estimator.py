"""
Is delta_single large because the MODEL is wrong, or because the single-period
FIT is ill-posed on a two-frequency signal?

delta_single = (T_pred - T_true)/T_true with BOTH measured the same way, so a
consistent estimator bias would cancel. If it does not cancel, the fit is
picking different fundamentals for the two signals.
"""
import numpy as np, math, sys
sys.path.insert(0, "src")
from quasiperiodic import CoupledSL
from period import period_from_signal

dt = 0.05
t = np.arange(0, 400, dt)
print("what fundamental does the single-period fit choose, and how stable is it?\n")
print("{:>6} {:>12} {:>12} {:>12} {:>10} {:>26}".format(
    "eps", "w_true[0]", "w_true[1]", "fit on truth", "as ratio", "identification"))
print("-" * 84)
for eps in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]:
    S = CoupledSL(eps=eps)
    y = S.rollout([0.7, 0.0, 0.7, 0.3], np.arange(0, 900, dt))[-1]
    tr = S.rollout(y, t)
    obs = tr[:, 0] + tr[:, 2]
    T = period_from_signal(t, obs)
    wf = 2 * math.pi / T
    # split the record in halves: a well-posed fit gives the same answer on both
    Ta = period_from_signal(t[:len(t)//2], obs[:len(t)//2])
    Tb = period_from_signal(t[len(t)//2:], obs[len(t)//2:])
    instab = abs(Ta - Tb) / T
    from quasiperiodic import peak_frequencies, two_freq_fit
    p = peak_frequencies(t, obs, 2)
    w, _ = two_freq_fit(t, obs, p if len(p) == 2 else [1.25, 2.02])
    lab = []
    for name, wv in [("w1", w[0]), ("w2", w[1])]:
        for k in range(1, 9):
            if abs(wf * k - wv) < 0.02 * wv:
                lab.append(f"{name}={k}x fund")
    print("{:>6.2f} {:>12.5f} {:>12.5f} {:>12.5f} {:>10.4f} {:>26}".format(
        eps, w[0], w[1], wf, w[1] / w[0] if w[0] else 0,
        ", ".join(lab) if lab else "neither is a harmonic"))
    print("{:>6} {:>62} half-vs-half instability {:>8.2e}".format("", "", instab))
