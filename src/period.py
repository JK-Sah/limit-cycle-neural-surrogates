"""
Period estimation. Single source of truth for the whole study.

This exists because the previous estimator was wrong in a way that mattered.
Phase regression on the analytic (Hilbert) signal is unbiased for a pure
sinusoid but not for anything with harmonics: the instantaneous phase of
cos(wt) + a cos(2wt) is not linear in t, so the fitted slope is biased. Measured
against signals of exactly known period:

    signal                    Hilbert     Poincare   harmonic fit
    pure sine                4.19e-09     2.02e-07       2.96e-09
    + 25% 2nd harmonic       1.08e-05     2.35e-06       1.44e-09
    wake-like                1.23e-05     7.77e-07       6.45e-09
    harmonic-rich            2.65e-05     5.09e-06       2.58e-09

The bias scales with harmonic content, so it differs between methods and
contaminates any cross-method comparison at delta below ~1e-5. Several reported
numbers sat below that.

`period_from_signal` is the harmonic fit: least squares over a Fourier series
with unknown fundamental. Exact for a periodic signal of any waveform, and
1e-9 to 1e-7 in the audit above. The initial frequency comes from the
periodogram, never from a reference value, so the estimator cannot be handed
the answer it is meant to measure.
"""
import math
import numpy as np
from scipy.signal import hilbert
from scipy.optimize import minimize_scalar

__all__ = ["period_from_signal", "period_hilbert", "period_poincare",
           "audit_estimators"]


def _fft_peak_omega(t, s):
    """Angular frequency of the dominant spectral peak, parabolic-refined."""
    dt = float(np.median(np.diff(t)))
    x = np.asarray(s, float) - np.mean(s)
    P = np.abs(np.fft.rfft(x)) ** 2
    fr = np.fft.rfftfreq(len(x), d=dt) * 2 * math.pi
    if len(P) < 4:
        return None
    P[0] = 0.0
    k = int(np.argmax(P))
    if 0 < k < len(P) - 1:                       # parabolic interpolation
        a, b, c = P[k - 1], P[k], P[k + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if abs(denom) > 0 else 0.0
        return float(fr[k] + shift * (fr[1] - fr[0]))
    return float(fr[k])


def period_from_signal(t, s, K=6, bracket=0.25, w_init=None):
    """
    Period by least squares over a Fourier series with unknown fundamental.

    K is the number of harmonics retained. The fundamental is initialised from
    the periodogram and refined by bounded scalar minimisation.
    """
    t = np.asarray(t, float)
    s = np.asarray(s, float)
    if not np.isfinite(s).all():
        return float("nan")
    w0 = w_init if w_init is not None else _fft_peak_omega(t, s)
    if not w0 or not np.isfinite(w0) or w0 <= 0:
        return float("nan")
    tc = t - t.mean()

    def obj(w):
        cols = [np.ones_like(tc)]
        for k in range(1, K + 1):
            cols += [np.cos(k * w * tc), np.sin(k * w * tc)]
        P = np.stack(cols, 1)
        with np.errstate(over="ignore", under="ignore", invalid="ignore",
                         divide="ignore"):
            return float(np.linalg.norm(
                P @ np.linalg.lstsq(P, s, rcond=None)[0] - s))

    # The residual is multimodal in w: subharmonics and near-harmonics are
    # local minima, and a bounded scalar search lands in whichever one its
    # first probes happen to bracket. Observed directly -- seeding with the
    # exact true frequency returned a period 4.3% off while a periodogram seed
    # returned it to 6e-9. Scan coarsely for the global basin, then refine.
    lo, hi = w0 * (1 - bracket), w0 * (1 + bracket)
    grid = np.linspace(lo, hi, 1200)
    vals = np.array([obj(w) for w in grid])
    j = int(np.argmin(vals))
    a = grid[max(0, j - 1)]
    b = grid[min(len(grid) - 1, j + 1)]
    r = minimize_scalar(obj, bounds=(a, b), method="bounded",
                        options=dict(xatol=1e-14))
    w_best = r.x if obj(r.x) <= vals[j] else grid[j]
    return float(2 * math.pi / w_best)


def period_hilbert(t, s, edge_frac=0.1):
    """The previous estimator. Kept so the bias can be reproduced, not used."""
    s = np.asarray(s, float)
    s = s - s.mean()
    ph = np.unwrap(np.angle(hilbert(s)))
    k = max(1, int(edge_frac * len(s)))
    return float(2 * math.pi / abs(np.polyfit(t[k:-k], ph[k:-k], 1)[0]))


def period_poincare(t, s):
    """Upward mean-crossings, linearly interpolated, fitted against index."""
    s = np.asarray(s, float)
    s = s - s.mean()
    cr = [t[i] + (t[i + 1] - t[i]) * (-s[i] / (s[i + 1] - s[i]))
          for i in range(len(s) - 1) if s[i] < 0 <= s[i + 1]]
    if len(cr) < 3:
        return float("nan")
    return float(np.polyfit(np.arange(len(cr)), np.array(cr), 1)[0])


def audit_estimators(n_periods=100, dt=0.2027, noise=0.0, seed=0):
    """Apparent |delta| on signals whose period is exactly 2*pi."""
    rng = np.random.default_rng(seed)
    T = 2 * math.pi
    w = 2 * math.pi / T
    t = np.arange(int(n_periods * T / dt) + 1) * dt
    cases = {"pure sine": [(1.0, 1)],
             "+25% 2nd": [(1.0, 1), (0.25, 2)],
             "wake-like": [(1.0, 1), (0.25, 2), (0.08, 3)],
             "harmonic-rich": [(1.0, 1), (0.5, 2), (0.3, 3), (0.15, 4)]}
    out = {}
    for label, comps in cases.items():
        s = sum(a * np.cos(k * w * t + 0.3 * k) for a, k in comps)
        if noise:
            s = s + noise * rng.standard_normal(len(s))
        out[label] = {name: abs((f(t, s) - T) / T) for name, f in
                      [("harmonic", period_from_signal),
                       ("hilbert", period_hilbert),
                       ("poincare", period_poincare)]}
    return out


if __name__ == "__main__":
    for nz in [0.0, 1e-3]:
        print(f"noise = {nz:g}")
        res = audit_estimators(noise=nz)
        names = ["harmonic", "hilbert", "poincare"]
        print("{:>16}".format("signal") + "".join(f"{n:>13}" for n in names))
        for lab, d in res.items():
            print("{:>16}".format(lab) + "".join(f"{d[n]:>13.2e}"
                                                 for n in names))
        print()
