"""
Audit the period estimator itself.

The Hilbert phase-regression estimator, given a signal with EXACTLY the right
period, returns delta = +1.06e-5. Many reported deltas sit at or below that, so
the measurement may be reporting its own bias rather than the model's error.
Compare candidate estimators on signals of known period.
"""
import math
import numpy as np
from scipy.signal import hilbert
from scipy.optimize import minimize_scalar

T = 2 * math.pi
W = 2 * math.pi / T


def hilbert_reg(t, s, edge=0.1):
    s = np.asarray(s, float); s = s - s.mean()
    ph = np.unwrap(np.angle(hilbert(s)))
    k = max(1, int(edge * len(s)))
    return 2 * math.pi / abs(np.polyfit(t[k:-k], ph[k:-k], 1)[0])


def poincare(t, s):
    """Upward mean-crossings, linearly interpolated; exact for clean signals."""
    s = np.asarray(s, float); s = s - s.mean()
    cr = []
    for i in range(len(s) - 1):
        if s[i] < 0 <= s[i + 1]:
            cr.append(t[i] + (t[i + 1] - t[i]) * (-s[i] / (s[i + 1] - s[i])))
    if len(cr) < 3:
        return float("nan")
    cr = np.array(cr)
    return float(np.polyfit(np.arange(len(cr)), cr, 1)[0])   # fit, not mean


def harmonic_fit(t, s, K=6, w0=None):
    """Least squares over a Fourier series with unknown fundamental."""
    s = np.asarray(s, float)
    w0 = w0 or W

    def obj(w):
        cols = [np.ones_like(t)]
        for k in range(1, K + 1):
            cols += [np.cos(k * w * t), np.sin(k * w * t)]
        P = np.stack(cols, 1)
        return np.linalg.norm(P @ np.linalg.lstsq(P, s, rcond=None)[0] - s)

    r = minimize_scalar(obj, bounds=(w0 * .8, w0 * 1.2), method="bounded",
                        options=dict(xatol=1e-14))
    return 2 * math.pi / r.x


EST = [("hilbert (current)", hilbert_reg), ("poincare", poincare),
       ("harmonic fit", harmonic_fit)]
rng = np.random.default_rng(0)
dt = 0.2027
n = int(100 * T / dt)
t = np.arange(n + 1) * dt

cases = [("pure sine", [(1.0, 1)]),
         ("+ 25% 2nd harmonic", [(1.0, 1), (0.25, 2)]),
         ("wake-like (1, .25, .08)", [(1.0, 1), (0.25, 2), (0.08, 3)]),
         ("harmonic-rich", [(1.0, 1), (0.5, 2), (0.3, 3), (0.15, 4)])]

print(f"apparent |delta| when the true period is exactly {T:.10f}\n")
hdr = "{:>26} {:>16} {:>13} {:>15}".format("signal", *[e[0] for e in EST])
print(hdr); print("-" * len(hdr))
for label, comps in cases:
    s = sum(a * np.cos(k * W * t + 0.3 * k) for a, k in comps)
    out = []
    for _, f in EST:
        try:
            out.append(abs((f(t, s) - T) / T))
        except Exception:
            out.append(float("nan"))
    print("{:>26} {:>16.2e} {:>13.2e} {:>15.2e}".format(label, *out))

print("\nwith 1e-3 additive noise (the case Hilbert was chosen for):")
print(hdr); print("-" * len(hdr))
for label, comps in cases:
    s = sum(a * np.cos(k * W * t + 0.3 * k) for a, k in comps)
    s = s + 1e-3 * rng.standard_normal(len(s))
    out = []
    for _, f in EST:
        try:
            out.append(abs((f(t, s) - T) / T))
        except Exception:
            out.append(float("nan"))
    print("{:>26} {:>16.2e} {:>13.2e} {:>15.2e}".format(label, *out))
