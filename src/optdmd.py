"""
Optimized DMD by variable projection (Askham & Kutz 2018), plus a
harmonic-locked variant that is the structurally correct estimator for a
limit cycle.

Exact DMD minimises one-step prediction error and its eigenvalues are known to
be biased. There is also a specific prediction to test. The architecture sweep
found `linear-sgd` -- DMD's own hypothesis class fitted by Adam on a 25-step
rollout -- beating exact DMD by 6x, and attributed that to the OBJECTIVE rather
than the optimiser: one-step least squares does not penalise accumulated phase
error, a multi-step loss does. Optimized DMD fits the same exponential model
against the whole trajectory by a completely different algorithm
(Levenberg-Marquardt on the exponents, linear coefficients projected out). If
the attribution is right, it should also beat exact DMD.

Two estimators:

  free       p independent (sigma, omega) pairs. The textbook method.
  harmonic   one fundamental omega, harmonics at k*omega, sigma = 0. This is
             what a limit cycle actually is, and it reduces the nonlinear part
             of the fit to a single scalar.

Real arithmetic throughout: each mode contributes exp(sigma t) cos(omega t) and
exp(sigma t) sin(omega t), so the fit stays real and conjugate symmetry is
automatic. Time is centred; over ~50 periods an uncentred origin makes
exp(sigma t) span hundreds-fold and the solve overflows.
"""
import math
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _coeffs(Phi, X):
    return np.linalg.lstsq(Phi, X, rcond=None)[0]


def _resid_norm(Phi, X):
    B = _coeffs(Phi, X)
    return float(np.linalg.norm(Phi @ B - X)), B


# ------------------------------------------------------------------ free form
def _basis_free(params, t, p):
    sig, om = params[:p], params[p:]
    E = np.exp(np.outer(t, sig))
    ang = np.outer(t, om)
    return np.concatenate([E * np.cos(ang), E * np.sin(ang)], axis=1)


def _fft_init(X, t, p, min_sep_frac=0.25):
    """
    Initial frequencies from the strongest spectral peaks, with real
    separation. A permissive separation test lets spectral leakage around one
    peak be selected as several distinct modes: an earlier version returned
    omega = [0.99, 2.00, 1.05, 0.93] for a signal whose true content was
    [1, 2, 3, 0.7], and the fit then spent two modes on one component.
    """
    dt = float(np.median(np.diff(t)))
    x = X - X.mean(0)
    P = (np.abs(np.fft.rfft(x, axis=0)) ** 2).sum(1)
    fr = np.fft.rfftfreq(len(x), d=dt) * 2 * math.pi
    P[0] = 0.0
    chosen = []
    for k in np.argsort(P)[::-1]:
        if fr[k] <= 1e-9:
            continue
        if all(abs(fr[k] - f) > min_sep_frac * min(fr[k], f) for f in chosen):
            chosen.append(fr[k])
        if len(chosen) == p:
            break
    while len(chosen) < p:                      # pad if the spectrum is sparse
        chosen.append(chosen[-1] * 1.5 if chosen else 1.0)
    return np.array(chosen)


def optdmd_free(X, t, p, mu0=None, sigma_bound=0.2, max_nfev=600):
    t = np.asarray(t, float); t = t - t.mean()
    # Cap sigma so exp(sigma t) cannot overflow while the optimiser explores.
    # Over the ~1500-snapshot spans used here a bound of 0.2 implies exp(31).
    span = float(np.abs(t).max()) or 1.0
    sigma_bound = min(sigma_bound, 20.0 / span)
    if mu0 is None:
        om0, sg0 = _fft_init(X, t, p), np.zeros(p)
    else:
        sg0 = np.clip(np.real(mu0), -sigma_bound * .5, sigma_bound * .5)
        om0 = np.abs(np.imag(mu0))
    lo = np.concatenate([np.full(p, -sigma_bound), np.full(p, 1e-6)])
    hi = np.concatenate([np.full(p, sigma_bound), np.full(p, 20.0)])
    x0 = np.clip(np.concatenate([sg0, om0]), lo + 1e-12, hi - 1e-12)

    def resid(q):
        return (_basis_free(q, t, p) @ _coeffs(_basis_free(q, t, p), X)
                - X).ravel()

    sol = least_squares(resid, x0, bounds=(lo, hi), method="trf",
                        max_nfev=max_nfev, xtol=1e-14, ftol=1e-14)
    sig, om = sol.x[:p], sol.x[p:]
    Phi = _basis_free(sol.x, t, p)
    _, B = _resid_norm(Phi, X)
    amp = np.array([np.linalg.norm(B[k]) + np.linalg.norm(B[k + p])
                    for k in range(p)])
    return sig, om, amp, float(np.linalg.norm(sol.fun))


# -------------------------------------------------------------- harmonic form
def _basis_harm(omega, t, K):
    cols = [np.ones_like(t)]
    for k in range(1, K + 1):
        cols += [np.cos(k * omega * t), np.sin(k * omega * t)]
    return np.stack(cols, axis=1)


def optdmd_harmonic(X, t, K, om0, bracket_frac=0.15):
    """
    Fit one fundamental with K harmonics and no growth. The nonlinear part is a
    single scalar, so this cannot spend modes on spurious content the way the
    free form can.
    """
    t = np.asarray(t, float); t = t - t.mean()

    def obj(w):
        Phi = _basis_harm(w, t, K)
        return _resid_norm(Phi, X)[0]

    lo, hi = om0 * (1 - bracket_frac), om0 * (1 + bracket_frac)
    sol = minimize_scalar(obj, bounds=(lo, hi), method="bounded",
                          options=dict(xatol=1e-12))
    return float(sol.x), float(sol.fun)


def dominant_period(om, amp, T_guess, rel_window=0.35):
    """
    Pick the mode whose period is nearest the reference among those within a
    generous window, breaking ties by amplitude. An earlier version filtered on
    amplitude first at a 5% threshold and discarded the true fundamental, whose
    amplitude had collapsed to 0.2% because two other modes had converged onto
    the same spurious component.
    """
    per = 2 * math.pi / np.maximum(om, 1e-12)
    near = [k for k in range(len(om))
            if abs(per[k] - T_guess) < rel_window * T_guess]
    if not near:
        near = list(range(len(om)))
    k = max(near, key=lambda k: amp[k])
    return float(per[k]), int(k)
