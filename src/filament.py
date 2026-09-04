"""
Inextensible Euler-Bernoulli filament: the structural half of the flexible-FSI
case.

Why this case exists. Every system measured so far inherits its period from the
flow: the cylinder wake sheds at a frequency set by Re, and the surrogate's job
is to reproduce one number it was shown. A flapping filament is the first case
where the structure carries its own timescale -- bending rigidity and mass ratio
set a structural frequency, the flow sets another, and the limit cycle that
emerges is a property of the coupling. If the phase-drift argument survives
there, it is not an artefact of problems whose frequency is handed to the model.

Governing equations, following Huang, Shin & Sung (JCP 226:2206, 2007):

    rho1 X_tt = d/ds (T X_s) - KB X_ssss - F        (momentum)
    X_s . X_s = 1                                   (inextensibility)

with T the tension acting as the Lagrange multiplier for inextensibility and F
the Lagrangian force the fluid exerts per unit length. Differentiating the
constraint twice in time and substituting the momentum equation turns T into a
1-D boundary value problem solved once per step:

    T_ss - |X_ss|^2 T = -rho1 |X_st|^2 + X_s . d/ds (KB X_ssss + F)

The bending term is the stiff one: its highest resolved mode scales as
KB/(ds^4), so an explicit treatment would fix the timestep at something far
below the lattice step the flow solver wants. It is taken implicitly, which
makes the scheme unconditionally stable in bending and leaves the timestep set
by the flow.

This module is the structure alone. It is separated from the coupling so that
the bending operator and its boundary conditions can be checked against a case
with a closed-form answer before any fluid is involved -- see `validate`, which
recovers the clamped-free beam eigenvalues.
"""
import argparse
import numpy as np
from scipy.linalg import lu_factor, lu_solve, solve_banded


# --------------------------------------------------------------- operators
def bending_operator(N, ds, root="clamped"):
    """Fourth-derivative operator on nodes j = 1..N, split by dependency.

    Returns (A, a0, at) such that for the interior/tip rows j = 1..N

        (d^4 X / ds^4)_j = (A @ X[1:])_j + a0_j * X_0 + at_j * t0

    where X_0 is the prescribed attachment point and t0 the prescribed root
    tangent. Keeping the X_0 and t0 dependence separate is what lets the same
    operator serve both the eigenvalue check (X_0 = 0, t0 = 0) and the moving
    filament, where the root sits at a real point in space.

    The free end at j = N carries the natural conditions X_ss = X_sss = 0,
    imposed through ghost nodes:

        X_{N+1} = 2 X_N - X_{N-1}
        X_{N+2} = 4 X_N - 4 X_{N-1} + X_{N-2}

    At the root, "clamped" fixes position and tangent, giving the ghost
    X_{-1} = X_1 - 2 ds t0; "pinned" fixes position and lets the tangent rotate,
    giving X_{-1} = 2 X_0 - X_1.
    """
    A = np.zeros((N, N))
    a0 = np.zeros(N)
    at = np.zeros(N)

    def put(row, col, val):
        if col == 0:
            a0[row - 1] += val
        elif 1 <= col <= N:
            A[row - 1, col - 1] += val

    for j in range(1, N + 1):
        stencil = {j - 2: 1.0, j - 1: -4.0, j: 6.0, j + 1: -4.0, j + 2: 1.0}
        for col, val in stencil.items():
            if col == -1:                       # ghost below the root
                if root == "clamped":
                    put(j, 1, val)
                    at[j - 1] += -2.0 * ds * val
                else:                           # pinned
                    put(j, 0, 2.0 * val)
                    put(j, 1, -1.0 * val)
            elif col == N + 1:                  # first ghost past the free end
                put(j, N, 2.0 * val)
                put(j, N - 1, -1.0 * val)
            elif col == N + 2:                  # second ghost past the free end
                put(j, N, 4.0 * val)
                put(j, N - 1, -4.0 * val)
                put(j, N - 2, 1.0 * val)
            else:
                put(j, col, val)
    d4 = ds ** 4
    return A / d4, a0 / d4, at / d4


def bending_matrix(N, ds, root="clamped"):
    """Interior block alone, for the fixed-root eigenvalue check."""
    return bending_operator(N, ds, root)[0]


def _d1(v, ds):
    """d/ds by central differences, one-sided at the ends. v is [N+1, 2]."""
    out = np.empty_like(v)
    out[1:-1] = (v[2:] - v[:-2]) / (2 * ds)
    out[0] = (-3 * v[0] + 4 * v[1] - v[2]) / (2 * ds)
    out[-1] = (3 * v[-1] - 4 * v[-2] + v[-3]) / (2 * ds)
    return out


def _d2(v, ds):
    """d^2/ds^2 by central differences, one-sided at the ends."""
    out = np.empty_like(v)
    out[1:-1] = (v[2:] - 2 * v[1:-1] + v[:-2]) / ds ** 2
    out[0] = (2 * v[0] - 5 * v[1] + 4 * v[2] - v[3]) / ds ** 2
    out[-1] = (2 * v[-1] - 5 * v[-2] + 4 * v[-3] - v[-4]) / ds ** 2
    return out


# ---------------------------------------------------------------- dynamics
class Filament:
    """Inextensible filament, root fixed, tip free.

    Inextensibility is imposed on the DISCRETE segments rather than by
    discretizing the continuum tension PDE. Differentiating |d_j|^2 = ds^2 twice
    in time, with d_j = X_j - X_{j-1}, gives

        d_j . (d/dt)^2 d_j + |d(d_j)/dt|^2 = 0

    and substituting the discrete momentum equation turns that into a
    tridiagonal system for the segment tensions. The advantage over discretizing
    the continuum equation is that the constraint being enforced is exactly the
    one the position update can violate, so `max_stretch` below is a genuine
    check on the solver rather than a discretization error that has to be
    tolerated. The awkward continuum boundary condition on T at the root also
    disappears: node 0 is fixed, so its acceleration is zero and it simply drops
    out of the system.

    Bending is taken implicitly. Its stiffest resolved mode scales as KB/ds^4,
    which for the lattice spacings this will run at is far faster than anything
    physical; treating it explicitly would set the timestep for no good reason.
    Tension is explicit, which is what leaves a residual stretch to monitor.
    """

    def __init__(self, N=40, L=1.0, rho1=1.0, KB=1e-3, dt=1.0,
                 root=(0.0, 0.0), tangent=(1.0, 0.0), root_type="clamped"):
        self.N, self.L, self.ds = N, L, L / N
        self.rho1, self.KB, self.dt = rho1, KB, dt
        self.root_type = root_type
        self.X0 = np.asarray(root, dtype=float)
        self.t0 = np.asarray(tangent, dtype=float)
        self.t0 = self.t0 / np.linalg.norm(self.t0)

        self.A, self.a0, self.at = bending_operator(N, self.ds, root_type)
        # implicit-bending operator, factorized once
        M = (rho1 / dt ** 2) * np.eye(N) + KB * self.A
        self.lu = lu_factor(M)
        # constant contribution of the prescribed root position and tangent
        self.bend_const = KB * (np.outer(self.a0, self.X0)
                                + np.outer(self.at, self.t0))

        s = np.arange(N + 1) * self.ds
        self.X = self.X0[None, :] + s[:, None] * self.t0[None, :]
        self.Xprev = self.X.copy()

    # -- diagnostics -------------------------------------------------------
    def segments(self, X=None):
        X = self.X if X is None else X
        return X[1:] - X[:-1]

    def max_stretch(self):
        """max | |d_j|/ds - 1 |. Zero is exact inextensibility."""
        return float(np.abs(np.linalg.norm(self.segments(), axis=1)
                            / self.ds - 1.0).max())

    def tip(self):
        return self.X[-1].copy()

    # -- solver ------------------------------------------------------------
    def _bending(self, X):
        """KB * d^4X/ds^4 on nodes 1..N, including the prescribed-root terms."""
        return self.KB * (self.A @ X[1:]) + self.bend_const

    def tension(self, B):
        """Segment tensions from the twice-differentiated length constraint.

        B is the non-tension force on nodes 1..N (bending plus fluid), already
        signed as it appears in rho1 * Xddot = tension + B.
        """
        N, ds = self.N, self.ds
        d = self.segments()                      # [N, 2], index 0 -> segment 1
        dprev = self.segments(self.Xprev)
        ddot = (d - dprev) / self.dt

        dot_self = np.einsum("ij,ij->i", d, d)               # |d_j|^2
        dot_next = np.einsum("ij,ij->i", d[:-1], d[1:])      # d_j . d_{j+1}

        lo = np.zeros(N); mid = np.zeros(N); up = np.zeros(N); rhs = np.zeros(N)
        # j = 1: node 0 is fixed, so d''_1 = Xdd_1 and only one tension acts
        mid[0] = -dot_self[0]
        if N > 1:
            up[0] = dot_next[0]
        rhs[0] = (-self.rho1 * ds ** 2 * float(ddot[0] @ ddot[0])
                  - ds ** 2 * float(d[0] @ B[0]))
        # j >= 2
        for j in range(1, N):
            lo[j] = dot_next[j - 1]
            mid[j] = -2.0 * dot_self[j]
            if j < N - 1:
                up[j] = dot_next[j]
            rhs[j] = (-self.rho1 * ds ** 2 * float(ddot[j] @ ddot[j])
                      - ds ** 2 * float(d[j] @ (B[j] - B[j - 1])))

        ab = np.zeros((3, N))
        ab[0, 1:] = up[:-1]
        ab[1] = mid
        ab[2, :-1] = lo[1:]
        return solve_banded((1, 1), ab, rhs)

    def tension_force(self, T):
        """(T_{j+1} d_{j+1} - T_j d_j) / ds^2 on nodes 1..N, with T_{N+1} = 0."""
        d = self.segments()
        out = -T[:, None] * d
        out[:-1] += T[1:, None] * d[1:]
        return out / self.ds ** 2

    def project(self, tol=1e-13, iters=8):
        """Pull the segment lengths back onto |d_j| = ds exactly.

        The tension is evaluated at the old state, so the implicit position
        update satisfies the constraint only to the order of that lag. The
        residual is small per step but it does not average out: over a long
        rollout the filament slowly stretches, which would show up as a drifting
        period -- precisely the quantity this whole study measures. A Newton
        projection removes it. The constraint gradients have the same
        tridiagonal structure as the tension solve, so this costs one extra
        banded solve per iteration and converges in two or three.
        """
        N, ds2 = self.N, self.ds ** 2
        for _ in range(iters):
            d = self.segments()
            g = np.einsum("ij,ij->i", d, d) - ds2
            if np.abs(g).max() <= tol * ds2:
                break
            dot_self = np.einsum("ij,ij->i", d, d)
            dot_next = np.einsum("ij,ij->i", d[:-1], d[1:])
            mid = 4.0 * dot_self
            mid[1:] += 4.0 * dot_self[1:]        # node j-1 also moves for j >= 2
            off = -4.0 * dot_next
            ab = np.zeros((3, N))
            ab[0, 1:] = off
            ab[1] = mid
            ab[2, :-1] = off
            lam = solve_banded((1, 1), ab, -g)
            # dX_k = sum_j lam_j * dg_j/dX_k, with node 0 held fixed
            dX = np.zeros((N, 2))
            dX += (2.0 * lam)[:, None] * d                  # dg_j/dX_j
            dX[:-1] -= (2.0 * lam[1:])[:, None] * d[1:]     # dg_{j+1}/dX_j
            self.X[1:] += dX
        return float(np.abs(np.linalg.norm(self.segments(), axis=1)
                            / self.ds - 1.0).max())

    def step(self, F=None, project=True):
        """Advance one step. F is the fluid force per unit length on nodes 1..N."""
        Fl = np.zeros((self.N, 2)) if F is None else np.asarray(F, dtype=float)
        B = -self._bending(self.X) + Fl
        T = self.tension(B)
        rhs = (self.rho1 / self.dt ** 2) * (2 * self.X[1:] - self.Xprev[1:]) \
            + self.tension_force(T) + Fl - self.bend_const
        Xnew = lu_solve(self.lu, rhs)
        self.Xprev = self.X
        self.X = np.vstack([self.X0[None, :], Xnew])
        if project:
            self.project()
        return T


# --------------------------------------------------------------- validation
def cantilever_beta_L(n_modes=5):
    """Roots of cos(bL) cosh(bL) + 1 = 0 -- the clamped-free beam eigenvalues.

    Solved by bisection on brackets around the asymptotic (2n-1)pi/2 rather than
    hard-coded, so the reference is generated rather than transcribed.
    """
    f = lambda x: np.cos(x) * np.cosh(x) + 1.0
    roots = []
    n = 1
    while len(roots) < n_modes:
        guess = (2 * n - 1) * np.pi / 2
        lo, hi = guess - 0.6, guess + 0.6
        if f(lo) * f(hi) < 0:
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if f(lo) * f(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append(0.5 * (lo + hi))
        n += 1
    return np.array(roots)


def validate(N=200, L=1.0, KB=1.0, rho1=1.0, n_modes=4, verbose=True):
    """Recover clamped-free beam frequencies from the discrete operator.

    For KB X'''' = rho1 omega^2 X with a clamped root and a free tip, the exact
    frequencies are omega_n = (beta_n L)^2 sqrt(KB / (rho1 L^4)). Matching them
    is a direct test of both the fourth-derivative stencil and the four boundary
    conditions folded into it: get any one of them wrong and the ratios move by
    percent, not by 1/N^2.
    """
    ds = L / N
    D4 = bending_matrix(N, ds, root="clamped")
    ev = np.linalg.eigvals(D4)
    ev = np.sort(ev.real[np.abs(ev.imag) < 1e-8 * np.abs(ev).max()])
    ev = ev[ev > 0]
    omega = np.sqrt(KB / rho1 * ev[:n_modes])

    bl = cantilever_beta_L(n_modes)
    exact = bl ** 2 * np.sqrt(KB / (rho1 * L ** 4))
    rel = (omega - exact) / exact

    if verbose:
        print(f"clamped-free beam, N = {N}, ds = {ds:.4g}")
        print(f"{'mode':>5} {'beta_n L':>10} {'exact':>14} {'discrete':>14} "
              f"{'rel err':>11}")
        for i in range(n_modes):
            print(f"{i+1:>5} {bl[i]:>10.4f} {exact[i]:>14.6f} "
                  f"{omega[i]:>14.6f} {rel[i]:>+11.2e}")
    return omega, exact, rel


def mode_shape(s, L, bl):
    """Clamped-free beam eigenfunction, normalised to unit tip amplitude."""
    b = bl / L
    sig = ((np.cosh(bl) + np.cos(bl)) / (np.sinh(bl) + np.sin(bl)))
    phi = (np.cosh(b * s) - np.cos(b * s)
           - sig * (np.sinh(b * s) - np.sin(b * s)))
    return phi / np.abs(phi).max()


def validate_dynamic(N=40, L=1.0, KB=1e-3, rho1=1.0, amp=1e-4,
                     dt=0.02, n_periods=12, verbose=True):
    """Free vibration in vacuum: does the time-stepper reproduce omega_1?

    The static check above tests the spatial operator. This tests the operator
    driven through the tension solve and the implicit update, which is where a
    sign error or a mis-set boundary condition would show up as a frequency
    offset rather than a blow-up. Period is measured with the same Fourier
    estimator the paper uses everywhere else, so the number is comparable to the
    delta measurements rather than merely adjacent to them.

    The second output is the one that validates the constraint: with tension
    solved from the discrete length condition, |d_j| should stay at ds to a
    tolerance set by the explicit tension lag, not by the mesh.
    """
    from period import period_from_signal

    bl = cantilever_beta_L(1)[0]
    omega1 = bl ** 2 * np.sqrt(KB / (rho1 * L ** 4))
    T_exact = 2 * np.pi / omega1

    fil = Filament(N=N, L=L, rho1=rho1, KB=KB, dt=dt)
    s = np.arange(N + 1) * fil.ds
    v = np.zeros((N + 1, 2))
    v[:, 1] = amp * omega1 * mode_shape(s, L, bl)
    # seed the leapfrog with a backward Euler half-step so the initial state is
    # a velocity, not a displacement: X^{-1} = X^0 - dt * v
    fil.Xprev = fil.X - dt * v

    steps = int(n_periods * T_exact / dt)
    ts, tip, stretch = [], [], 0.0
    for n in range(steps):
        fil.step()
        ts.append(n * dt)
        tip.append(fil.tip()[1])
        stretch = max(stretch, fil.max_stretch())

    ts, tip = np.array(ts), np.array(tip)
    m = ts > 0.2 * ts.max()                    # drop the startup transient
    T_meas = period_from_signal(ts[m], tip[m])
    rel = (T_meas - T_exact) / T_exact

    if verbose:
        print(f"free vibration: N={N} dt={dt} KB={KB:g} amp={amp:g}")
        print(f"  omega_1 exact      {omega1:.8f}")
        print(f"  T exact            {T_exact:.6f}")
        print(f"  T measured         {T_meas:.6f}")
        print(f"  relative error     {rel:+.3e}")
        print(f"  max stretch        {stretch:.3e}")
        print(f"  tip amplitude      {np.abs(tip[m]).max():.3e}")
    return T_meas, T_exact, rel, stretch


def convergence(Ns=(50, 100, 200, 400), **kw):
    """First-mode error against resolution. Central differences give O(ds^2)."""
    print(f"{'N':>6} {'rel err (mode 1)':>18} {'ratio':>8}")
    prev = None
    for N in Ns:
        _, _, rel = validate(N=N, n_modes=1, verbose=False, **kw)
        e = abs(rel[0])
        print(f"{N:>6} {e:>18.3e} {'' if prev is None else f'{prev/e:>8.2f}'}")
        prev = e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--modes", type=int, default=4)
    ap.add_argument("--convergence", action="store_true")
    ap.add_argument("--dynamic", action="store_true")
    a = ap.parse_args()
    validate(N=a.N, n_modes=a.modes)
    if a.convergence:
        print()
        convergence()
    if a.dynamic:
        print()
        validate_dynamic()


if __name__ == "__main__":
    main()
