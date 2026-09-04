"""
Immersed-boundary coupling between the D2Q9 lattice Boltzmann flow and a
Lagrangian body.

The wake work uses halfway bounce-back on a static mask, which is accurate and
cheap but only represents bodies that sit still on lattice nodes. A flapping
filament is neither: it is a curve that moves through the lattice and does not
align with it. The immersed-boundary method handles that by keeping the fluid
grid fixed and carrying the body as a set of Lagrangian markers, exchanging
velocity and force through a smoothed delta function.

Two things here matter for what this study measures.

First, no-slip is enforced by multi-direct forcing rather than by a penalty law.
The alternative in the flapping-filament literature (Huang, Shin & Sung 2007)
applies a feedback force with two large stiffness constants. Those constants
introduce a fast artificial timescale, and a fast artificial timescale in a study
whose entire subject is the period of a limit cycle is a bad trade. Direct
forcing iterated two or three times drives the interfacial slip down without
adding any timescale that was not already there.

Second, the force returned to the structure is the negated spread force, summed
with the same weights used to interpolate. Using matched interpolation and
spreading operators is what makes the exchange conservative: whatever momentum
is added to the fluid is removed from the body. The drag retraction in the wake
work came from a force estimate that was not conservative in this sense, and the
lesson is applied here rather than repeated.
"""
import numpy as np


# ------------------------------------------------------------ delta function
def peskin4(r):
    """Peskin's 4-point regularised delta, support |r| < 2.

    Chosen over the 3-point kernel because its first moment is exact and it is
    continuously differentiable, which keeps the force from ringing as markers
    cross lattice lines -- a spurious oscillation at the marker-crossing
    frequency would land directly on top of the quantity being measured.
    """
    a = np.abs(r)
    out = np.zeros_like(a)
    m1 = a <= 1.0
    m2 = (a > 1.0) & (a <= 2.0)
    r1 = a[m1]
    out[m1] = (3 - 2 * r1 + np.sqrt(np.maximum(1 + 4 * r1 - 4 * r1 ** 2, 0))) / 8
    r2 = a[m2]
    out[m2] = (5 - 2 * r2 - np.sqrt(np.maximum(-7 + 12 * r2 - 4 * r2 ** 2, 0))) / 8
    return out


class Stencil:
    """Precomputed 4x4 lattice support and weights for each Lagrangian marker.

    Rebuilt every step because the markers move. Kept as explicit index arrays
    so that interpolation and spreading are the same gather/scatter with the
    same weights, which is what makes the pair adjoint.
    """

    def __init__(self, X, nx, ny):
        self.n = len(X)
        base = np.floor(X).astype(int) - 1               # [n, 2]
        off = np.arange(4)
        ix = (base[:, 0:1] + off[None, :])               # [n, 4]
        iy = (base[:, 1:2] + off[None, :])
        wx = peskin4(X[:, 0:1] - ix)
        wy = peskin4(X[:, 1:2] - iy)
        self.w = wx[:, :, None] * wy[:, None, :]         # [n, 4, 4]
        self.ix = np.clip(ix, 0, nx - 1)
        self.iy = np.clip(iy, 0, ny - 1)
        # markers whose stencil left the domain contribute nothing
        inside = ((ix >= 0) & (ix < nx))[:, :, None] & \
                 ((iy >= 0) & (iy < ny))[:, None, :]
        self.w = self.w * inside

    def interpolate(self, field):
        """Eulerian field [nx, ny] -> value at each marker. h = 1."""
        vals = field[self.ix[:, :, None], self.iy[:, None, :]]   # [n, 4, 4]
        return np.einsum("nij,nij->n", self.w, vals)

    def spread(self, q, nx, ny, ds):
        """Marker quantity q [n] -> Eulerian field [nx, ny], weighted by ds."""
        out = np.zeros((nx, ny))
        contrib = self.w * (q * ds)[:, None, None]
        np.add.at(out, (self.ix[:, :, None].repeat(4, 2),
                        self.iy[:, None, :].repeat(4, 1)), contrib)
        return out


# ------------------------------------------------------------------- forcing
def guo_source(ux, uy, gx, gy, c, w, tau):
    """Guo (2002) forcing term for the BGK collision.

        S_i = (1 - 1/(2 tau)) w_i [ (c_i - u)/cs^2 + (c_i.u) c_i / cs^4 ] . g

    with cs^2 = 1/3. Guo's form is used rather than a naive shift because it is
    the one that recovers the Navier-Stokes forcing without a spurious term at
    second order -- the discrete lattice otherwise leaves an error proportional
    to the force gradient, which for a moving boundary is exactly where the
    force is largest.
    """
    pref = (1.0 - 0.5 / tau)
    cx = c[:, 0][:, None, None]
    cy = c[:, 1][:, None, None]
    cu = cx * ux + cy * uy
    sx = 3.0 * (cx - ux) + 9.0 * cu * cx
    sy = 3.0 * (cy - uy) + 9.0 * cu * cy
    return pref * w[:, None, None] * (sx * gx + sy * gy)


def multi_direct_force(interp_u, u_target, rho=1.0, dt=1.0):
    """Direct-forcing Lagrangian force density for one correction pass.

    F = 2 rho (u_target - u_interpolated) / dt. The factor 2 is the Guo
    half-step: the velocity the collision sees is (sum c f + g/2)/rho, so a
    force of 2 rho du/dt is what moves the observed velocity by du in one step.
    """
    return 2.0 * rho * (u_target - interp_u) / dt
