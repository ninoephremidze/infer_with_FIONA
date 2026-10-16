"""
Generic (lens-agnostic) geometric-optics (GO) and beyond-GO (bGO) amplification
factor, for ANY fiona.lenses.AxisymmetricLens subclass that exposes psi_r(r).

Implements the same general bGO expansion cis_go.py already implements for CIS
specifically (Tambalo+2023's beyond-geometric-optics correction), but derives
alpha(x)=psi_r'(x) and its higher derivatives NUMERICALLY (5-point central
finite-difference stencils applied directly to lens.psi_r), rather than
hand-coded per-lens analytic formulas. This lets it work for any lens, including
ones without closed-form derivatives (e.g. ULASoliton, whose potential is a
spline-tabulated integral with no analytic derivative available at all).

IMPORTANT sign convention: lens.psi_r(r) expects r >= 0 (some implementations,
e.g. ULASoliton, are only correct for non-negative input). The 1D signed
lens-plane coordinate x used throughout GO/bGO theory can be negative (images on
the opposite side of the lens center), and the axisymmetric potential as a
function of signed x is psi(x) = psi_r(|x|). All lens evaluations in this module
therefore go through _psi(lens, x), which applies abs() before calling
lens.psi_r -- never call lens.psi_r(x) directly with a possibly-negative x.

Validated against cis_go.py's own analytic CIS implementation (see
validate_generic_go.py) before being trusted for other lens models.
"""
import numpy as np
from scipy.optimize import brentq


def _psi(lens, x):
    """psi(x) = psi_r(|x|) for the SIGNED 1D lens-plane coordinate x."""
    x = np.asarray(x, dtype=float)
    return lens.psi_r(np.abs(x))


def _step(x):
    """Finite-difference step size, scaled to the local coordinate magnitude."""
    return max(1e-3, 1e-3 * abs(x))


def _alpha(lens, x, h=None):
    """Deflection angle alpha(x) = d(psi)/dx, central difference (scalar or array x)."""
    x = np.asarray(x, dtype=float)
    if h is None:
        h = np.maximum(1e-3, 1e-3 * np.abs(x))
    return (_psi(lens, x + h) - _psi(lens, x - h)) / (2.0 * h)


def _derivatives_1234(lens, x):
    """
    Returns (alpha, dalpha_dx, psi3, psi4) = (psi', psi'', psi''', psi'''') at a
    single scalar x, from ONE 5-point central-difference stencil on psi(x) --
    all four derivatives come from the same 5 function evaluations, avoiding
    the noise amplification of repeatedly finite-differencing a finite
    difference. Standard stencils, O(h^4) for the first two, O(h^2) for the
    third/fourth (matching cis_go.py's own use of central differences for its
    3rd/4th-order bGO-correction terms).
    """
    x = float(x)
    h = _step(x)
    f_m2 = _psi(lens, x - 2 * h)
    f_m1 = _psi(lens, x - h)
    f_0 = _psi(lens, x)
    f_p1 = _psi(lens, x + h)
    f_p2 = _psi(lens, x + 2 * h)

    d1 = (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * h)
    d2 = (-f_p2 + 16 * f_p1 - 30 * f_0 + 16 * f_m1 - f_m2) / (12 * h ** 2)
    d3 = (f_p2 - 2 * f_p1 + 2 * f_m1 - f_m2) / (2 * h ** 3)
    d4 = (f_p2 - 4 * f_p1 + 6 * f_0 - 4 * f_m1 + f_m2) / h ** 4
    return float(d1), float(d2), float(d3), float(d4)


def _lens_eq(x, y, lens):
    return x - _alpha(lens, x) - y


def find_images(lens, y, x_max=20.0, n_grid=2000):
    """Find all image positions x_I for an arbitrary axisymmetric lens, via grid
    search for sign changes of the lens equation + Brent's method refinement
    (identical strategy to cis_go.py's cis_images)."""
    x_grid = np.concatenate([
        np.linspace(-x_max, -1e-6, n_grid // 2),
        np.linspace(1e-6, x_max, n_grid // 2),
    ])
    f_grid = _lens_eq(x_grid, y, lens)

    images = []
    for i in range(len(x_grid) - 1):
        if f_grid[i] * f_grid[i + 1] < 0:
            try:
                root = brentq(_lens_eq, x_grid[i], x_grid[i + 1], args=(y, lens),
                              xtol=1e-12, rtol=1e-12)
                images.append(root)
            except Exception:
                pass
    return sorted(images)


def image_properties(lens, x_I, y):
    """
    GO/bGO properties of the image at x_I, for an arbitrary axisymmetric lens.
    Same formulas as cis_go.py's cis_image_props, generalized: f1 = 1-alpha/x
    (tangential eigenvalue), f2 = 1-alpha' (radial eigenvalue), Morse index from
    their signs, bGO correction from psi'''/psi'''' (Tambalo+2023 general form).

    Returns
    -------
    phi_I : float    Fermat potential
    mu_I  : float    Signed magnification
    n_I   : float    Morse index (0, 0.5, or 1)
    delta_I : float  bGO correction Delta_I
    """
    alpha, dalpha_dx, psi3, psi4 = _derivatives_1234(lens, x_I)
    psi_val = float(_psi(lens, np.array([x_I]))[0])

    phi_I = 0.5 * (x_I - y) ** 2 - psi_val

    f1 = 1.0 - alpha / x_I
    f2 = 1.0 - dalpha_dx
    inv_mu = f1 * f2
    mu_I = 1.0 / inv_mu if abs(inv_mu) > 1e-14 else np.sign(inv_mu) * 1e14

    if f1 > 0 and f2 > 0:
        n_I = 0.0
    elif f1 < 0 and f2 < 0:
        n_I = 1.0
    else:
        n_I = 0.5

    a = f2 / 2.0
    b = f1 / 2.0
    delta_I = (1.0 / 16.0) * (
        psi4 / (2.0 * a ** 2)
        + 5.0 * psi3 ** 2 / (12.0 * a ** 3)
        + psi3 / (a ** 2 * x_I)
        + (a - b) / (a * b * x_I ** 2)
    )

    return phi_I, mu_I, n_I, delta_I


# Cored profiles (ULA soliton, small-x_c CIS) produce an extra image very close to
# x=0 (the standard faint "central" image of any singular-free lens). Its flux is
# always negligible (|mu_I| << 1), but its Delta_I bGO correction is large (checked:
# psi3/psi4 there are genuine, well-converged features of the potential's curvature
# near x=0, NOT a finite-difference artifact) -- consistent with the bGO leading-order
# expansion converging more slowly for near-critical/highly-demagnified images, a
# known caveat of the approximation, not a bug in this module's numerics. Given its
# already-negligible flux and uncertain convergence at the w where this module is
# used, dropping it is the conservative choice. 1e-2 comfortably exceeds every
# central-image |mu_I| seen in testing (max ~4e-3 across a wide (y, x_c/r_s) sweep)
# while never approaching the O(0.1-few) magnifications of genuine images. NOTE: this
# is NOT the primary explanation for the wide-prior parameter bias seen across
# CIS/SIDM/ULA/PBH mass-scan runs -- PBH has no such image at all (exact, core-free
# point lens) and shows comparable-order bias, so that bias has a different, still
# unidentified cause (see notes accompanying the mass-scan results).
MU_MIN_BGO = 1.0e-2


def F_bGO(lens, y, w_vec):
    """
    bGO amplification factor F(w) for an array of w values, for an arbitrary
    axisymmetric lens: F(w) = Sum_I |mu_I|^(1/2) (1 + i Delta_I/w) exp(i w
    phi_I - i pi n_I). Same formula as cis_go.py's cis_F_bGO, generalized.
    Images with negligible magnification (|mu_I| < MU_MIN_BGO) are dropped -- see
    MU_MIN_BGO's docstring above.
    """
    w_vec = np.asarray(w_vec, dtype=float)
    images = find_images(lens, y)
    F = np.zeros(len(w_vec), dtype=complex)
    for x_I in images:
        phi_I, mu_I, n_I, delta_I = image_properties(lens, x_I, y)
        if abs(mu_I) < MU_MIN_BGO:
            continue
        amp = np.sqrt(abs(mu_I))
        F += amp * (1.0 + 1j * delta_I / w_vec) * np.exp(
            1j * w_vec * phi_I - 1j * np.pi * n_I
        )
    return F
