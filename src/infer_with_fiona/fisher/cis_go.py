"""
CIS lens geometric optics (GO) and beyond-GO (bGO) amplification factor
and its parameter derivatives.

Reference: Tambalo+2023, Eq. 1907 (general bGO derivative) and
Eqs. 1922-1923 (logM derivative).  compute_F_and_derivatives blends F and its
derivatives from FIONA into these above the measured switch frequency w_s(x_c, y).
"""
import numpy as np
from scipy.optimize import brentq


# ─── Basic CIS quantities ───────────────────────────────────────────────────

def _s(x, xc):
    return np.sqrt(x * x + xc * xc)


def _psi(x, xc):
    """CIS lensing potential ψ(x) with psi0=1."""
    s = _s(x, xc)
    return s + xc * np.log(2.0 * xc / (s + xc))


def _alpha(x, xc):
    """Deflection angle α(x) = dψ/dx = x/(s+xc)."""
    return x / (_s(x, xc) + xc)


def _dalpha_dx(x, xc):
    """dα/dx = xc / (s*(s+xc))."""
    s = _s(x, xc)
    return xc / (s * (s + xc))


def _psi_n(x, xc, n, dx=1e-5):
    """n-th derivative of ψ via central differences (n=3 or 4)."""
    if n == 3:
        return (_dalpha_dx(x + dx, xc) - _dalpha_dx(x - dx, xc)) / (2.0 * dx)
    if n == 4:
        return (_dalpha_dx(x + dx, xc) - 2.0 * _dalpha_dx(x, xc)
                + _dalpha_dx(x - dx, xc)) / dx ** 2
    raise ValueError(f"n must be 3 or 4, got {n}")


# ─── Image finding ─────────────────────────────────────────────────────────

def _lens_eq(x, y, xc):
    s = _s(x, xc)
    return x * (s + xc - 1.0) / (s + xc) - y


def cis_images(y, xc, x_max=20.0, n_grid=1000):
    """
    Find all image positions x_I for a CIS lens.

    Searches for sign changes of the lens equation on a grid and refines
    each with Brent's method.  Returns a list sorted by x_I.
    """
    x_grid = np.concatenate([
        np.linspace(-x_max, -1e-6, n_grid // 2),
        np.linspace(1e-6, x_max, n_grid // 2),
    ])
    f_grid = _lens_eq(x_grid, y, xc)

    images = []
    for i in range(len(x_grid) - 1):
        if f_grid[i] * f_grid[i + 1] < 0:
            try:
                root = brentq(
                    _lens_eq, x_grid[i], x_grid[i + 1],
                    args=(y, xc), xtol=1e-12, rtol=1e-12,
                )
                images.append(root)
            except Exception:
                pass
    return sorted(images)


def _find_image_near(y, xc, x_ref, tol=1e-10):
    """Find the image closest to x_ref for given (y, xc)."""
    bracket = max(abs(x_ref) * 0.05 + 1e-4, 0.01)
    for factor in (1.0, 2.0, 5.0, 10.0, 20.0):
        lo = x_ref - bracket * factor
        hi = x_ref + bracket * factor
        # Avoid x=0 crossing
        if lo * hi <= 0:
            lo = x_ref * 0.5 if x_ref > 0 else x_ref * 2.0
            hi = x_ref * 2.0 if x_ref > 0 else x_ref * 0.5
            if lo > hi:
                lo, hi = hi, lo
        try:
            fa = _lens_eq(lo, y, xc)
            fb = _lens_eq(hi, y, xc)
            if fa * fb < 0:
                return brentq(_lens_eq, lo, hi, args=(y, xc),
                               xtol=tol, rtol=tol)
        except Exception:
            pass
    # Fall back to full search
    imgs = cis_images(y, xc)
    if imgs:
        return min(imgs, key=lambda x: abs(x - x_ref))
    return x_ref


# ─── Image properties ──────────────────────────────────────────────────────

def cis_image_props(x_I, y, xc):
    """
    GO properties of image at x_I.

    Returns
    -------
    phi_I : float  Fermat potential
    mu_I  : float  Signed magnification
    n_I   : float  Morse index (0, 0.5, or 1)
    delta_I : float  bGO correction Δ_I
    """
    s = _s(x_I, xc)

    # Fermat potential
    phi_I = 0.5 * (x_I - y) ** 2 - _psi(x_I, xc)

    # f1 = 1 - α/x = (s+xc-1)/(s+xc);  f2 = 1 - dα/dx = (s(s+xc)-xc)/(s(s+xc))
    f1 = (s + xc - 1.0) / (s + xc)
    f2 = (s * (s + xc) - xc) / (s * (s + xc))
    inv_mu = f1 * f2
    mu_I = 1.0 / inv_mu if abs(inv_mu) > 1e-14 else np.sign(inv_mu) * 1e14

    if f1 > 0 and f2 > 0:
        n_I = 0.0    # minimum
    elif f1 < 0 and f2 < 0:
        n_I = 1.0    # maximum
    else:
        n_I = 0.5    # saddle

    # bGO correction
    a = f2 / 2.0  # (1 - ψ'')/2
    b = f1 / 2.0  # (1 - ψ'/x)/2
    psi3 = _psi_n(x_I, xc, 3)
    psi4 = _psi_n(x_I, xc, 4)

    delta_I = (1.0 / 16.0) * (
        psi4 / (2.0 * a ** 2)
        + 5.0 * psi3 ** 2 / (12.0 * a ** 3)
        + psi3 / (a ** 2 * x_I)
        + (a - b) / (a * b * x_I ** 2)
    )

    return phi_I, mu_I, n_I, delta_I


# ─── bGO amplification factor ──────────────────────────────────────────────

def cis_F_bGO(w_vec, y, xc):
    """
    bGO amplification factor F(w) for an array of w values.

    F(w) = Σ_I |μ_I|^{1/2} (1 + i Δ_I / w) exp(iw φ_I − iπ n_I)
    """
    w_vec = np.asarray(w_vec, dtype=float)
    images = cis_images(y, xc)
    F = np.zeros(len(w_vec), dtype=complex)
    for x_I in images:
        phi_I, mu_I, n_I, delta_I = cis_image_props(x_I, y, xc)
        amp = np.sqrt(abs(mu_I))
        F += amp * (1.0 + 1j * delta_I / w_vec) * np.exp(
            1j * w_vec * phi_I - 1j * np.pi * n_I
        )
    return F


# ─── bGO derivatives ───────────────────────────────────────────────────────

def cis_dF_bGO(w_vec, y, xc, eps_y=1e-5, eps_xc=None):
    """
    Derivatives of F_bGO w.r.t. {log w (≡ log M_Lz), y, xc}.

    Uses Tambalo+2023 Eq. 1907:
        ∂F/∂θ = Σ_I [iw ψ_{I,θ} + μ_{I,θ}/(2μ_I) + i/w Δ_{I,θ}] F_I(w)

    For log w (Eq. 1922–1923):
        ∂F/∂(log w) = Σ_I [iw φ_I − i/w Δ_I] F_I(w)

    ψ_{I,y}  = −α(x_I)  (analytic)
    ψ_{I,xc} = ∂φ_I/∂xc  (finite difference; x_I fixed at stationary point so
                            the stationarity condition kills the ∂x_I/∂xc term)
    μ_{I,θ} and Δ_{I,θ} are obtained via finite differences in θ.

    Parameters
    ----------
    w_vec : array_like, shape (n_w,)
    y, xc : float
    eps_y, eps_xc : float  Central-difference steps.

    Returns
    -------
    dF_dlogw, dF_dy, dF_dxc : ndarrays, complex, shape (n_w,)
    """
    if eps_xc is None:
        eps_xc = max(1e-7, 1e-4 * float(xc))

    w_vec = np.asarray(w_vec, dtype=float)
    images = cis_images(y, xc)

    dF_dlogw = np.zeros(len(w_vec), dtype=complex)
    dF_dy    = np.zeros(len(w_vec), dtype=complex)
    dF_dxc   = np.zeros(len(w_vec), dtype=complex)

    for x_I in images:
        phi_I, mu_I, n_I, delta_I = cis_image_props(x_I, y, xc)
        amp = np.sqrt(abs(mu_I))
        phase = np.exp(1j * w_vec * phi_I - 1j * np.pi * n_I)
        F_I_GO  = amp * phase
        F_I_bGO = amp * (1.0 + 1j * delta_I / w_vec) * phase

        # ── log w derivative (Eq. 1922) ──
        dF_dlogw += (1j * w_vec * phi_I - 1j * delta_I / w_vec) * F_I_bGO

        # ── y derivative ──
        psi_y = -_alpha(x_I, xc)   # ∂φ_I/∂y = −(x_I−y)·(−1) ... = −α(x_I)

        xI_yp = _find_image_near(y + eps_y, xc, x_I)
        xI_ym = _find_image_near(y - eps_y, xc, x_I)
        _, mu_yp, _, delta_yp = cis_image_props(xI_yp, y + eps_y, xc)
        _, mu_ym, _, delta_ym = cis_image_props(xI_ym, y - eps_y, xc)

        dmu_dy     = (mu_yp - mu_ym)     / (2.0 * eps_y)
        ddelta_dy  = (delta_yp - delta_ym) / (2.0 * eps_y)

        dF_dy += (
            1j * w_vec * psi_y
            + dmu_dy / (2.0 * mu_I)
            + 1j * ddelta_dy / w_vec
        ) * F_I_bGO

        # ── xc derivative ──
        xI_xcp = _find_image_near(y, xc + eps_xc, x_I)
        xI_xcm = _find_image_near(y, xc - eps_xc, x_I)
        phi_xcp, mu_xcp, _, delta_xcp = cis_image_props(xI_xcp, y, xc + eps_xc)
        phi_xcm, mu_xcm, _, delta_xcm = cis_image_props(xI_xcm, y, xc - eps_xc)

        # ψ_{I,xc} = ∂φ_I/∂xc|_{y,x_I on-shell} ≈ (φ_xcp − φ_xcm)/(2 eps_xc)
        psi_xc   = (phi_xcp - phi_xcm)    / (2.0 * eps_xc)
        dmu_dxc  = (mu_xcp  - mu_xcm)     / (2.0 * eps_xc)
        ddelta_dxc = (delta_xcp - delta_xcm) / (2.0 * eps_xc)

        dF_dxc += (
            1j * w_vec * psi_xc
            + dmu_dxc / (2.0 * mu_I)
            + 1j * ddelta_dxc / w_vec
        ) * F_I_bGO

    return dF_dlogw, dF_dy, dF_dxc
