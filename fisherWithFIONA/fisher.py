"""
Fisher matrix assembly and marginalization for GW lensing forecasts.

Parameter ordering throughout:  [log M_Lz, y, x_c, log D_S, phi_0]
Indices:                          [0,        1, 2,   3,       4    ]
"""
import warnings
import numpy as np


def _inner_product(dh_i, dh_j, S_n, f_vec):
    """
    Noise-weighted inner product contribution to Fisher matrix element.

    F_ij = 4 Re ∫ df  (∂_i h_L)* (∂_j h_L) / S_n(f)
    """
    integrand = np.real(np.conj(dh_i) * dh_j / S_n)
    return 4.0 * np.trapz(integrand, x=f_vec)


def assemble_fisher_matrix(h_L, dh_logM, dh_y, dh_xc, S_n, f_vec):
    """
    Build the 5×5 Fisher matrix for parameters {log M_Lz, y, x_c, log D_S, φ_0}.

    Derivatives w.r.t. log D_S and φ_0 are computed analytically from h_L:
        ∂ h_L / ∂ log D_S = −h_L   (since h_L ∝ 1/D_L)
        ∂ h_L / ∂ φ_0     = i h_L

    Parameters
    ----------
    h_L : ndarray, complex, shape (n_f,)
        Lensed waveform.
    dh_logM, dh_y, dh_xc : ndarray, complex, shape (n_f,)
        ∂ h_L / ∂(log M_Lz), ∂ h_L / ∂y, ∂ h_L / ∂x_c.
    S_n : ndarray, shape (n_f,)
        Noise PSD [Hz^{-1}].
    f_vec : ndarray, shape (n_f,)
        Frequency array [Hz].

    Returns
    -------
    F_mat : ndarray, shape (5, 5)
        Fisher matrix.
    SNR : float
        Signal-to-noise ratio.
    """
    dh_logD = -h_L
    dh_phi0 = 1j * h_L

    derivs = [dh_logM, dh_y, dh_xc, dh_logD, dh_phi0]

    F_mat = np.zeros((5, 5))
    for i in range(5):
        for j in range(i, 5):
            val = _inner_product(derivs[i], derivs[j], S_n, f_vec)
            F_mat[i, j] = val
            F_mat[j, i] = val

    # SNR^2 = (h_L | h_L)
    SNR = np.sqrt(max(_inner_product(h_L, h_L, S_n, f_vec), 0.0))
    return F_mat, SNR


def _invert_fisher(F_mat):
    """Invert F_mat, falling back to pseudo-inverse if truly singular.

    Condition number context
    -----------------------
    The 5×5 Fisher matrix always has two dominant eigenvalues ≈ SNR²
    (from the log D_S and φ₀ directions).  Lens-parameter eigenvalues
    are smaller, so κ ≈ SNR²/λ_min is naturally 10³–10⁵ even when the
    constraints are tight.  In the geometric-optics limit (large M_Lz),
    changing log M_Lz shifts interference fringes in frequency, mimicking
    a global phase shift φ₀.  This logM–φ₀ near-degeneracy drives κ
    upward with M_Lz.  κ up to ~10⁸ is expected and the inversion
    remains numerically reliable (relative error ≈ κ × ε_machine < 10⁻⁸).
    Only beyond ~10¹⁰ does the inversion become genuinely unreliable.
    """
    cond = np.linalg.cond(F_mat)
    if cond > 1e10:
        warnings.warn(
            f"Fisher matrix ill-conditioned (cond={cond:.2e}); "
            "using pseudo-inverse. Sigma values unreliable.",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.linalg.pinv(F_mat), cond
    if cond > 1e8:
        warnings.warn(
            f"Fisher matrix moderately ill-conditioned (cond={cond:.2e}); "
            "results may carry ~1% numerical error.",
            RuntimeWarning,
            stacklevel=3,
        )
    try:
        return np.linalg.inv(F_mat), cond
    except np.linalg.LinAlgError:
        warnings.warn("Fisher matrix singular; using pseudo-inverse.",
                      RuntimeWarning, stacklevel=3)
        return np.linalg.pinv(F_mat), cond


def marginalized_errors(F_mat):
    """
    Invert the 5×5 Fisher matrix and extract 1-σ errors on lens parameters.

    Marginalises over log D_S and φ_0 automatically via full inversion.

    Parameters
    ----------
    F_mat : ndarray, shape (5, 5)
        Fisher matrix (must be positive definite).

    Returns
    -------
    sigma_logM : float
    sigma_y : float
    sigma_xc : float
    cond : float
        Condition number of the Fisher matrix.
    """
    F_inv, cond = _invert_fisher(F_mat)
    sigma_logM = float(np.sqrt(abs(F_inv[0, 0])))
    sigma_y    = float(np.sqrt(abs(F_inv[1, 1])))
    sigma_xc   = float(np.sqrt(abs(F_inv[2, 2])))
    return sigma_logM, sigma_y, sigma_xc, cond


def marginalised_covariance(F_mat):
    """
    Return the 3×3 marginalised covariance matrix for {log M_Lz, y, x_c}.

    Computed as the top-left 3×3 block of the full 5×5 inverse Fisher matrix,
    which automatically marginalises over log D_S and φ_0.

    Parameters
    ----------
    F_mat : ndarray, shape (5, 5)

    Returns
    -------
    cov : ndarray, shape (3, 3)
    cond : float
    """
    F_inv, cond = _invert_fisher(F_mat)
    return F_inv[:3, :3].copy(), cond


def lsa_validity(M_Lz_Msun, f_star_Hz, phi_H, sigma_logM):
    """
    Check whether the Linearised Signal Approximation (LSA) is valid.

    Criterion (Tambalo+2023 App. B):
        Δ_LSA = 8π (G/c³) M_Lz f_* φ_H σ(log M_Lz) ≲ 1

    Parameters
    ----------
    M_Lz_Msun : float
    f_star_Hz : float
        Frequency enclosing 90% of the SNR [Hz].
    phi_H : float
        Central-image time delay (Fermat potential) [dimensionless].
    sigma_logM : float
        1-σ uncertainty on log M_Lz from Fisher analysis.

    Returns
    -------
    delta_lsa : float
        Δ_LSA value (should be < 1 for LSA to hold).
    valid : bool
    """
    from .utils import G_over_c3_Msun
    delta = 8.0 * np.pi * G_over_c3_Msun * float(M_Lz_Msun) * float(f_star_Hz) * float(phi_H) * float(sigma_logM)
    return float(delta), delta < 1.0
