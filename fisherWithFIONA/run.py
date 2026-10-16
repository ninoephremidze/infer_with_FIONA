"""
Top-level Fisher forecast runner for FisherWithFIONA.
"""
import time
import warnings
import numpy as np

from .utils import luminosity_distance, c200_to_xs
from .noise import lisa_psd, ligo_psd
from .waveform import imrphenomd_amplitude, f_isco, f_1month
from .lensing import compute_F_and_derivatives, compute_F_and_derivatives_3p, _DEFAULT_GL_DIR
from .fisher import assemble_fisher_matrix, marginalized_errors, marginalised_covariance

_LISA_F_MIN = 1e-4   # Hz — hard lower limit of LISA band
_LISA_F_MAX = 1.0    # Hz — hard upper limit of LISA band


def run_fisher(
    M_Lz_Msun,
    y,
    x_c,
    m1_Msun,
    m2_Msun,
    z_s=3.0,
    lens_model='CIS',
    n_f=512,
    eps_logM=5e-6,
    eps_y=5e-6,
    eps_xc=None,
    gl_dir=None,
    n_workers=1,
    return_cov=False,
    waveform_func=None,
    psd_func=None,
    f_min=None,
    f_max=None,
    w_high=100.0,
):
    """
    Run Fisher forecasts for GW lensing by a CIS lens observed with LISA.

    Replicates the Tambalo+2023 methodology, replacing their contour-integral
    F(w) evaluation with FIONA's Fresnel solver.  Fisher constraints are
    computed for all (M_Lz, y) pairs simultaneously.

    Parameters
    ----------
    M_Lz_Msun : float or array_like
        Effective lens mass(es) [M_sun].
    y : float or array_like
        Source impact parameter(s) (dimensionless).  All y-values are
        evaluated in a single FIONA call per M_Lz value.
    x_c : float
        CIS core radius (dimensionless).
    m1_Msun, m2_Msun : float
        Component masses of the GW source [M_sun].
    z_s : float
        Source redshift (default 3.0).
    lens_model : str
        Lens model identifier.  Currently only 'CIS' is supported.
    n_f : int
        Number of frequency points (log-spaced from f_1month to f_ISCO).
    eps_logM : float
        Finite-difference step in log M_Lz (default 0.01).
    eps_y : float
        Finite-difference step in y (default 0.01, additive).
    eps_xc : float or None
        Finite-difference step in x_c (default max(0.001, 0.1 * x_c)).
    gl_dir : str or None
        Path to FIONA Gauss-Legendre node cache.
    n_workers : int
        Worker processes for FIONA parallelism (default 1).
    return_cov : bool
        If True, include the 3×3 marginalised covariance matrix for each
        (M_Lz, y) pair in the returned dict under key 'cov',
        shape (n_M, n_y, 3, 3).  Parameters: [log M_Lz, y, x_c].
    waveform_func : callable or None
        If provided, called as waveform_func(f_vec) and must return |h_0(f)|
        as a real ndarray with the same shape as f_vec.  When None (default)
        the built-in SPA inspiral amplitude is used.
    w_high : float
        bGO threshold: for CIS lens, derivatives at w > w_high are computed
        with the analytic bGO formula instead of FIONA finite differences.
        Default 100.0 (per Tambalo+2023).  Set to np.inf to disable.

    Returns
    -------
    dict with keys:
        'M_Lz'           : ndarray, shape (n_M,)   — lens masses [M_sun]
        'y'              : ndarray, shape (n_y,)   — impact parameters
        'sigma_logM'     : ndarray, shape (n_M, n_y)
        'sigma_y'        : ndarray, shape (n_M, n_y)
        'sigma_xc'       : ndarray, shape (n_M, n_y)
        'SNR'            : ndarray, shape (n_M, n_y)
        'condition_number': ndarray, shape (n_M, n_y)
        'cov'            : ndarray, shape (n_M, n_y, 3, 3)  [if return_cov]
    """
    gl_dir = gl_dir or _DEFAULT_GL_DIR

    M_Lz_arr = np.atleast_1d(np.asarray(M_Lz_Msun, dtype=float))
    y_arr = np.atleast_1d(np.asarray(y, dtype=float))
    n_M = len(M_Lz_arr)
    n_y = len(y_arr)

    M_total = m1_Msun + m2_Msun
    eta = (m1_Msun * m2_Msun) / M_total**2

    D_L_Mpc = luminosity_distance(z_s)

    # Waveform frequency window (source-dependent, M_Lz-independent)
    _f_min = f_min if f_min is not None else _LISA_F_MIN
    _f_max = f_max if f_max is not None else _LISA_F_MAX
    f_hi = min(f_isco(M_total, z_s), _f_max)
    f_lo = max(f_1month(M_total, eta, z_s), _f_min)
    if f_lo >= f_hi:
        raise ValueError(
            f"No valid frequency band: f_lo={f_lo:.3e} Hz >= f_hi={f_hi:.3e} Hz. "
            "Try lighter masses or shorter observation time."
        )
    f_vec = np.logspace(np.log10(f_lo), np.log10(f_hi), n_f)

    # Noise PSD and unlensed waveform (same for all M_Lz)
    _psd = psd_func if psd_func is not None else lisa_psd
    S_n = _psd(f_vec)
    if waveform_func is not None:
        h0 = waveform_func(f_vec)
    else:
        h0 = imrphenomd_amplitude(f_vec, M_total, eta, z_s, D_L_Mpc)

    # Output arrays
    sigma_logM = np.full((n_M, n_y), np.nan)
    sigma_y_out = np.full((n_M, n_y), np.nan)
    sigma_xc = np.full((n_M, n_y), np.nan)
    SNR_out = np.full((n_M, n_y), np.nan)
    cond_out = np.full((n_M, n_y), np.nan)
    if return_cov:
        cov_out = np.full((n_M, n_y, 3, 3), np.nan)

    t_start_total = time.perf_counter()
    for i_M, M_Lz in enumerate(M_Lz_arr):
        t0 = time.perf_counter()
        # ---- Four FIONA calls: base (with batched y±ε), M_Lz±, xc± ---- #
        F, dF_dlogM, dF_dy, dF_dxc = compute_F_and_derivatives(
            M_Lz_Msun=M_Lz,
            y_vec=y_arr,
            x_c=x_c,
            f_vec=f_vec,
            lens_model=lens_model,
            gl_dir=gl_dir,
            n_workers=n_workers,
            eps_logM=eps_logM,
            eps_y=eps_y,
            eps_xc=eps_xc,
            w_high=w_high,
        )
        # F, dF_* all have shape (n_f, n_y)

        for i_y in range(n_y):
            h_L = F[:, i_y] * h0

            dh_logM = dF_dlogM[:, i_y] * h0
            dh_y = dF_dy[:, i_y] * h0
            dh_xc = dF_dxc[:, i_y] * h0

            F_mat, snr = assemble_fisher_matrix(
                h_L, dh_logM, dh_y, dh_xc, S_n, f_vec
            )
            sig_logM, sig_y, sig_xc, cond = marginalized_errors(F_mat)

            sigma_logM[i_M, i_y] = sig_logM
            sigma_y_out[i_M, i_y] = sig_y
            sigma_xc[i_M, i_y] = sig_xc
            SNR_out[i_M, i_y] = snr
            cond_out[i_M, i_y] = cond

            if return_cov:
                cov_3x3, _ = marginalised_covariance(F_mat)
                cov_out[i_M, i_y] = cov_3x3

        dt = time.perf_counter() - t0
        elapsed = time.perf_counter() - t_start_total
        remaining = dt * (n_M - i_M - 1)
        print(
            f"  M_Lz={M_Lz:.2e} M_sun  [{i_M+1}/{n_M}]  "
            f"{dt:.1f}s/iter  elapsed={elapsed:.0f}s  "
            f"ETA={remaining:.0f}s",
            flush=True,
        )

    out = {
        'M_Lz': M_Lz_arr,
        'y': y_arr,
        'sigma_logM': sigma_logM,
        'sigma_y': sigma_y_out,
        'sigma_xc': sigma_xc,
        'SNR': SNR_out,
        'condition_number': cond_out,
    }
    if return_cov:
        out['cov'] = cov_out
    return out


# ─────────────────────────────────────────────────────────────────────────── #
# Generic 3-parameter Fisher runner (gSIS, NFW, etc.)                        #
# ─────────────────────────────────────────────────────────────────────────── #

def _run_fisher_3p(
    M_Lz_Msun, y, param3, m1_Msun, m2_Msun,
    lens_model, sigma3_key,
    z_s=3.0, n_f=512,
    eps_logM=5e-6, eps_y=5e-6, eps_p3=None,
    gl_dir=None, n_workers=1, return_cov=False,
    waveform_func=None,
):
    """Generic Fisher runner for lens models with a third shape parameter."""
    gl_dir = gl_dir or _DEFAULT_GL_DIR

    M_Lz_arr = np.atleast_1d(np.asarray(M_Lz_Msun, dtype=float))
    y_arr    = np.atleast_1d(np.asarray(y, dtype=float))
    n_M      = len(M_Lz_arr)
    n_y      = len(y_arr)

    M_total = m1_Msun + m2_Msun
    eta     = (m1_Msun * m2_Msun) / M_total ** 2
    D_L_Mpc = luminosity_distance(z_s)

    f_hi = min(f_isco(M_total, z_s), _LISA_F_MAX)
    f_lo = max(f_1month(M_total, eta, z_s), _LISA_F_MIN)
    if f_lo >= f_hi:
        raise ValueError(
            f"No valid frequency band: f_1month={f_lo:.3e} Hz >= f_ISCO={f_hi:.3e} Hz. "
            "Try lighter masses or shorter observation time."
        )
    f_vec = np.logspace(np.log10(f_lo), np.log10(f_hi), n_f)

    S_n = lisa_psd(f_vec)
    if waveform_func is not None:
        h0 = waveform_func(f_vec)
    else:
        h0  = imrphenomd_amplitude(f_vec, M_total, eta, z_s, D_L_Mpc)

    sigma_logM = np.full((n_M, n_y), np.nan)
    sigma_y_out = np.full((n_M, n_y), np.nan)
    sigma_p3   = np.full((n_M, n_y), np.nan)
    SNR_out    = np.full((n_M, n_y), np.nan)
    cond_out   = np.full((n_M, n_y), np.nan)
    if return_cov:
        cov_out = np.full((n_M, n_y, 3, 3), np.nan)

    t_start_total = time.perf_counter()
    for i_M, M_Lz in enumerate(M_Lz_arr):
        t0 = time.perf_counter()
        F, dF_dlogM, dF_dy, dF_dp3 = compute_F_and_derivatives_3p(
            M_Lz_Msun=M_Lz,
            y_vec=y_arr,
            param3=param3,
            f_vec=f_vec,
            lens_model=lens_model,
            gl_dir=gl_dir,
            n_workers=n_workers,
            eps_logM=eps_logM,
            eps_y=eps_y,
            eps_p3=eps_p3,
        )

        for i_y in range(n_y):
            h_L   = F[:, i_y] * h0
            dh_logM = dF_dlogM[:, i_y] * h0
            dh_y  = dF_dy[:, i_y] * h0
            dh_p3 = dF_dp3[:, i_y] * h0

            F_mat, snr = assemble_fisher_matrix(
                h_L, dh_logM, dh_y, dh_p3, S_n, f_vec
            )
            sig_logM, sig_y, sig_3, cond = marginalized_errors(F_mat)

            sigma_logM[i_M, i_y] = sig_logM
            sigma_y_out[i_M, i_y] = sig_y
            sigma_p3[i_M, i_y]   = sig_3
            SNR_out[i_M, i_y]    = snr
            cond_out[i_M, i_y]   = cond

            if return_cov:
                cov_3x3, _ = marginalised_covariance(F_mat)
                cov_out[i_M, i_y] = cov_3x3

        dt      = time.perf_counter() - t0
        elapsed = time.perf_counter() - t_start_total
        remaining = dt * (n_M - i_M - 1)
        print(
            f"  M_Lz={M_Lz:.2e} M_sun  [{i_M+1}/{n_M}]  "
            f"{dt:.1f}s/iter  elapsed={elapsed:.0f}s  "
            f"ETA={remaining:.0f}s",
            flush=True,
        )

    out = {
        'M_Lz':            M_Lz_arr,
        'y':               y_arr,
        'sigma_logM':      sigma_logM,
        'sigma_y':         sigma_y_out,
        sigma3_key:        sigma_p3,
        'SNR':             SNR_out,
        'condition_number': cond_out,
    }
    if return_cov:
        out['cov'] = cov_out
    return out


def run_fisher_gSIS(
    M_Lz_Msun, y, gamma, m1_Msun, m2_Msun,
    z_s=3.0, n_f=512,
    eps_logM=5e-6, eps_y=5e-6, eps_gamma=5e-6,
    gl_dir=None, n_workers=1, return_cov=False,
    waveform_func=None,
):
    """
    Run Fisher forecasts for GW lensing by a generalized SIS (power-law) lens.

    The gSIS has potential ψ(r) = ψ₀ r^(3−γ) / (3−γ).  At gamma=2 this
    reduces to the standard SIS.  The Fisher parameter set is
    {log M_Lz, y, γ, log D_S, φ₀}.

    Parameters
    ----------
    M_Lz_Msun : float or array_like
    y : float or array_like
    gamma : float
        Power-law slope (2 = SIS; typical range 1.0–2.9).
    m1_Msun, m2_Msun : float
    z_s : float
    n_f : int
    eps_logM, eps_y, eps_gamma : float
        Central-difference steps.
    gl_dir : str or None
    n_workers : int
    return_cov : bool

    Returns
    -------
    dict with keys: M_Lz, y, sigma_logM, sigma_y, sigma_gamma, SNR,
                    condition_number [, cov].
    """
    return _run_fisher_3p(
        M_Lz_Msun=M_Lz_Msun, y=y, param3=gamma,
        m1_Msun=m1_Msun, m2_Msun=m2_Msun,
        lens_model='gSIS', sigma3_key='sigma_gamma',
        z_s=z_s, n_f=n_f,
        eps_logM=eps_logM, eps_y=eps_y, eps_p3=eps_gamma,
        gl_dir=gl_dir, n_workers=n_workers, return_cov=return_cov,
        waveform_func=waveform_func,
    )


def run_fisher_NFW(
    M_Lz_Msun, y, xs, m1_Msun, m2_Msun,
    z_s=3.0, n_f=512,
    eps_logM=5e-6, eps_y=5e-6, eps_xs=None,
    gl_dir=None, n_workers=1, return_cov=False,
    waveform_func=None,
):
    """
    Run Fisher forecasts for GW lensing by an NFW lens.

    The NFW scale radius xs = r_s / ξ₀ is the dimensionless concentration
    parameter.  The Fisher parameter set is {log M_Lz, y, xs, log D_S, φ₀}.

    Parameters
    ----------
    M_Lz_Msun : float or array_like
    y : float or array_like
    xs : float
        Dimensionless scale radius (typical range 0.01–1.0).
    m1_Msun, m2_Msun : float
    z_s : float
    n_f : int
    eps_logM, eps_y : float
    eps_xs : float or None
        Central-difference step for xs.  Defaults to max(5e-6, 5e-4 * xs).
    gl_dir : str or None
    n_workers : int
    return_cov : bool

    Returns
    -------
    dict with keys: M_Lz, y, sigma_logM, sigma_y, sigma_xs, SNR,
                    condition_number [, cov].
    """
    return _run_fisher_3p(
        M_Lz_Msun=M_Lz_Msun, y=y, param3=xs,
        m1_Msun=m1_Msun, m2_Msun=m2_Msun,
        lens_model='NFW', sigma3_key='sigma_xs',
        z_s=z_s, n_f=n_f,
        eps_logM=eps_logM, eps_y=eps_y, eps_p3=eps_xs,
        gl_dir=gl_dir, n_workers=n_workers, return_cov=return_cov,
        waveform_func=waveform_func,
    )


def run_fisher_NFW_c200(
    M_Lz_Msun, y, c200, m1_Msun, m2_Msun,
    z_s=3.0, z_L=1.0, n_f=512,
    eps_logM=5e-6, eps_y=5e-6, eps_c200=None,
    gl_dir=None, n_workers=1, return_cov=False,
):
    """
    Fisher forecast for NFW lens parameterised by concentration c200.

    For each lens mass M_Lz the dimensionless scale radius

        xs = r_s / ξ_E = r_200(M_Lz, z_L) / (c200 × ξ_E(M_Lz, z_L, z_s))

    is computed via :func:`c200_to_xs` and passed to the NFW solver.
    The Fisher parameter set is {ln M_Lz, y, c200, ln D_S, φ₀}.

    The derivative ∂F/∂c200 = -(xs/c200) × ∂F/∂xs is obtained analytically
    from the Jacobian dxs/dc200 = −xs/c200, so no extra FIONA calls are
    needed compared to the xs-basis calculation.  Equivalently the c200
    derivative is computed directly as a central difference in c200 (with
    automatic conversion to xs), giving the same result with no approximation.

    Parameters
    ----------
    M_Lz_Msun : float or array_like   Effective lens mass(es) [M_sun].
    y          : float or array_like   Source impact parameter(s).
    c200       : float                 Virial concentration.
    m1_Msun, m2_Msun : float          Binary masses [M_sun].
    z_s        : float                 Source redshift (default 3.0).
    z_L        : float                 Lens redshift for c200→xs conversion
                                       (default 1.0).
    n_f        : int                   Frequency grid points.
    eps_logM, eps_y : float            Finite-difference steps.
    eps_c200   : float or None         Step in c200 (default 0.01 * c200).
    gl_dir     : str or None
    n_workers  : int
    return_cov : bool                  Include 3×3 cov matrix in output.

    Returns
    -------
    dict with keys: M_Lz, y, sigma_logM, sigma_y, sigma_c200, SNR,
                    condition_number [, cov  (in the c200 basis)].
    """
    gl_dir = gl_dir or _DEFAULT_GL_DIR

    M_Lz_arr = np.atleast_1d(np.asarray(M_Lz_Msun, dtype=float))
    y_arr    = np.atleast_1d(np.asarray(y,          dtype=float))
    n_M      = len(M_Lz_arr)
    n_y      = len(y_arr)

    if eps_c200 is None:
        eps_c200 = max(5e-6, 1e-2 * float(c200))

    M_total  = m1_Msun + m2_Msun
    eta      = (m1_Msun * m2_Msun) / M_total ** 2
    D_L_Mpc  = luminosity_distance(z_s)

    f_hi = min(f_isco(M_total, z_s), _LISA_F_MAX)
    f_lo = max(f_1month(M_total, eta, z_s), _LISA_F_MIN)
    if f_lo >= f_hi:
        raise ValueError(
            f"No valid frequency band: f_1month={f_lo:.3e} >= f_ISCO={f_hi:.3e}."
        )
    f_vec = np.logspace(np.log10(f_lo), np.log10(f_hi), n_f)

    S_n = lisa_psd(f_vec)
    h0  = imrphenomd_amplitude(f_vec, M_total, eta, z_s, D_L_Mpc)

    sigma_logM  = np.full((n_M, n_y), np.nan)
    sigma_y_out = np.full((n_M, n_y), np.nan)
    sigma_c200  = np.full((n_M, n_y), np.nan)
    SNR_out     = np.full((n_M, n_y), np.nan)
    cond_out    = np.full((n_M, n_y), np.nan)
    if return_cov:
        cov_out = np.full((n_M, n_y, 3, 3), np.nan)

    t_start = time.perf_counter()
    for i_M, M_Lz in enumerate(M_Lz_arr):
        t0 = time.perf_counter()

        # xs corresponding to fiducial c200 and this M_Lz
        xs_fid = c200_to_xs(M_Lz, c200,       z_L=z_L, z_S=z_s)
        xs_p   = c200_to_xs(M_Lz, c200 + eps_c200, z_L=z_L, z_S=z_s)
        xs_m   = c200_to_xs(M_Lz, c200 - eps_c200, z_L=z_L, z_S=z_s)

        # Use xs-based derivatives; dF/dc200 via finite difference in c200
        # (equivalent to Jacobian: dF/dc200 = -(xs/c200) × dF/dxs)
        F, dF_dlogM, dF_dy, dF_dxs = compute_F_and_derivatives_3p(
            M_Lz_Msun=M_Lz,
            y_vec=y_arr,
            param3=xs_fid,
            f_vec=f_vec,
            lens_model='NFW',
            gl_dir=gl_dir,
            n_workers=n_workers,
            eps_logM=eps_logM,
            eps_y=eps_y,
            eps_p3=abs(xs_p - xs_m) / 2.0,  # actual xs step used
        )
        # dF/dc200 via Jacobian (exact, no extra FIONA calls)
        dF_dc200 = -(xs_fid / c200) * dF_dxs

        for i_y in range(n_y):
            h_L      = F[:, i_y] * h0
            dh_logM  = dF_dlogM[:, i_y] * h0
            dh_y     = dF_dy[:, i_y]    * h0
            dh_c200  = dF_dc200[:, i_y] * h0  # already in c200 basis

            # Fisher matrix is assembled in the c200 basis directly
            F_mat, snr = assemble_fisher_matrix(
                h_L, dh_logM, dh_y, dh_c200, S_n, f_vec
            )
            sig_logM, sig_y, sig_c, cond = marginalized_errors(F_mat)

            sigma_logM[i_M, i_y]  = sig_logM
            sigma_y_out[i_M, i_y] = sig_y
            sigma_c200[i_M, i_y]  = sig_c
            SNR_out[i_M, i_y]     = snr
            cond_out[i_M, i_y]    = cond

            if return_cov:
                cov_3x3, _ = marginalised_covariance(F_mat)
                cov_out[i_M, i_y] = cov_3x3

        dt        = time.perf_counter() - t0
        elapsed   = time.perf_counter() - t_start
        remaining = dt * (n_M - i_M - 1)
        xs_str    = f"xs={xs_fid:.3f}"
        print(
            f"  M_Lz={M_Lz:.2e} M_sun  [{i_M+1}/{n_M}]  {xs_str}  "
            f"{dt:.1f}s/iter  elapsed={elapsed:.0f}s  ETA={remaining:.0f}s",
            flush=True,
        )

    out = {
        'M_Lz':            M_Lz_arr,
        'y':               y_arr,
        'sigma_logM':      sigma_logM,
        'sigma_y':         sigma_y_out,
        'sigma_c200':      sigma_c200,
        'SNR':             SNR_out,
        'condition_number': cond_out,
    }
    if return_cov:
        out['cov'] = cov_out
    return out


def run_fisher_ULA(
    M_Lz_Msun, y, r_s, m1_Msun, m2_Msun,
    z_s=3.0, n_f=512,
    eps_logM=5e-6, eps_y=5e-6, eps_rs=None,
    gl_dir=None, n_workers=1, return_cov=False,
    waveform_func=None,
):
    """
    Run Fisher forecasts for GW lensing by a ULA (fuzzy DM) soliton lens.

    The soliton 3D density profile follows Schive+2014:
        ρ(r) = ρ_c / (1 + 0.091 (r/r_s)^2)^8

    The dimensionless core radius r_s = r_c / ξ₀ is the third Fisher parameter.
    The Fisher parameter set is {log M_Lz, y, r_s, log D_S, φ₀}.

    Parameters
    ----------
    M_Lz_Msun : float or array_like
    y : float or array_like
    r_s : float
        Soliton core radius in dimensionless lens-plane units.
    m1_Msun, m2_Msun : float
    z_s : float
    n_f : int
    eps_logM, eps_y : float
    eps_rs : float or None
        Central-difference step for r_s. Defaults to max(5e-6, 5e-4 * r_s).
    gl_dir : str or None
    n_workers : int
    return_cov : bool

    Returns
    -------
    dict with keys: M_Lz, y, sigma_logM, sigma_y, sigma_rs, SNR,
                    condition_number [, cov].
    """
    return _run_fisher_3p(
        M_Lz_Msun=M_Lz_Msun, y=y, param3=r_s,
        m1_Msun=m1_Msun, m2_Msun=m2_Msun,
        lens_model='ULA', sigma3_key='sigma_rs',
        z_s=z_s, n_f=n_f,
        eps_logM=eps_logM, eps_y=eps_y, eps_p3=eps_rs,
        gl_dir=gl_dir, n_workers=n_workers, return_cov=return_cov,
        waveform_func=waveform_func,
    )


def run_fisher_PointLens(
    M_Lz_Msun, y, xc, m1_Msun, m2_Msun,
    z_s=3.0, n_f=512,
    eps_logM=5e-6, eps_y=5e-6, eps_xc=None,
    gl_dir=None, n_workers=1, return_cov=False,
    waveform_func=None,
):
    """
    Run Fisher forecasts for GW lensing by a (softened) point-mass lens.

    The lens potential is ψ(r) = 0.5 psi0 log(r^2 + xc^2), which is an
    exact Plummer softening of the point mass.  In the limit xc → 0 this
    recovers the pure Schwarzschild (PBH) potential log(r).

    For PBH studies set xc to a small value (e.g. 1e-5) so the lens is
    effectively a point mass while avoiding the r=0 singularity.  The
    parameter sigma_xc returned by this function is then numerically
    meaningless (the constraint on the softening radius of a black hole),
    and the physically interesting outputs are sigma_logM and sigma_y.

    The Fisher parameter set is {log M_Lz, y, xc, log D_S, φ₀}.

    Parameters
    ----------
    M_Lz_Msun : float or array_like
    y : float or array_like
    xc : float
        Plummer softening radius (dimensionless).  Use ~1e-5 for PBH.
    m1_Msun, m2_Msun : float
    z_s : float
    n_f : int
    eps_logM, eps_y : float
    eps_xc : float or None
    gl_dir : str or None
    n_workers : int
    return_cov : bool

    Returns
    -------
    dict with keys: M_Lz, y, sigma_logM, sigma_y, sigma_xc, SNR,
                    condition_number [, cov].
    """
    return _run_fisher_3p(
        M_Lz_Msun=M_Lz_Msun, y=y, param3=xc,
        m1_Msun=m1_Msun, m2_Msun=m2_Msun,
        lens_model='PointLens', sigma3_key='sigma_xc',
        z_s=z_s, n_f=n_f,
        eps_logM=eps_logM, eps_y=eps_y, eps_p3=eps_xc,
        gl_dir=gl_dir, n_workers=n_workers, return_cov=return_cov,
        waveform_func=waveform_func,
    )
