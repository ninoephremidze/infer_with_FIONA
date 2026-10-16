"""
FIONA-based lensing amplification factor and its parameter derivatives.

All derivatives are computed via central finite differences. The key
efficiency gain is that FIONA evaluates F(w_vec, y_vec) for all source
positions simultaneously in one call, so the y-derivative is obtained
for free by batching y ± ε into the same FIONA evaluation.
"""
import os
import numpy as np

from ..paths import GL_RULES_DIR

_DEFAULT_GL_DIR = str(GL_RULES_DIR)

# fiona.utils reads FIONA_GL2D_DIR once at import time into a module-level
# variable.  Pre-set the env var so the snapshot captures the right path.
if not os.environ.get("FIONA_GL2D_DIR"):
    os.environ["FIONA_GL2D_DIR"] = _DEFAULT_GL_DIR

from ..fiona import CIS, SIS, NFW, PointLens, ULASoliton, AxisymmetricLens, FresnelNUFHTBatched  # noqa: E402
from ..fiona import utils as _fiona_utils                                # noqa: E402
from .utils import G_over_c3_Msun                                         # noqa: E402


class gSIS(AxisymmetricLens):
    r"""
    Generalized SIS (spherical power-law): ψ(r) = ψ₀ r^(3−γ) / (3−γ).

    gamma=2 recovers the standard SIS.  Valid for gamma ∈ (0, 3).
    """

    def __init__(self, psi0: float = 1.0, gamma: float = 2.0):
        self.psi0  = float(psi0)
        self.gamma = float(gamma)

    def psi_r(self, r):
        r   = np.asarray(r, dtype=float)
        eta = 3.0 - self.gamma
        return self.psi0 * r ** eta / eta


def _make_lens(lens_model, x_c):
    if lens_model == 'CIS':
        return CIS(psi0=1.0, xc=float(x_c))
    elif lens_model == 'SIS':
        return SIS(psi0=1.0)
    elif lens_model == 'NFW':
        raise ValueError("NFW requires an xs parameter; use lens_model='CIS'.")
    else:
        raise ValueError(f"Unknown lens_model: {lens_model!r}")


def _make_lens_3p(lens_model, param3):
    """Create a lens from (lens_model, third-parameter) pair."""
    if lens_model == 'CIS':
        return CIS(psi0=1.0, xc=float(param3))
    elif lens_model == 'SIS':
        return SIS(psi0=1.0)
    elif lens_model == 'gSIS':
        return gSIS(psi0=1.0, gamma=float(param3))
    elif lens_model == 'NFW':
        return NFW(psi0=1.0, xs=float(param3))
    elif lens_model == 'PointLens':
        return PointLens(psi0=1.0, xc=float(param3))
    elif lens_model == 'ULA':
        return ULASoliton(psi0=1.0, r_s=float(param3))
    else:
        raise ValueError(f"Unknown lens_model: {lens_model!r}")


def _make_solver(lens, gl_dir, n_workers):
    """Create a FresnelNUFHTBatched solver, patching the fiona.utils GL dir."""
    _fiona_utils._FIONA_GL2D_DIR = gl_dir
    os.environ["FIONA_GL2D_DIR"] = gl_dir
    return FresnelNUFHTBatched(
        lens,
        gl_dir=gl_dir,
        n_workers=n_workers,
        print_timing=False,
    )


def compute_F(M_Lz_Msun, y_vec, x_c, f_vec,
              lens_model='CIS',
              gl_dir=None,
              n_workers=1):
    """
    Evaluate the lensing amplification factor F(w(f), y).

    Parameters
    ----------
    M_Lz_Msun : float
        Effective lens mass [M_sun].
    y_vec : ndarray, shape (n_y,)
        Source impact parameters (dimensionless).
    x_c : float
        CIS core radius (dimensionless).
    f_vec : ndarray, shape (n_f,)
        GW frequencies [Hz].
    lens_model : str
        'CIS' (default).
    gl_dir : str or None
        Path to Gauss-Legendre node cache.
    n_workers : int
        Worker processes for FIONA parallelism.

    Returns
    -------
    ndarray, complex, shape (n_f, n_y)
        F(w(f_i), y_j) for all i, j.
    """
    gl_dir = gl_dir or _DEFAULT_GL_DIR
    w_vec = 8.0 * np.pi * G_over_c3_Msun * float(M_Lz_Msun) * np.asarray(f_vec)
    solver = _make_solver(_make_lens(lens_model, x_c), gl_dir, n_workers)
    return solver(w_vec, np.asarray(y_vec))  # shape (n_w, n_y)


def compute_F_and_derivatives(
    M_Lz_Msun,
    y_vec,
    x_c,
    f_vec,
    lens_model='CIS',
    gl_dir=None,
    n_workers=1,
    eps_logM=0.01,
    eps_y=0.01,
    eps_xc=None,
    w_high=100.0,
):
    """
    Evaluate F(w, y) and its derivatives w.r.t. {log M_Lz, y, x_c}.

    Uses central finite differences.  The y-derivative is computed for free
    by batching y ± ε into the base FIONA call (no extra solver overhead).

    Solver objects are reused across finite-difference calls that share the
    same lens model, avoiding redundant ProcessPoolExecutor creation.

    Number of solver creations per call:
      - 1  solver for lens_base  →  called 3× (base+y-batch, M_Lz+, M_Lz−)
      - 1  solver for lens xc+   →  called 1×
      - 1  solver for lens xc−   →  called 1×
      Total: 3 pool creations instead of 5.

    Parameters
    ----------
    M_Lz_Msun : float
    y_vec : ndarray, shape (n_y,)
    x_c : float
    f_vec : ndarray, shape (n_f,)
    lens_model : str
    gl_dir : str or None
    n_workers : int
    eps_logM : float
        Finite-diff step in log M_Lz (default 0.01).
    eps_y : float
        Finite-diff step in y (default 0.01; applied additively).
    eps_xc : float or None
        Finite-diff step in x_c. Defaults to max(0.001, 0.1 * x_c).
    w_high : float
        Dimensionless frequency threshold above which bGO analytic derivatives
        replace FIONA finite differences (CIS only).  Tambalo+2023 uses
        w_high ~ 100.  Set to np.inf to disable bGO entirely.

    Returns
    -------
    F : ndarray, complex, shape (n_f, n_y)
        Amplification factor at (f_i, y_j).
    dF_dlogM : ndarray, complex, shape (n_f, n_y)
        ∂F / ∂ log M_Lz
    dF_dy : ndarray, complex, shape (n_f, n_y)
        ∂F / ∂y
    dF_dxc : ndarray, complex, shape (n_f, n_y)
        ∂F / ∂x_c
    """
    gl_dir = gl_dir or _DEFAULT_GL_DIR
    y_vec = np.asarray(y_vec, dtype=float)
    f_vec = np.asarray(f_vec, dtype=float)
    n_y = len(y_vec)

    if eps_xc is None:
        eps_xc = max(5e-6, 5e-4 * float(x_c))

    w_base = 8.0 * np.pi * G_over_c3_Msun * float(M_Lz_Msun) * f_vec
    w_p    = w_base * np.exp(+eps_logM)
    w_m    = w_base * np.exp(-eps_logM)

    # ------------------------------------------------------------------ #
    # One solver for the base lens — reused for all three w-grid calls.   #
    # ------------------------------------------------------------------ #
    solver_base = _make_solver(_make_lens(lens_model, x_c), gl_dir, n_workers)

    # Base call: y, y+eps_y, y-eps_y batched into one evaluation.
    y_batch = np.concatenate([y_vec, y_vec + eps_y, y_vec - eps_y])
    F_batch = solver_base(w_base, y_batch)   # shape (n_f, 3*n_y)

    F    = F_batch[:, :n_y]
    F_yp = F_batch[:, n_y: 2 * n_y]
    F_ym = F_batch[:, 2 * n_y:]
    dF_dy = (F_yp - F_ym) / (2.0 * eps_y)

    # log M_Lz derivative: same lens, shifted w grids.
    F_Mp = solver_base(w_p, y_vec)
    F_Mm = solver_base(w_m, y_vec)
    dF_dlogM = (F_Mp - F_Mm) / (2.0 * eps_logM)

    # ------------------------------------------------------------------ #
    # Two separate solvers for x_c perturbations (different lens shape).  #
    # ------------------------------------------------------------------ #
    solver_xcp = _make_solver(_make_lens(lens_model, float(x_c) + eps_xc), gl_dir, n_workers)
    solver_xcm = _make_solver(_make_lens(lens_model, float(x_c) - eps_xc), gl_dir, n_workers)

    F_xcp = solver_xcp(w_base, y_vec)
    F_xcm = solver_xcm(w_base, y_vec)
    dF_dxc = (F_xcp - F_xcm) / (2.0 * eps_xc)

    # ------------------------------------------------------------------ #
    # bGO override for CIS at high dimensionless frequencies (w > w_high) #
    # ------------------------------------------------------------------ #
    if lens_model == 'CIS' and np.isfinite(w_high) and np.any(w_base > w_high):
        from .cis_go import cis_dF_bGO
        hi_mask = w_base > w_high       # shape (n_f,)
        w_hi = w_base[hi_mask]          # subset of w values in bGO regime
        for i_y, y_val in enumerate(y_vec):
            dlogw, dy, dxc = cis_dF_bGO(w_hi, float(y_val), float(x_c))
            dF_dlogM[hi_mask, i_y] = dlogw
            dF_dy   [hi_mask, i_y] = dy
            dF_dxc  [hi_mask, i_y] = dxc

    # ------------------------------------------------------------------ #
    # Tambalo normalization correction for CIS                            #
    # ------------------------------------------------------------------ #
    # FIONA returns F_full(w,y) which includes the absolute Fermat phase  #
    # exp(iw φ_m) of the minimum image.  Tambalo's contour-method and     #
    # time-domain finite differences use F_T = F_full * exp(-iw φ_m),    #
    # which normalizes the minimum time delay to zero.  This subtraction  #
    # is physically required because the absolute arrival time of the      #
    # minimum image is degenerate with the (unknown) coalescence time t_c. #
    #                                                                      #
    # When Tambalo computes dF_T/dθ via finite differences, they get:     #
    #   dF_T/dy     = (dF_full/dy   + iw α_min       F_full) exp(-iw φ_m) #
    #   dF_T/dlogM  = (dF_full/dlogM - iw φ_m        F_full) exp(-iw φ_m) #
    #   dF_T/dxc    = (dF_full/dxc  - iw ψ_min_xc   F_full) exp(-iw φ_m) #
    # where α_min = -∂φ_m/∂y and ψ_min_xc = ∂φ_m/∂xc.                  #
    #                                                                      #
    # In Fisher matrix elements the exp(-iw φ_m) phase factors cancel, so #
    # the effective correction to apply to FIONA's derivatives is:         #
    #   dF_dy   += +iw_base * α_min     * F  (per y_val column)           #
    #   dF_dlogM-=  iw_base * φ_m       * F                               #
    #   dF_dxc  -=  iw_base * ψ_min_xc  * F                               #
    # ------------------------------------------------------------------ #
    if lens_model == 'CIS':
        from .cis_go import cis_images, cis_image_props, _alpha

        for i_y, y_val in enumerate(y_vec):
            y_val = float(y_val)
            xc_val = float(x_c)

            # Find minimum image (Morse index 0, largest positive x root)
            images = cis_images(y_val, xc_val)
            x_min = max(images)   # minimum Fermat potential image

            # Minimum Fermat potential φ_m and deflection angle α_min
            phi_m, _, _, _ = cis_image_props(x_min, y_val, xc_val)
            alpha_min = _alpha(x_min, xc_val)   # = -∂φ_m/∂y

            # ψ_min_xc = ∂φ_m/∂xc via finite difference (x_min held fixed;
            # stationarity kills ∂x_min/∂xc contribution at leading order)
            eps_phi = max(1e-7, 1e-5 * xc_val)
            phi_xcp, _, _, _ = cis_image_props(x_min, y_val, xc_val + eps_phi)
            phi_xcm, _, _, _ = cis_image_props(x_min, y_val, xc_val - eps_phi)
            psi_min_xc = (phi_xcp - phi_xcm) / (2.0 * eps_phi)

            F_col = F[:, i_y]   # shape (n_f,)

            # Apply corrections: match Tambalo's φ_m-subtracted convention
            dF_dy   [:, i_y] += 1j * w_base * alpha_min   * F_col
            dF_dlogM[:, i_y] -= 1j * w_base * phi_m       * F_col
            dF_dxc  [:, i_y] -= 1j * w_base * psi_min_xc  * F_col

    return F, dF_dlogM, dF_dy, dF_dxc


def compute_F_3p(M_Lz_Msun, y_vec, param3, f_vec, lens_model,
                 gl_dir=None, n_workers=1):
    """
    Evaluate F(w(f), y) for any lens model with a third shape parameter.

    Like compute_F but uses _make_lens_3p so 'gSIS' and 'NFW' are supported.
    """
    gl_dir = gl_dir or _DEFAULT_GL_DIR
    w_vec  = 8.0 * np.pi * G_over_c3_Msun * float(M_Lz_Msun) * np.asarray(f_vec)
    solver = _make_solver(_make_lens_3p(lens_model, param3), gl_dir, n_workers)
    return solver(w_vec, np.asarray(y_vec))


def compute_F_and_derivatives_3p(
    M_Lz_Msun,
    y_vec,
    param3,
    f_vec,
    lens_model,
    gl_dir=None,
    n_workers=1,
    eps_logM=5e-6,
    eps_y=5e-6,
    eps_p3=None,
):
    """
    Evaluate F(w, y) and ∂F/∂{log M_Lz, y, param3} for any lens model.

    Generic version of compute_F_and_derivatives that accepts any lens via
    _make_lens_3p.  Supported models: 'CIS' (param3=x_c), 'gSIS' (param3=gamma),
    'NFW' (param3=xs).

    Uses central finite differences. The y-derivative is batched into the base
    FIONA call at no extra cost.

    Parameters
    ----------
    M_Lz_Msun : float
    y_vec : ndarray, shape (n_y,)
    param3 : float
        Third lens parameter: x_c for CIS, gamma for gSIS, xs for NFW.
    f_vec : ndarray, shape (n_f,)
    lens_model : str
    gl_dir : str or None
    n_workers : int
    eps_logM : float
    eps_y : float
    eps_p3 : float or None
        Central-difference step for param3.  Defaults to max(5e-6, 5e-4 * |param3|).

    Returns
    -------
    F, dF_dlogM, dF_dy, dF_dp3 : ndarrays, complex, shape (n_f, n_y)
    """
    gl_dir = gl_dir or _DEFAULT_GL_DIR
    y_vec  = np.asarray(y_vec, dtype=float)
    f_vec  = np.asarray(f_vec, dtype=float)
    n_y    = len(y_vec)

    if eps_p3 is None:
        eps_p3 = max(5e-6, 5e-4 * abs(float(param3)))

    w_base = 8.0 * np.pi * G_over_c3_Msun * float(M_Lz_Msun) * f_vec
    w_p    = w_base * np.exp(+eps_logM)
    w_m    = w_base * np.exp(-eps_logM)

    solver_base = _make_solver(_make_lens_3p(lens_model, param3), gl_dir, n_workers)

    y_batch = np.concatenate([y_vec, y_vec + eps_y, y_vec - eps_y])
    F_batch = solver_base(w_base, y_batch)

    F    = F_batch[:, :n_y]
    F_yp = F_batch[:, n_y: 2 * n_y]
    F_ym = F_batch[:, 2 * n_y:]
    dF_dy = (F_yp - F_ym) / (2.0 * eps_y)

    F_Mp = solver_base(w_p, y_vec)
    F_Mm = solver_base(w_m, y_vec)
    dF_dlogM = (F_Mp - F_Mm) / (2.0 * eps_logM)

    solver_pp = _make_solver(_make_lens_3p(lens_model, float(param3) + eps_p3), gl_dir, n_workers)
    solver_pm = _make_solver(_make_lens_3p(lens_model, float(param3) - eps_p3), gl_dir, n_workers)

    F_pp = solver_pp(w_base, y_vec)
    F_pm = solver_pm(w_base, y_vec)
    dF_dp3 = (F_pp - F_pm) / (2.0 * eps_p3)

    return F, dF_dlogM, dF_dy, dF_dp3


# --------------------------------------------------------------------------- #
# Lens-parameter batching (additive; existing functions above are untouched) #
# --------------------------------------------------------------------------- #
#
# fiona.axisym.FresnelNUFHTLensBatched packs many lens instances (any mix of
# types/parameters) into a single NUFHT call per frequency, instead of
# building one FresnelNUFHTBatched solver per lens (as compute_F_and_derivatives_3p
# and _make_solver do above). This wrapper exposes that for a grid over a
# lens's third shape parameter (x_c for CIS, gamma for gSIS, xs for NFW, xc for
# PointLens, r_s for ULA), at a single fixed M_Lz.

from ..fiona.axisym import FresnelNUFHTLensBatched  # noqa: E402


def compute_F_param_batch(M_Lz_Msun, y_vec, param3_vec, f_vec, lens_model,
                          gl_dir=None, n_workers=1):
    """
    Evaluate F(w(f), y) for MANY values of a lens's third shape parameter
    simultaneously, at fixed M_Lz, in one batched NUFHT call per frequency.

    Generalises ``compute_F``/``compute_F_3p`` (which build one solver per
    parameter value) by packing every ``param3_vec`` value as extra columns
    into a single ``fiona.axisym.FresnelNUFHTLensBatched`` call, exploiting
    the fact that the radial quadrature nodes and NUFHT output points depend
    only on ``w`` and ``y_vec`` -- never on the lens's shape parameter.

    Parameters
    ----------
    M_Lz_Msun : float
        Effective lens mass [M_sun] (fixed for this call).
    y_vec : ndarray, shape (n_y,)
        Source impact parameters (dimensionless).
    param3_vec : ndarray, shape (n_p3,)
        Grid of third-parameter values: x_c for CIS, gamma for gSIS, xs for
        NFW, xc for PointLens, r_s for ULA.
    f_vec : ndarray, shape (n_f,)
        GW frequencies [Hz].
    lens_model : str
        One of 'CIS', 'SIS', 'gSIS', 'NFW', 'PointLens', 'ULA'
        (same set supported by ``_make_lens_3p``).
    gl_dir : str or None
        Path to Gauss-Legendre node cache.
    n_workers : int
        Worker processes for FIONA parallelism.

    Returns
    -------
    ndarray, complex, shape (n_f, n_p3, n_y)
        F(w(f_i), param3_j, y_k) for all i, j, k.
    """
    gl_dir = gl_dir or _DEFAULT_GL_DIR
    param3_vec = np.asarray(param3_vec, dtype=float)
    w_vec = 8.0 * np.pi * G_over_c3_Msun * float(M_Lz_Msun) * np.asarray(f_vec)

    lenses = [_make_lens_3p(lens_model, p3) for p3 in param3_vec]

    _fiona_utils._FIONA_GL2D_DIR = gl_dir
    os.environ["FIONA_GL2D_DIR"] = gl_dir
    solver = FresnelNUFHTLensBatched(
        lenses,
        gl_dir=gl_dir,
        n_workers=n_workers,
        print_timing=False,
    )
    return solver(w_vec, np.asarray(y_vec))  # shape (n_w, n_p3, n_y)
