"""
GW waveform generation for Fisher forecasts.

Uses the Stationary Phase Approximation (SPA) for the inspiral amplitude,
which is the dominant contribution for LISA sources.  No pycbc/lalsuite
dependency required.
"""
import numpy as np
from .utils import G_SI, c_SI, M_sun_kg, Mpc_m

_1MONTH_S = 30.44 * 24 * 3600.0  # 1 month in seconds


def f_isco(M_total_Msun, z_s):
    """
    Observed (redshifted) ISCO frequency for a circular binary [Hz].

    Parameters
    ----------
    M_total_Msun : float
        Total source-frame mass [M_sun].
    z_s : float
        Source redshift.

    Returns
    -------
    float
        f_ISCO [Hz].
    """
    M_obs_kg = M_total_Msun * (1.0 + z_s) * M_sun_kg
    return c_SI**3 / (6.0**1.5 * np.pi * G_SI * M_obs_kg)


def f_1month(M_total_Msun, eta, z_s):
    """
    GW frequency at which 1 month of integration time remains before merger.

    Uses the leading-order (Newtonian) chirp-time formula.

    Parameters
    ----------
    M_total_Msun : float
        Total source-frame mass [M_sun].
    eta : float
        Symmetric mass ratio (0.25 for equal mass).
    z_s : float
        Source redshift.

    Returns
    -------
    float
        f_1month [Hz].
    """
    M_cz_kg = eta**0.6 * M_total_Msun * (1.0 + z_s) * M_sun_kg
    G_Mcz_over_c3 = G_SI * M_cz_kg / c_SI**3

    # t_chirp(f) = 5/(256 eta) * (G M_total_z / c^3)^{-5/3} * (pi f)^{-8/3}
    # Note: written in terms of M_cz = eta^{3/5} M_total_z
    # t = 5/256 * (G M_cz / c^3)^{-5/3} * (pi f)^{-8/3}
    # Solve for f: f = [5 / (256 * t * pi^{8/3})]^{3/8} * (G M_cz / c^3)^{-5/8}
    t = _1MONTH_S
    f = ((5.0 / (256.0 * t)) ** (3.0 / 8.0)
         * (np.pi ** (-1.0)) ** 1.0
         * G_Mcz_over_c3 ** (-5.0 / 8.0)
         * np.pi ** (-1.0))
    # Cleaner: f = (5/(256 t))^{3/8} (pi G_Mcz_over_c3)^{-5/8} / pi
    f = (5.0 / (256.0 * t * np.pi**(8.0 / 3.0)))**(3.0 / 8.0) * G_Mcz_over_c3**(-5.0 / 8.0)
    return float(f)


def imrphenomd_pycbc_amplitude(f_vec, m1_Msun, m2_Msun, z_s, D_L_Mpc, f_lo, f_hi,
                               delta_f=None):
    """
    |h_+(f)| from pycbc IMRPhenomD, interpolated onto f_vec.

    Generates the waveform using detector-frame masses m1*(1+z_s), m2*(1+z_s),
    exactly as in Tambalo+2023.  pycbc returns a uniform frequency grid; the
    amplitude is linearly interpolated onto the caller's (typically log-spaced)
    f_vec and zeroed outside [f_lo, f_hi].

    Parameters
    ----------
    f_vec : ndarray, shape (n_f,)   Frequency grid [Hz] to evaluate on.
    m1_Msun, m2_Msun : float        Source-frame component masses [M_sun].
    z_s : float                     Source redshift.
    D_L_Mpc : float                 Luminosity distance [Mpc].
    f_lo, f_hi : float              Frequency window [Hz].
    delta_f : float, optional
        Frequency resolution [Hz] for pycbc.  Defaults to 1e-7 Hz (suitable for
        LISA sources with chirp durations ~1 month).  For short-chirp LIGO sources
        use a larger value, e.g. 1/128 ≈ 0.0078 Hz.

    Returns
    -------
    ndarray, shape (n_f,)  |h_+(f)| [Hz^{-1/2}]
    """
    from pycbc.waveform import get_fd_waveform

    m1_det = m1_Msun * (1.0 + z_s)   # detector-frame [M_sun]
    m2_det = m2_Msun * (1.0 + z_s)

    if delta_f is None:
        # delta_f = 1/T_segment; T must exceed chirp duration at f_lo (~1 month).
        # 1 month ≈ 2.63e6 s → delta_f ≈ 3.8e-7 Hz; use 1e-7 for safety margin.
        delta_f = 1e-7   # Hz

    hp, _ = get_fd_waveform(
        approximant='IMRPhenomD',
        mass1=m1_det,
        mass2=m2_det,
        f_lower=float(f_lo),
        delta_f=delta_f,
        distance=float(D_L_Mpc),
    )
    f_pycbc = hp.sample_frequencies.numpy()
    h_pycbc = np.abs(hp.numpy())

    f_vec = np.asarray(f_vec, dtype=float)
    h0 = np.interp(f_vec, f_pycbc, h_pycbc, left=0.0, right=0.0)
    h0[(f_vec < f_lo) | (f_vec > f_hi)] = 0.0
    return h0


def imrphenomd_amplitude(f_vec, M_total_Msun, eta, z_s, D_L_Mpc):
    """
    |h_+(f)| via the inspiral SPA (Stationary Phase Approximation).

    Only frequencies within [f_1month(M, eta, z), f_ISCO(M, z)] contribute;
    the amplitude is set to zero outside this window.

    This is the leading-order (Newtonian) inspiral amplitude and is
    sufficient for computing the shape of |h|^2 / S_n needed for Fisher
    matrices.

    Parameters
    ----------
    f_vec : array_like
        Frequency array [Hz].
    M_total_Msun : float
        Total source-frame binary mass [M_sun].
    eta : float
        Symmetric mass ratio (0.25 for equal mass).
    z_s : float
        Source redshift.
    D_L_Mpc : float
        Luminosity distance [Mpc].

    Returns
    -------
    ndarray
        |h_+(f)| [Hz^{-1/2}] — real, non-negative.
    """
    f_vec = np.asarray(f_vec, dtype=float)

    M_cz_kg = eta**0.6 * M_total_Msun * (1.0 + z_s) * M_sun_kg
    D_L_m = D_L_Mpc * Mpc_m

    # Frequency window
    f_lo = f_1month(M_total_Msun, eta, z_s)
    f_hi = f_isco(M_total_Msun, z_s)

    # Overall amplitude constant  C = sqrt(5/96) / pi^{2/3} * (G M_cz)^{5/6} / (c^{3/2} D_L)
    G_Mcz = G_SI * M_cz_kg
    C = (np.sqrt(5.0 / 96.0) / np.pi**(2.0 / 3.0)
         * G_Mcz**(5.0 / 6.0) / (c_SI**(3.0 / 2.0) * D_L_m))

    h = C * f_vec**(-7.0 / 6.0)
    h = np.where((f_vec >= f_lo) & (f_vec <= f_hi), h, 0.0)
    return h
