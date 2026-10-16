"""
Noise power spectral densities for LISA and aLIGO.

LISA: analytic sky-averaged PSD from Robson, Cornish & Liu (2019), arXiv:1803.01944.

aLIGO: tabulated curves from pycbc, covering design and past/future observing runs.
All aLIGO functions share the same interface: f -> S_n(f) [Hz^{-1}], returning
np.inf outside the tabulated range.  Pass any of them as psd_func to run_fisher().

Available aLIGO functions
-------------------------
ligo_psd            Zero-detuned high-power design sensitivity (O5 target).
                    Same curve as Tambalo+2023.  pycbc: aLIGOZeroDetHighPower.
ligo_o3_psd         O3 (2019–2020) low sensitivity.
                    pycbc: aLIGOO3LowT1800545.
ligo_o4_psd         O4 intermediate projected sensitivity.
                    pycbc: aLIGOAdVO4IntermediateT1800545.
ligo_aplus_psd      A+ upgrade design sensitivity (post-O5 goal).
                    pycbc: aLIGOAPlusDesignSensitivityT1800042.
"""
import numpy as np


def _pycbc_psd(psd_func_name, f):
    """Shared helper: load a pycbc PSD by name and interpolate onto f."""
    import importlib
    import pycbc.psd as _pycbc_psd_mod
    func = getattr(_pycbc_psd_mod, psd_func_name)
    f = np.asarray(f, dtype=float)
    delta_f = 0.1
    n = int(np.ceil(f.max() * 1.05 / delta_f)) + 2
    psd_obj = func(n, delta_f, 1.0)
    f_tab = psd_obj.sample_frequencies.numpy()
    s_tab = psd_obj.numpy()
    return np.interp(f, f_tab, s_tab, left=np.inf, right=np.inf)


def lisa_psd(f):
    """
    Sky-averaged LISA strain noise PSD S_n(f) [Hz^{-1}].

    Parameters
    ----------
    f : array_like
        Frequency [Hz].  Must be > 0.

    Returns
    -------
    ndarray
        S_n(f) in units of Hz^{-1} (strain^2/Hz).
    """
    f = np.asarray(f, dtype=float)

    L = 2.5e9          # arm length [m]
    f_star = 19.09e-3  # transfer frequency c / (2π L) [Hz]

    # Optical metrology system (OMS) noise [m^2/Hz]
    P_oms = (1.5e-11)**2 * (1.0 + (2e-3 / f)**4)

    # Test-mass acceleration noise [m^2/Hz]
    P_acc = ((3e-15)**2
             * (1.0 + (0.4e-3 / f)**2)
             * (1.0 + (f / 8e-3)**4)
             / (2.0 * np.pi * f)**4)

    # Sky- and polarisation-averaged transfer function
    R_star = (3.0 / 20.0) / (1.0 + 0.6 * (f / f_star)**2)

    S_n = ((10.0 / (3.0 * L**2))
           * (P_oms + 2.0 * (1.0 + np.cos(f / f_star)**2) * P_acc)
           / R_star)

    return S_n


def ligo_psd(f):
    """
    aLIGO zero-detuned high-power (ZDHP) design sensitivity [Hz^{-1}].

    Target sensitivity for O5 (~2026–2027).  Used by Tambalo+2023.
    pycbc curve: aLIGOZeroDetHighPower.
    """
    return _pycbc_psd('aLIGOZeroDetHighPower', f)


def ligo_o3_psd(f):
    """
    aLIGO O3 (2019–2020) low sensitivity [Hz^{-1}].

    Measured/projected sensitivity from the third observing run.
    pycbc curve: aLIGOO3LowT1800545 (LIGO-T1800545).
    """
    return _pycbc_psd('aLIGOO3LowT1800545', f)


def ligo_o4_psd(f):
    """
    aLIGO O4 proxy sensitivity [Hz^{-1}].

    Uses the aLIGO 140 Mpc BNS-range curve (LIGO-T1800545), which gives a
    BNS horizon of ~157 Mpc — close to actual O4 performance (H1 ~140 Mpc,
    L1 ~160 Mpc).  This is a tabulated projection, not a measured PSD;
    for a real measured O4 PSD download from LIGO DCC and use
    pycbc.psd.read_from_txt.
    pycbc curve: aLIGO140MpcT1800545.
    """
    return _pycbc_psd('aLIGO140MpcT1800545', f)


def ligo_aplus_psd(f):
    """
    A+ upgrade design sensitivity [Hz^{-1}].

    Post-O5 target with squeezed light and improved coatings.
    pycbc curve: aLIGOAPlusDesignSensitivityT1800042 (LIGO-T1800042).
    """
    return _pycbc_psd('aLIGOAPlusDesignSensitivityT1800042', f)
