"""Lensing parameters -> dark-matter parameters, following T23 (Tambalo et al. 2023) as written in
the paper (Sec. "Constraints on dark matter models"), with these fixes relative to earlier notebooks:

  * xi_0 uses the lens mass M_L = M_Lz/(1+z_L), not M_Lz (infer_with_fiona.fisher.utils.xi_E_kpc uses M_Lz),
  * the ULA conversion always solves the Schive r_c and M_c relations jointly for (m_a, M_vir),
  * one lens redshift for all three profiles: T23's choice, the z_L that maximizes (1+z_L) d_eff.

Units: masses Msun, lengths kpc, d_eff Gpc, sigma/m cm^2/g, m_a eV. Cosmology flat LCDM, H0=70, Om0=0.3
(as infer_with_fiona.fisher.utils).
"""
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .fisher.utils import G_SI, M_sun_kg, Mpc_m, c_SI, _comoving_distance_m, luminosity_distance

KPC_M = Mpc_m / 1.0e3
T_AGE_GYR = 5.0


@lru_cache(maxsize=None)
def z_source(d_L_Mpc):
    """Source redshift from its luminosity distance."""
    return brentq(lambda z: luminosity_distance(z) - d_L_Mpc, 1e-6, 10.0)


@lru_cache(maxsize=None)
def d_eff_Gpc(z_L, z_S):
    """d_eff = D_L D_LS / ((1+z_L) D_S) in Gpc (T23 Eq. 8), angular-diameter distances."""
    chi_L, chi_S = _comoving_distance_m(z_L), _comoving_distance_m(z_S)
    D_L, D_S, D_LS = chi_L / (1 + z_L), chi_S / (1 + z_S), (chi_S - chi_L) / (1 + z_S)
    return D_L * D_LS / ((1 + z_L) * D_S) / (1.0e3 * Mpc_m)


@lru_cache(maxsize=None)
def zL_pessimistic(z_S):
    """T23's conservative lens redshift: maximizes (1+z_L) d_eff."""
    r = minimize_scalar(lambda z: -(1 + z) * d_eff_Gpc(z, z_S), bounds=(1e-6, z_S - 1e-6), method='bounded')
    return float(r.x)


def xi0_kpc(M_Lz, z_L, z_S):
    """Einstein-radius scale xi_0 = sqrt(4 G M_L D_L D_LS / (c^2 D_S)), M_L = M_Lz/(1+z_L)
    (equivalently sqrt(4 G M_Lz d_eff / c^2))."""
    return np.sqrt(4 * G_SI * np.asarray(M_Lz) * M_sun_kg * d_eff_Gpc(z_L, z_S) * 1.0e3 * Mpc_m / c_SI ** 2) / KPC_M


# ── SIDM (CIS lens) ─────────────────────────────────────────────────────────────────────

def Mvir_sidm(M_Lz, z_L, z_S):
    """Virial mass of the CIS halo, leading order in r_c/r_vir: M_vir = (M_Lz M_0^(1/3) / gamma_c)^(3/4),
    M_0 = 6.5e17 Msun (d_eff/Gpc)^-3, gamma_c = 2 (1+z_L)^2 (h = 0.7)."""
    M0 = 6.5e17 * d_eff_Gpc(z_L, z_S) ** -3
    return (np.asarray(M_Lz) * M0 ** (1 / 3) / (2 * (1 + z_L) ** 2)) ** 0.75


def sigma_over_m(x_c, M_Lz, z_L, z_S, t_age_gyr=T_AGE_GYR):
    """<sigma v>/m [cm^2/g] from the CIS core radius (paper Eq. sidm_sigma)."""
    d = d_eff_Gpc(z_L, z_S)
    rhs = np.asarray(x_c) * (1 + z_L) * d * (Mvir_sidm(M_Lz, z_L, z_S) / 1e12) ** (1 / 3) / 100.3
    return rhs ** 2 / (t_age_gyr / 5.0)


# ── ULA soliton ─────────────────────────────────────────────────────────────────────────

def ma_and_Mvir(r_s, M_Lz, z_L, z_S):
    """Axion mass [eV] and host virial mass [Msun] from the soliton core radius r_c = r_s xi_0 and the
    lens mass taken as the soliton core mass M_c = M_L (paper Eq. ula_ma: Schive r_c relation and
    T23's M_c relation solved jointly)."""
    M_c = np.asarray(M_Lz) / (1 + z_L)
    r_c = np.asarray(r_s) * xi0_kpc(M_Lz, z_L, z_S)
    m22 = np.sqrt(6.7e7 * 1.6 * np.sqrt(1 + z_L) / (10 ** (1 / 3) * r_c * M_c))
    return m22 * 1e-22, 1e9 * (1.6 / (m22 * r_c)) ** 3


# ── PBH ─────────────────────────────────────────────────────────────────────────────────

def M_pbh(M_Lz, z_L):
    return np.asarray(M_Lz) / (1 + z_L)


def _self_test():
    """Reproduce the translate notebooks where they agree with the fixed formulas."""
    zS = z_source(500.0)
    zL = zL_pessimistic(zS)
    assert abs(zS - 0.1080) < 5e-4 and abs(zL - 0.0519) < 5e-4, (zS, zL)
    s = float(sigma_over_m(0.02, 100.0, zL, zS))
    assert abs(s / 8.95e-14 - 1) < 0.01, s                      # notebook: 8.95e-14 cm^2/g
    assert abs(float(Mvir_sidm(100.0, zL, zS)) / 2.744e6 - 1) < 0.01
    # ULA: notebook (z_L = z_S/2, xi_0 with M_Lz, M_c = M_Lz, M_Lz = 1e3, r_s = 0.1) gave m_a = 5.97e4 x 1e-22.
    # With xi_0 and M_c both using M_L, m_a ∝ (r_c M_c)^-1/2 rises by (1+z_L)^(1/4) (1+z_L)^(1/2).
    zh = 0.5 * zS
    m, _ = ma_and_Mvir(0.1, 1e3, zh, zS)
    expect = 5.97e4 * 1e-22 * (1 + zh) ** 0.75
    assert abs(float(m) / expect - 1) < 0.01, (m, expect)
    return zS, zL


if __name__ == '__main__':
    zS, zL = _self_test()
    print(f'self-test passed: z_S = {zS:.4f}, pessimistic z_L = {zL:.4f}, d_eff = {d_eff_Gpc(zL, zS):.4f} Gpc')
