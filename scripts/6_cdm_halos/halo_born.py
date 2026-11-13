"""NFW halos as GW lenses: physical mapping and the Born-approximation amplification factor.

Units follow C24 (and the package's 'NFWkappa' lens): lengths in units of the scale radius r_s
(xi_0 = r_s), w = 8 pi G M_Lz f / c^3 with M_Lz the redshifted mass whose Einstein radius is r_s,
and kappa = 4 kappa_s, kappa_s = rho_s r_s / Sigma_cr.

Born approximation (first order in the potential). With psi-tilde = -2 kappa-tilde / k^2,

    F(w, y) ~= exp(-i w psi(y)) [1 + delta(w, y)],
    delta(w, y) = (i w / pi) Int_0^inf dk/k J0(k y) kappa-tilde(k) (exp(-i k^2 / 2w) - 1),

where exp(-i w psi(y)) is a frequency-linear phase (an arrival-time shift, absorbed by t_c). For
large w, delta -> kappa(y), the geometric-optics magnification. With u = k^2 / 2w,

    delta = (i w / 2 pi) Int_0^inf du/u J0(y sqrt(2 w u)) kappa-tilde(sqrt(2 w u)) (exp(-i u) - 1).

kappa-tilde is the 2D Fourier transform of the convergence, equal to the 3D transform of rho:
for an NFW truncated at r_200 = c r_s (Scoccimarro et al. 2001), in r_s units,

    kappa-tilde(q) = 4 pi kappa_s { sin q [Si((1+c)q) - Si(q)] - sin(cq) / ((1+c) q)
                                    + cos q [Ci((1+c)q) - Ci(q)] },

and for the untruncated profile (FIONA's NFW potential), sin q [pi/2 - Si q] - cos q Ci q.
"""
import numpy as np
from scipy.special import sici, j0
import astropy.units as u
from astropy.constants import G, c as C
from astropy.cosmology import Planck18 as COSMO

MSUN_S = (G * u.Msun / C ** 3).to(u.s).value     # G Msun / c^3 [s]


def m_nfw(x):
    return np.log1p(x) - x / (1 + x)


def halo(M200, c200, z_L, z_S):
    """Physical NFW halo -> lensing quantities (dict). M200 [Msun], c200 w.r.t. rho_crit(z_L)."""
    rhoc = COSMO.critical_density(z_L).to(u.Msun / u.pc ** 3).value
    DL = COSMO.angular_diameter_distance(z_L).to(u.pc).value
    DS = COSMO.angular_diameter_distance(z_S).to(u.pc).value
    DLS = COSMO.angular_diameter_distance_z1z2(z_L, z_S).to(u.pc).value
    G_pc = (G / C ** 2).to(u.pc / u.Msun).value              # G/c^2 [pc/Msun]
    Scr = DS / (4 * np.pi * G_pc * DL * DLS)                 # [Msun/pc^2]
    r200 = (3 * M200 / (4 * np.pi * 200 * rhoc)) ** (1 / 3)
    rs = r200 / c200
    rhos = M200 / (4 * np.pi * rs ** 3 * m_nfw(c200))
    kappa_s = rhos * rs / Scr
    M_Lz = rs ** 2 * DS * (1 + z_L) / (4 * G_pc * DL * DLS)   # Einstein radius of M_Lz/(1+z_L) = r_s
    return dict(M200=M200, c200=c200, z_L=z_L, z_S=z_S, rs_pc=rs, rhos=rhos, kappa_s=kappa_s,
                kappa=4 * kappa_s, M_Lz=M_Lz, Sigma_cr=Scr)


def w_of_f(M_Lz, f):
    return 8 * np.pi * MSUN_S * M_Lz * np.asarray(f, float)


def kappa_tilde(q, kappa_s, c200=None):
    """2D Fourier transform of the NFW convergence (r_s units); untruncated if c200 is None."""
    q = np.asarray(q, float)
    si, ci = sici(q)
    if c200 is None:
        b = np.sin(q) * (np.pi / 2 - si) - np.cos(q) * ci
    else:
        si2, ci2 = sici((1 + c200) * q)
        b = np.sin(q) * (si2 - si) - np.sin(c200 * q) / ((1 + c200) * q) + np.cos(q) * (ci2 - ci)
    return 4 * np.pi * kappa_s * b


U_MAX = 2000.0


def _u_grid(a_max):
    """u-grid resolving both exp(-i u) and J0(a sqrt(u)) for a = y sqrt(2w) <= a_max: log-spaced below
    1 (step in ln u <= 0.3/a_max), uniform above (<= 0.125 and <= 0.6/a_max); trapezoid weights."""
    a = max(float(a_max), 1.0)
    n_log = int(np.ceil(np.log(1e14) / min(0.05, 0.3 / a)))
    du = min(0.125, 0.6 / a)
    U = np.concatenate([np.geomspace(1e-14, 1.0, n_log, endpoint=False), np.arange(1.0, U_MAX, du)])
    return U, np.gradient(U)


def born_delta(w, y, kappa_s, c200=None, a_max=None, chunk=16):
    """delta(w, y) for arrays w (n_w,) at one impact parameter y. The u-integral is truncated at U_MAX
    (the tail falls off as kappa-tilde(k)/u ~ ln k/u^2, a ~1e-3 relative error)."""
    w = np.atleast_1d(np.asarray(w, float))
    U, WU = _u_grid(a_max if a_max is not None else y * np.sqrt(2 * w.max()))
    out = np.empty(w.shape, complex)
    ph = (np.exp(-1j * U) - 1) / U * WU
    for i in range(0, len(w), chunk):
        wi = w[i:i + chunk, None]
        k = np.sqrt(2 * wi * U)
        integ = j0(k * y) * kappa_tilde(k, kappa_s, c200) * ph
        out[i:i + chunk] = 1j * w[i:i + chunk] / (2 * np.pi) * integ.sum(axis=1)
    return out
