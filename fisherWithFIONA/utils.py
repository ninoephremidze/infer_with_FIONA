"""
Physical constants and cosmological helpers for FisherWithFIONA.
"""
import numpy as np
from scipy.integrate import quad

# SI constants
G_SI = 6.674e-11           # m^3 kg^-1 s^-2
c_SI = 2.998e8             # m/s
M_sun_kg = 1.989e30        # kg
Mpc_m = 3.085677581e22     # m / Mpc

# Conversion: solar mass → seconds  (G M_sun / c^3)
G_over_c3_Msun = G_SI * M_sun_kg / c_SI**3  # ~4.9255e-6 s / M_sun


def _comoving_distance_m(z, H0=70.0, Om0=0.3, Ode0=0.7):
    """Comoving distance to redshift z [metres], flat ΛCDM."""
    H0_SI = H0 * 1e3 / Mpc_m  # s^-1
    def E(zp):
        return np.sqrt(Om0 * (1 + zp)**3 + Ode0)
    val, _ = quad(lambda zp: 1.0 / E(zp), 0, z)
    return c_SI / H0_SI * val  # metres


def xi_E_kpc(M_Lz_Msun, z_L=1.0, z_S=3.0, H0=70.0, Om0=0.3, Ode0=0.7):
    """
    Einstein-radius length scale ξ_E for a point lens of mass M_Lz, in kpc.

        ξ_E = sqrt( 4 G M_Lz / c² × D_L D_LS / D_S )

    where D_L, D_S, D_LS are angular-diameter distances (flat ΛCDM).
    This is the physical scale that converts a dimensionless lens-plane
    coordinate (e.g. the ULA soliton core radius r_s or NFW scale radius xs)
    into a physical length: r_phys = r_dimensionless × ξ_E.

    Parameters
    ----------
    M_Lz_Msun : float   Effective lens mass [M_sun].
    z_L        : float   Lens redshift (default 1.0).
    z_S        : float   Source redshift (default 3.0).
    H0, Om0, Ode0 : floats  Cosmology.

    Returns
    -------
    float  ξ_E in kpc.
    """
    chi_L = _comoving_distance_m(z_L, H0, Om0, Ode0)
    chi_S = _comoving_distance_m(z_S, H0, Om0, Ode0)

    D_L  = chi_L / (1.0 + z_L)            # angular diam. dist. lens   [m]
    D_S  = chi_S / (1.0 + z_S)            # angular diam. dist. source  [m]
    D_LS = (chi_S - chi_L) / (1.0 + z_S)  # angular diam. dist. L→S    [m]

    xi_E_m = np.sqrt(4.0 * G_SI * M_Lz_Msun * M_sun_kg / c_SI**2
                     * D_L * D_LS / D_S)
    kpc_m = Mpc_m / 1.0e3
    return xi_E_m / kpc_m


def rs_to_axion_mass(r_s, M_Lz_Msun, M_halo_Msun, z_L=1.0, z_S=3.0,
                     H0=70.0, Om0=0.3, Ode0=0.7):
    """
    Convert a dimensionless ULA soliton core radius r_s (in Einstein-radius
    units) to an ultralight axion mass, via the Schive+2014 soliton-halo
    core-mass relation

        r_c [kpc] = 1.6 × (m_a / 1e-22 eV)^-1 × (M_halo / 1e9 M_sun)^-1/3

    Solving for the axion mass:

        m_a / 1e-22 eV = 1.6 / r_c[kpc] × (M_halo / 1e9 M_sun)^-1/3

    with the physical core radius r_c = r_s × ξ_E(M_Lz, z_L, z_S).

    Parameters
    ----------
    r_s : float or array_like
        Dimensionless soliton core radius (Fisher parameter).
    M_Lz_Msun : float or array_like
        Effective lens mass [M_sun], sets ξ_E.
    M_halo_Msun : float or array_like
        Host (sub)halo virial mass [M_sun].
    z_L, z_S : float
        Lens/source redshifts for the ξ_E distance calculation.
    H0, Om0, Ode0 : floats
        Cosmology.

    Returns
    -------
    float or ndarray
        Axion mass m_a in units of 1e-22 eV, broadcasting over the inputs.
    """
    r_s   = np.asarray(r_s, dtype=float)
    M_Lz  = np.asarray(M_Lz_Msun, dtype=float)
    M_h   = np.asarray(M_halo_Msun, dtype=float)

    xi_E_val = xi_E_kpc(M_Lz, z_L=z_L, z_S=z_S, H0=H0, Om0=Om0, Ode0=Ode0)
    r_c_kpc  = r_s * xi_E_val

    return 1.6 / r_c_kpc * (M_h / 1.0e9) ** (-1.0 / 3.0)


def c200_to_xs(M_Lz_Msun, c200, z_L=1.0, z_S=3.0, H0=70.0, Om0=0.3, Ode0=0.7):
    """
    Convert NFW concentration c200 → dimensionless scale radius xs = r_s / ξ_E.

    The Einstein-radius scale for a point lens of mass M_Lz is

        ξ_E = sqrt( 4 G M_Lz / c² × D_L D_LS / D_S )

    where D_L, D_S, D_LS are angular-diameter distances (flat ΛCDM).

    The virial radius r_200 is computed from M_Lz ≈ M_200 and the critical
    density at z_L:

        r_200 = [ 3 M_200 / (4π × 200 × ρ_c(z_L)) ]^{1/3}

    and the scale radius is r_s = r_200 / c200, giving xs = r_s / ξ_E.

    Parameters
    ----------
    M_Lz_Msun : float   Effective lens mass [M_sun].
    c200       : float   Virial concentration.
    z_L        : float   Lens redshift (default 1.0).
    z_S        : float   Source redshift (default 3.0).
    H0, Om0, Ode0 : floats  Cosmology.

    Returns
    -------
    float  xs (dimensionless scale radius).
    """
    chi_L = _comoving_distance_m(z_L, H0, Om0, Ode0)
    chi_S = _comoving_distance_m(z_S, H0, Om0, Ode0)

    D_L  = chi_L / (1.0 + z_L)            # angular diam. dist. lens   [m]
    D_S  = chi_S / (1.0 + z_S)            # angular diam. dist. source  [m]
    D_LS = (chi_S - chi_L) / (1.0 + z_S)  # angular diam. dist. L→S    [m]

    # Einstein radius [m]
    xi_E = np.sqrt(4.0 * G_SI * M_Lz_Msun * M_sun_kg / c_SI**2
                   * D_L * D_LS / D_S)

    # Critical density at z_L [kg/m³]
    H0_SI = H0 * 1e3 / Mpc_m
    H_L = H0_SI * np.sqrt(Om0 * (1.0 + z_L)**3 + Ode0)
    rho_crit_L = 3.0 * H_L**2 / (8.0 * np.pi * G_SI)

    # Virial radius (assuming M_Lz ≈ M_200) [m]
    M_200_kg = M_Lz_Msun * M_sun_kg
    r_200 = (3.0 * M_200_kg / (4.0 * np.pi * 200.0 * rho_crit_L))**(1.0 / 3.0)

    return (r_200 / c200) / xi_E


def luminosity_distance(z, H0=70.0, Om0=0.3, Ode0=0.7):
    """
    Luminosity distance in Mpc for flat ΛCDM cosmology.

    Parameters
    ----------
    z : float
        Redshift.
    H0 : float
        Hubble constant [km/s/Mpc].
    Om0, Ode0 : float
        Matter and dark-energy density parameters.

    Returns
    -------
    float
        D_L in Mpc.
    """
    H0_SI = H0 * 1e3 / Mpc_m  # s^-1

    def E(zp):
        return np.sqrt(Om0 * (1 + zp)**3 + Ode0)

    comoving, _ = quad(lambda zp: 1.0 / E(zp), 0, z)
    D_H = c_SI / H0_SI  # Hubble distance in m
    D_L_m = (1 + z) * D_H * comoving
    return D_L_m / Mpc_m  # Mpc
