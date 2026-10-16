"""
infer_with_fiona.fisher (formerly infer_with_fiona.fisher) — Fisher forecasts for gravitational wave lensing with LISA.

Uses FIONA's Fresnel integral solver (FresnelNUFHTBatched) to compute the
lensing amplification factor F(w, y) and its derivatives, then assembles
the Fisher matrix for lens parameter constraints.

Quick start
-----------
>>> from infer_with_fiona.fisher import run_fisher
>>> results = run_fisher(
...     M_Lz_Msun=1e7,
...     y=0.3,
...     x_c=0.01,
...     m1_Msun=5e5,
...     m2_Msun=5e5,
...     z_s=3.0,
... )
>>> print(results['sigma_logM'])
"""
from .run import run_fisher, run_fisher_gSIS, run_fisher_NFW, run_fisher_NFW_c200, run_fisher_ULA, run_fisher_PointLens
from .waveform import imrphenomd_amplitude, imrphenomd_pycbc_amplitude, f_isco, f_1month
from .noise import lisa_psd, ligo_psd, ligo_o3_psd, ligo_o4_psd, ligo_aplus_psd
from .lensing import compute_F, compute_F_3p, compute_F_and_derivatives, compute_F_and_derivatives_3p, compute_F_param_batch, gSIS
from .fisher import assemble_fisher_matrix, marginalized_errors, lsa_validity
from .utils import G_over_c3_Msun, luminosity_distance, c200_to_xs, xi_E_kpc, rs_to_axion_mass
from .cis_go import cis_images, cis_image_props, cis_F_bGO, cis_dF_bGO

__all__ = [
    'run_fisher',
    'run_fisher_gSIS',
    'run_fisher_NFW',
    'run_fisher_NFW_c200',
    'run_fisher_ULA',
    'run_fisher_PointLens',
    'imrphenomd_amplitude',
    'imrphenomd_pycbc_amplitude',
    'f_isco',
    'f_1month',
    'lisa_psd',
    'ligo_psd',
    'ligo_o3_psd',
    'ligo_o4_psd',
    'ligo_aplus_psd',
    'compute_F',
    'compute_F_3p',
    'compute_F_and_derivatives',
    'compute_F_and_derivatives_3p',
    'compute_F_param_batch',
    'gSIS',
    'assemble_fisher_matrix',
    'marginalized_errors',
    'lsa_validity',
    'G_over_c3_Msun',
    'luminosity_distance',
    'c200_to_xs',
    'xi_E_kpc',
    'rs_to_axion_mass',
    'cis_images',
    'cis_image_props',
    'cis_F_bGO',
    'cis_dF_bGO',
]

__version__ = '0.1.0'
