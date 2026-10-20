#!/usr/bin/env python
"""Fisher grid of the CIS core uncertainty for the LIGO inference benchmark, used to calibrate Fisher forecasts
against the mass-scan posteriors (notebooks/04_core_size_calibration.ipynb).

Source: the PE benchmark (30+30 Msun at d_L = 500 Mpc, z_S = 0.108), full IMRPhenomD amplitude (pycbc) on 20-1024 Hz,
aLIGO O4 design PSD, amplitude rescaled so the unlensed SNR equals the PE network SNR (31.7). With `LISA`, an
equal-mass binary (--m-bbh) at z_S = 3, last month of inspiral, at its native SNR.

Grid: x_c in {0.002 ... 0.45} (12 values), M_Lz log-spaced (15 values, 1-2000 Msun for LIGO), y in {0.1, 0.3, 0.6, 1, 2}.
F and its derivatives come from infer_with_fiona.fisher.compute_F_and_derivatives (converged `cis` schedule, bGO
blend above the measured switch, first-image time reference). Points that would need FIONA above w = 254 are
left nan. Two Fisher matrices per point, from the same derivatives:
  5 parameters {ln M_Lz, y, x_c, ln D_L, phi_0}         -> sigma_logM, sigma_y, sigma_xc, cond, cov
  6 parameters, adding the arrival time t_c (as in PE)  -> sigma_logM_t, sigma_y_t, sigma_xc_t, cond_t, cov_t
cov and cov_t are the covariances of (ln M_Lz, y, x_c), arrays [n_xc, n_M, n_y, 3, 3].

  python scripts/1_fisher/core_size_fisher.py LIGO      # -> data/core_fisher_LIGO_CIS.npz (shipped)
"""
import argparse
import os
import time

import finufft  # noqa: F401 -- must precede any fiona import
import numpy as np

from infer_with_fiona import paths, quadrature
quadrature.get_schedule('cis').apply()
from infer_with_fiona import dm_conversions as D
from infer_with_fiona.fisher import (compute_F_and_derivatives, f_1month, f_isco, imrphenomd_amplitude,
                                     imrphenomd_pycbc_amplitude, ligo_o4_psd, lisa_psd, luminosity_distance)
from infer_with_fiona.fisher.fisher import _inner_product

ap = argparse.ArgumentParser()
ap.add_argument('det', choices=['LIGO', 'LISA'])
ap.add_argument('--smoke', action='store_true', help='one x_c, three masses')
ap.add_argument('--n-f', type=int, default=512)
ap.add_argument('--m-bbh', type=float, default=1.0e6, help='LISA component mass [Msun]')
ap.add_argument('--out', default=None, help='output file (default: data/core_fisher_LIGO_CIS.npz for LIGO)')
args = ap.parse_args()

N_WORKERS = int(os.environ.get('SLURM_CPUS_ON_NODE', 8))
YS = np.array([0.1, 0.3, 0.6, 1.0, 2.0])
XCS = np.array([0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.45])
EPS = 5e-6
SNR_PE = 31.7                                          # unlensed H1+L1 SNR of the PE benchmark

if args.det == 'LIGO':
    m1 = m2 = 30.0
    D_L = 500.0
    z_s = D.z_source(D_L)
    MASSES = np.logspace(0, np.log10(2000.0), 15)
    f_vec = np.logspace(np.log10(20.0), np.log10(1024.0), args.n_f)
    h0 = imrphenomd_pycbc_amplitude(f_vec, m1 / (1 + z_s), m2 / (1 + z_s), z_s, D_L, 20.0, 1024.0, delta_f=1.0 / 64)
    S_n = ligo_o4_psd(f_vec)
else:
    m1 = m2 = args.m_bbh
    z_s = 3.0
    D_L = luminosity_distance(z_s)
    MASSES = np.logspace(3, 9, 19)
    M_tot, eta = m1 + m2, 0.25
    f_lo, f_hi = max(f_1month(M_tot, eta, z_s), 1e-4), min(f_isco(M_tot, z_s), 1.0)
    f_vec = np.logspace(np.log10(f_lo), np.log10(f_hi), args.n_f)
    h0 = imrphenomd_amplitude(f_vec, M_tot, eta, z_s, D_L)
    S_n = lisa_psd(f_vec)

snr_unl = np.sqrt(_inner_product(h0, h0, S_n, f_vec))
amp_scale = SNR_PE / snr_unl if args.det == 'LIGO' else 1.0
h0 = h0 * amp_scale
print(f'{args.det}: z_S = {z_s:.4f}, band {f_vec[0]:.3g}-{f_vec[-1]:.3g} Hz, unlensed SNR {snr_unl:.2f} '
      f'-> amplitude x{amp_scale:.3g}', flush=True)
if args.smoke:
    XCS, MASSES = XCS[[3]], MASSES[[0, len(MASSES) // 2, -1]]


def sigmas(der):
    Fm = np.array([[_inner_product(a, b, S_n, f_vec) for b in der] for a in der])
    cond = np.linalg.cond(Fm)
    C = np.linalg.pinv(Fm) if cond > 1e12 else np.linalg.inv(Fm)
    return np.sqrt(np.abs(np.diag(C)[:3])), cond, C[:3, :3]


KEYS = ('sigma_logM', 'sigma_y', 'sigma_xc', 'cond', 'sigma_logM_t', 'sigma_y_t', 'sigma_xc_t', 'cond_t', 'SNR')
shape = (len(XCS), len(MASSES), len(YS))
out = {k: np.full(shape, np.nan) for k in KEYS}
out['cov'] = np.full(shape + (3, 3), np.nan)        # covariance of (ln M_Lz, y, x_c), others marginalized
out['cov_t'] = np.full(shape + (3, 3), np.nan)
t_all = time.time()
for i, xc in enumerate(XCS):
    for j, M in enumerate(MASSES):
        t0 = time.time()
        try:
            F, dL, dy, dx = compute_F_and_derivatives(M, YS, xc, f_vec, lens_model='CIS', n_workers=N_WORKERS,
                                                      eps_logM=EPS, eps_y=EPS, eps_xc=EPS)
        except ValueError as err:
            print(f'x_c {xc:<6g} M_Lz {M:9.3g}: refused ({str(err)[:70]}...)', flush=True)
            continue
        for k in range(len(YS)):
            h = F[:, k] * h0
            der = [dL[:, k] * h0, dy[:, k] * h0, dx[:, k] * h0, -h, 1j * h]
            s5, c5, out['cov'][i, j, k] = sigmas(der)
            s6, c6, out['cov_t'][i, j, k] = sigmas(der + [2j * np.pi * f_vec * h])
            vals = (*s5, c5, *s6, c6, np.sqrt(_inner_product(h, h, S_n, f_vec)))
            for key, v in zip(KEYS, vals):
                out[key][i, j, k] = v
        print(f'x_c {xc:<6g} M_Lz {M:9.3g}: sigma_xc(y) = ' + ' '.join(f'{v:.3g}' for v in out['sigma_xc'][i, j])
              + ' | with t_c ' + ' '.join(f'{v:.3g}' for v in out['sigma_xc_t'][i, j]) + f'   [{time.time() - t0:.1f} s]',
              flush=True)

tag = args.det if args.det == 'LIGO' else f'LISA_mbbh{args.m_bbh:.0e}'.replace('+0', '')
fn = args.out or (str(paths.CORE_FISHER_GRID) if args.det == 'LIGO' and not args.smoke
                  else str(paths.OUTPUTS_DIR / f'core_fisher_{tag}_CIS{"_smoke" if args.smoke else ""}.npz'))
os.makedirs(os.path.dirname(fn) or '.', exist_ok=True)
np.savez(fn, lens='CIS', M_Lz=MASSES, y=YS, x_c=XCS, z_s=z_s, D_L=D_L, m1=m1, m2=m2, snr_unlensed=snr_unl,
         amp_scale=amp_scale, **out)
print(f'wrote {fn} in {time.time() - t_all:.0f} s')
