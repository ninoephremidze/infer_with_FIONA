#!/usr/bin/env python
"""Detectability of point-mass (PBH) lensing for the LIGO inference benchmark, as a function of the redshifted
lens mass and the source position, for the PBH optical-depth forecast of notebooks/04_core_size_calibration.ipynb.

Source: the PE benchmark (30+30 Msun at d_L = 500 Mpc), full IMRPhenomD amplitude (pycbc) on 20-1024 Hz with
df = 1/4 Hz (the 4 s segment of the inference), aLIGO O4 design PSD, amplitude rescaled so the unlensed SNR equals
the PE network SNR (31.7). For each (M_Lz, y), F comes from FIONA's softened point lens (steep_core schedule,
w <= 254, sorted y), and the lensed signal h_L = F h_0 is compared with the best unlensed template, i.e. h_0
with free amplitude, phase and arrival time:

  match = max_t |<h_L, h_0 e^{2 pi i f t}>| / (|h_L| |h_0|),    Delta chi^2 = rho_L^2 (1 - match^2),

the chi^2 by which the unlensed hypothesis fits worse. Writes data/pbh_detectability_LIGO.npz with the grids and
Delta chi^2 [n_M, n_y].

  python scripts/1_fisher/pbh_detectability.py      # about 1-2 hours on 112 cores (slurm/pbh_detectability.sh)

Results are saved after every mass and a rerun resumes from the partial file.
"""
import os
import time

import finufft  # noqa: F401 -- must precede any fiona import
import numpy as np

from infer_with_fiona import paths, quadrature, table
quadrature.get_schedule('steep_core').apply()
from infer_with_fiona import dm_conversions as D
from infer_with_fiona.fiona.axisym import FresnelNUFHTLensBatched
from infer_with_fiona.fisher import imrphenomd_pycbc_amplitude, ligo_o4_psd
from infer_with_fiona.fisher.fisher import _inner_product
from infer_with_fiona.fisher.utils import G_over_c3_Msun

N_WORKERS = int(os.environ.get('SLURM_CPUS_ON_NODE', 8))
SNR_PE = 31.7
DF, F_LO, F_HI = 0.25, 20.0, 1024.0
MASSES = np.geomspace(1.0, 2000.0, 34)
YS = np.geomspace(0.05, 6.0, 60)                      # ascending: the NUFHT needs sorted output radii
MTOW = 8 * np.pi * G_over_c3_Msun

z_s = D.z_source(500.0)
f = np.arange(F_LO, F_HI + DF / 2, DF)
h0 = imrphenomd_pycbc_amplitude(f, 30.0 / (1 + z_s), 30.0 / (1 + z_s), z_s, 500.0, F_LO, F_HI, delta_f=DF)
S = ligo_o4_psd(f)
h0 = h0 * SNR_PE / np.sqrt(_inner_product(h0, h0, S, f))
w_unit = h0 ** 2 / S                                  # noise weight |h_0|^2 / S_n per frequency bin
n_fft = 1 << int(np.ceil(np.log2(len(f) * 8)))        # zero padding: time resolution 1/(n_fft df)

DT = 1.0 / (n_fft * DF)                                  # spacing of the FFT time grid
J = np.arange(len(f))


def refine_peak(X, k0):
    """max_t |sum_j X_j e^{2 pi i j df t}| near the FFT peak k0, by two passes of direct sums on finer t grids
    (the grid maximum alone leaves a mismatch floor of Delta chi^2 ~ 0.3 at this SNR)."""
    t_best, half = k0 * DT, DT
    for _ in range(2):
        ts = t_best + np.linspace(-half, half, 41)
        vals = np.abs(np.exp(2j * np.pi * DF * np.outer(ts, J)) @ X)
        t_best, half = ts[np.argmax(vals)], half / 20
    return vals.max()


lens = [table.make_lens('PointLens', table.POINT_LENS_SOFTENING)]
solver = FresnelNUFHTLensBatched(lens, gl_dir=str(paths.GL_RULES_DIR), n_workers=N_WORKERS, print_timing=False)
OUT = paths.DATA_DIR / 'pbh_detectability_LIGO.npz'
dchi2 = np.full((len(MASSES), len(YS)), np.nan)
rho_L = np.full_like(dchi2, np.nan)
if OUT.exists():
    old = np.load(OUT)
    if np.array_equal(old['M_Lz'], MASSES) and np.array_equal(old['y'], YS):
        dchi2, rho_L = old['dchi2'].copy(), old['rho_L'].copy()
t0 = time.time()
for i, M in enumerate(MASSES):
    if np.isfinite(dchi2[i]).all():
        continue
    F = solver(MTOW * M * f, YS)[:, 0, :]                       # (n_f, n_y)
    for k in range(len(YS)):
        norm_L = 4 * DF * np.sum(np.abs(F[:, k]) ** 2 * w_unit)
        norm_0 = 4 * DF * np.sum(w_unit)
        X = F[:, k] * w_unit
        corr = np.fft.ifft(X, n=n_fft) * n_fft                  # <h_L, h_0 e^{2 pi i f t}> on a t grid (up to 4 df)
        match = 4 * DF * refine_peak(X, np.argmax(np.abs(corr))) / np.sqrt(norm_L * norm_0)
        rho_L[i, k] = np.sqrt(norm_L)
        dchi2[i, k] = norm_L * (1 - min(match, 1.0) ** 2)
    print(f'M_Lz {M:8.3g}: Delta chi^2(y = {YS[0]:.2g}, 0.3, 1, 3) = '
          + ' '.join(f'{np.interp(yy, YS, dchi2[i]):.3g}' for yy in (YS[0], 0.3, 1.0, 3.0))
          + f'   [{time.time() - t0:.0f} s]', flush=True)
    np.savez(OUT, M_Lz=MASSES, y=YS, dchi2=dchi2, rho_L=rho_L, z_s=z_s, snr_unlensed=SNR_PE,
             f_band=(F_LO, F_HI), df=DF)
print('wrote', OUT)
