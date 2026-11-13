"""Detectability of NFW halo lensing in the (M200, c200) plane, for LIGO or LISA.

For each halo (M200, c200) at the midpoint lens redshift (half the source's comoving distance) and each
impact parameter b, given in units of the Fresnel radius r_F(f_*) = r_s / sqrt(w(f_*)) at the source's
reference frequency f_* (90% of the unlensed SNR^2 below it), this computes

  Delta chi^2 = rho_L^2 (1 - match^2),  match maximized over amplitude, phase and arrival time
              (the criterion of the paper's PBH forecast, Sec. dm_constraints), and
  Fisher errors on (ln M200, ln c200, ln b), marginalized over ln A, phi_0 and t_c, at fixed physical b.

F comes from the Born approximation (halo_born.py, NFW truncated at r_200) where kappa = 4 kappa_s <=
KAPPA_BORN. Points above it are left as NaN and flagged (born=False) for a FIONA pass.

  python scan_detect.py LIGO BBH_30   |   python scan_detect.py LISA SMBHB_1e5
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18 as COSMO, z_at_value

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import halo_born as H
from infer_with_fiona.fisher import (imrphenomd_pycbc_amplitude, ligo_o4_psd, lisa_psd, f_isco, f_1month,
                                     luminosity_distance)
from infer_with_fiona.fisher.fisher import _inner_product

KAPPA_BORN = float(os.environ.get('KAPPA_BORN', 0.1))
B_RF = np.array([0.3, 1.0, 3.0])
C200 = np.geomspace(3.0, 1e5, 31)
SOURCES = {   # det, name: (m1, m2, z_s, SNR, M200 grid)
    ('LIGO', 'BBH_30'): (30.0, 30.0, None, 31.7, np.geomspace(1.0, 1e8, 33)),
    ('LISA', 'SMBHB_1e4'): (1e4, 1e4, 3.0, 1e3, np.geomspace(1e3, 1e12, 37)),
    ('LISA', 'SMBHB_1e5'): (1e5, 1e5, 3.0, 1e3, np.geomspace(1e3, 1e12, 37)),
    ('LISA', 'SMBHB_1e6'): (1e6, 1e6, 3.0, 1e3, np.geomspace(1e3, 1e12, 37)),
    ('LISA', 'SMBHB_1e7'): (1e7, 1e7, 3.0, 1e3, np.geomspace(1e3, 1e14, 45)),   # heavier source, halos to 1e14
}
DET, NAME = sys.argv[1], sys.argv[2]
m1, m2, z_s, SNR, M200 = SOURCES[(DET, NAME)]

if DET == 'LIGO':     # the PE benchmark: 30+30 Msun at d_L = 500 Mpc, 20-1024 Hz, O4 PSD (as pbh_detectability.py)
    z_s = float(z_at_value(COSMO.luminosity_distance, 500 * u.Mpc))
    f = np.arange(20.0, 1024.0 + 0.125, 0.25)
    h0 = imrphenomd_pycbc_amplitude(f, m1 / (1 + z_s), m2 / (1 + z_s), z_s, 500.0, 20.0, 1024.0, delta_f=0.25)
    S = ligo_o4_psd(f)
else:                 # the paper's LISA sources: last month before merger at z_s = 3
    M_tot, eta = m1 + m2, m1 * m2 / (m1 + m2) ** 2
    f_lo, f_hi = max(f_1month(M_tot, eta, z_s), 1e-4), min(f_isco(M_tot, z_s), 1.0)
    if os.environ.get('LISA_BAND') == 'imr':           # full IMRPhenomD band: up to its cutoff M f = 0.2 (detector frame)
        f_hi = min(0.2 / (M_tot * (1 + z_s) * 4.925491e-6), 1.0)
    f = np.geomspace(f_lo, f_hi, 768)
    h0 = imrphenomd_pycbc_amplitude(f, m1, m2, z_s, luminosity_distance(z_s), f_lo, f_hi, delta_f=1e-7)
    S = lisa_psd(f)
keep = h0 > 0
f, h0, S = f[keep], h0[keep], S[keep]
SNR_NATIVE = float(np.sqrt(_inner_product(h0, h0, S, f)))
if os.environ.get('NATIVE_SNR') and DET == 'LISA':   # keep the source at its native SNR
    SNR = SNR_NATIVE
h0 = h0 * SNR / SNR_NATIVE
dens = h0 ** 2 / S
cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(f))])
F_STAR = float(np.interp(0.9 * cum[-1], cum, f))
Z_L = float(z_at_value(COSMO.comoving_distance, COSMO.comoving_distance(z_s) / 2))
WGT = 4 * np.gradient(f) * dens                       # trapezoid-like weights for <a h0, b h0>


def F_born(M, c, b_pc):
    h = H.halo(M, c, Z_L, z_s)
    w = H.w_of_f(h['M_Lz'], f)
    y = b_pc / h['rs_pc']
    a_max = y * np.sqrt(2 * w.max())
    return 1 + H.born_delta(w, y, h['kappa_s'], c, a_max=a_max)


def best_match(F):
    """max over t (and amplitude, phase) of |<F h0, h0 e^{2 pi i f t}>| / (|F h0| |h0|)."""
    X = F * WGT
    def m(ts):
        return np.abs(np.exp(2j * np.pi * np.outer(ts, f)) @ X)
    span = 2.0 / F_STAR
    ts = np.linspace(-span, span, 801)
    v = m(ts)
    t0, half = ts[np.argmax(v)], ts[1] - ts[0]
    for _ in range(3):
        ts = np.linspace(t0 - half, t0 + half, 41)
        v = m(ts)
        t0, half = ts[np.argmax(v)], half / 15
    nL = np.sum(np.abs(F) ** 2 * WGT)
    return v.max() / np.sqrt(nL * np.sum(WGT)), nL


def point(args):
    i, j = args
    M, c = M200[i], C200[j]
    h = H.halo(M, c, Z_L, z_s)
    res = dict(i=i, j=j, kappa=h['kappa'], M_Lz=h['M_Lz'], w_star=H.w_of_f(h['M_Lz'], F_STAR),
               born=h['kappa'] <= KAPPA_BORN, dchi2=np.full(len(B_RF), np.nan),
               sig=np.full((len(B_RF), 3), np.nan), snrL=np.full(len(B_RF), np.nan))
    if not res['born']:
        return res
    rF_pc = h['rs_pc'] / np.sqrt(res['w_star'])
    for k, bb in enumerate(B_RF):
        b_pc = bb * rF_pc
        F0 = F_born(M, c, b_pc)
        match, nL = best_match(F0)
        res['dchi2'][k] = nL * (1 - min(match, 1.0) ** 2)
        res['snrL'][k] = np.sqrt(nL)
        eps = 1e-3                                       # central differences in ln M200, ln c200, ln b
        dF = [(F_born(M * np.exp(eps), c, b_pc) - F_born(M * np.exp(-eps), c, b_pc)) / (2 * eps),
              (F_born(M, c * np.exp(eps), b_pc) - F_born(M, c * np.exp(-eps), b_pc)) / (2 * eps),
              (F_born(M, c, b_pc * np.exp(eps)) - F_born(M, c, b_pc * np.exp(-eps))) / (2 * eps)]
        der = dF + [F0, 1j * F0, 2j * np.pi * f * F0]      # ln A, phi_0, t_c
        Fish = np.array([[np.sum(np.real(np.conj(a) * b) * WGT) for b in der] for a in der])
        try:
            cov = np.linalg.inv(Fish)
            res['sig'][k] = np.sqrt(np.abs(np.diag(cov)[:3]))
        except np.linalg.LinAlgError:
            pass
    return res


if __name__ == '__main__':
    out_path = f'scan_{DET}_{NAME}.npz'
    print(f'{DET} {NAME}: z_s {z_s:.3f}, z_L {Z_L:.3f}, f {f[0]:.3g}-{f[-1]:.3g} Hz ({len(f)} bins), '
          f'f_* {F_STAR:.3g} Hz, SNR {SNR:g}, kappa_Born <= {KAPPA_BORN}', flush=True)
    shape = (len(M200), len(C200))
    D = dict(dchi2=np.full(shape + (len(B_RF),), np.nan), sig=np.full(shape + (len(B_RF), 3), np.nan),
             snrL=np.full(shape + (len(B_RF),), np.nan), kappa=np.full(shape, np.nan),
             M_Lz=np.full(shape, np.nan), w_star=np.full(shape, np.nan), born=np.zeros(shape, bool))
    t0 = time.time()
    tasks = [(i, j) for i in range(shape[0]) for j in range(shape[1])]
    with Pool(int(os.environ.get('SLURM_CPUS_ON_NODE', 8))) as pool:
        for n, r in enumerate(pool.imap_unordered(point, tasks, chunksize=1)):
            i, j = r['i'], r['j']
            for k in ('dchi2', 'sig', 'snrL', 'kappa', 'M_Lz', 'w_star', 'born'):
                D[k][i, j] = r[k]
            if (n + 1) % 100 == 0 or n + 1 == len(tasks):
                print(f'  {n + 1}/{len(tasks)} points, {time.time() - t0:.0f} s', flush=True)
                np.savez(out_path, M200=M200, c200=C200, b_rF=B_RF, f_star=F_STAR, z_L=Z_L, z_s=z_s, snr=SNR,
                         kappa_born=KAPPA_BORN, **D)
    print('wrote', out_path)
