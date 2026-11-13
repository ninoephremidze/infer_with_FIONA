"""FIONA F(w) for NFW halos on the CDM c-M relation (Diemer & Joyce 2019) for a LISA source, on a grid of lens
redshifts and impact parameters, saving F and its lens-parameter derivatives so that the post-processing
(fiona_cdm_rate_post.py) can (i) marginalize over the binary's intrinsic parameters and (ii) integrate the
detection cross-section over the halo mass function (optical depth).

Units as in fiona_cdm.py: lengths in units of the fiducial halo's Fresnel radius xi_0 = r_s/sqrt(w_s(f_*)), so
w' = f/f_* for every halo, one FIONA call per (z_L, M200) gives F at all b and all finite-difference neighbours.
FIONA's untruncated NFW (stable small-u form, nfw_stable.py). The phase exp(-i w' psi'(y')) is removed
analytically (a time shift).

Saved per (z_L, M200, b): Delta chi^2 maximized over amplitude, phase, time; F0 (n_f); dF (3, n_f) for
(ln M200, ln c200, ln b) with step EPS[0]; the 6x6 Fisher matrices for both steps (lens + ln A, phi_0, t_c).

  python fiona_cdm_rate.py SMBHB_1e4 [--sched steep_core] [--zidx 0,1] [--midx 3,4]
"""
import finufft  # noqa: F401 -- must precede any fiona import
import argparse, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('name')
ap.add_argument('--sched', default='steep_core')
ap.add_argument('--zidx', default=None)
ap.add_argument('--midx', default=None)
ap.add_argument('--out', default=None)
A = ap.parse_args()
sys.argv = [sys.argv[0], 'LISA', A.name]

from infer_with_fiona import paths, quadrature
quadrature.get_schedule(A.sched).apply()
from infer_with_fiona.fiona import utils as _fu
_fu._FIONA_GL2D_DIR = str(paths.GL_RULES_DIR)
from infer_with_fiona.fiona.axisym import FresnelNUFHTLensBatched

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scan_detect as SD
import halo_born as H
from nfw_stable import NFWStable
from colossus.cosmology import cosmology
from colossus.halo import concentration
from astropy.cosmology import Planck18 as COSMO, z_at_value
import astropy.units as u
cosmology.setCosmology('planck18')

NW = int(os.environ.get('SLURM_CPUS_ON_NODE', 8))
EPS = (1e-2, 3e-3)
B_RF = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
XFRAC = np.array([0.1, 0.3, 0.5, 0.7, 0.9])                 # chi_L / chi_S
f, WGT, F_STAR, z_s = SD.f, SD.WGT, SD.F_STAR, SD.z_s
WP = f / F_STAR
M200 = SD.M200
CHI_S = COSMO.comoving_distance(z_s)
ZL = np.array([float(z_at_value(COSMO.comoving_distance, x * CHI_S)) for x in XFRAC])
ZI = range(len(ZL)) if A.zidx is None else [int(i) for i in A.zidx.split(',')]
MI = range(len(M200)) if A.midx is None else [int(i) for i in A.midx.split(',')]


def lens_of(M, c, zl, xi0_pc):
    h = H.halo(M, c, zl, z_s)
    r = h['rs_pc'] / xi0_pc
    return NFWStable(psi0=h['kappa'] * r * r, xs=r)


def dchi2_unlensed(F):
    """rho_L^2 (1 - match^2) of F h0 against h0, match maximized over amplitude, phase, time."""
    X = F * WGT
    span = 2.0 / F_STAR
    ts = np.linspace(-span, span, 801)
    m = lambda t: np.abs(np.exp(2j * np.pi * np.outer(t, f)) @ X)
    v = m(ts); t0, half = ts[np.argmax(v)], ts[1] - ts[0]
    for _ in range(3):
        ts = np.linspace(t0 - half, t0 + half, 41); v = m(ts); t0, half = ts[np.argmax(v)], half / 15
    nL = np.sum(np.abs(F) ** 2 * WGT)
    return nL * (1 - min(v.max() / np.sqrt(nL * np.sum(WGT)), 1.0) ** 2), t0


def fisher_mat(F0, dF):
    der = list(dF) + [F0, 1j * F0, 2j * np.pi * f * F0]
    return np.array([[np.sum(np.real(np.conj(a) * b) * WGT) for b in der] for a in der])


nz, nm, nb, ne, nf = len(ZL), len(M200), len(B_RF), len(EPS), len(f)
out = dict(name=A.name, M200=M200, z_L=ZL, xfrac=XFRAC, b_rF=B_RF, f=f, wgt=WGT, f_star=F_STAR, z_s=z_s,
           snr=SD.SNR, eps=np.array(EPS), c200=np.full((nz, nm), np.nan), kappa=np.full((nz, nm), np.nan),
           rF_pc=np.full((nz, nm), np.nan), dchi2=np.full((nz, nm, nb), np.nan), tbest=np.full((nz, nm, nb), np.nan),
           F0=np.full((nz, nm, nb, nf), np.nan, complex), dF=np.full((nz, nm, nb, 3, nf), np.nan, complex),
           fish=np.full((nz, nm, nb, ne, 6, 6), np.nan), done=np.zeros((nz, nm), bool))
out_path = A.out or os.path.join(os.environ.get('CDM_RAW', str(paths.OUTPUTS_DIR / 'cdm')), f'fiona_cdm_rate_{A.name}_{A.sched}.npz')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
print(f'LISA {A.name}: {nf} freqs, f* {F_STAR:.3g} Hz, z_L {np.round(ZL, 3)}, sched {A.sched}, {NW} workers',
      flush=True)
t_all = time.time()
for iz in ZI:
    zl = ZL[iz]
    CC = np.array([concentration.concentration(m * COSMO.h, '200c', zl, model='diemer19') for m in M200])
    for i in MI:
        t0 = time.time()
        M, c = M200[i], CC[i]
        h0 = H.halo(M, c, zl, z_s)
        xi0 = h0['rs_pc'] / np.sqrt(H.w_of_f(h0['M_Lz'], F_STAR))
        out['c200'][iz, i], out['kappa'][iz, i], out['rF_pc'][iz, i] = c, h0['kappa'], xi0
        lenses = [lens_of(M, c, zl, xi0)]
        for e in EPS:
            lenses += [lens_of(M * np.exp(e), c, zl, xi0), lens_of(M * np.exp(-e), c, zl, xi0),
                       lens_of(M, c * np.exp(e), zl, xi0), lens_of(M, c * np.exp(-e), zl, xi0)]
        ys = np.array([bb * s for bb in B_RF for s in [1.0] + [np.exp(sg * e) for e in EPS for sg in (1, -1)]])
        solver = FresnelNUFHTLensBatched(lenses, gl_dir=str(paths.GL_RULES_DIR), n_workers=NW, print_timing=False)
        raw = solver(WP, ys)
        psi_y = np.array([L.psi_r(ys) for L in lenses])
        Fall = raw * np.exp(1j * WP[:, None, None] * psi_y[None])
        nyb = 1 + 2 * ne
        for k in range(nb):
            jb = k * nyb
            F0 = Fall[:, 0, jb]
            out['dchi2'][iz, i, k], out['tbest'][iz, i, k] = dchi2_unlensed(F0)
            out['F0'][iz, i, k] = F0
            for ie, e in enumerate(EPS):
                l0 = 1 + 4 * ie
                dF = ((Fall[:, l0, jb] - Fall[:, l0 + 1, jb]) / (2 * e),
                      (Fall[:, l0 + 2, jb] - Fall[:, l0 + 3, jb]) / (2 * e),
                      (Fall[:, 0, jb + 1 + 2 * ie] - Fall[:, 0, jb + 2 + 2 * ie]) / (2 * e))
                out['fish'][iz, i, k, ie] = fisher_mat(F0, dF)
                if ie == 0:
                    out['dF'][iz, i, k] = np.array(dF)
        out['done'][iz, i] = True
        print(f'z_L {zl:.3f} M200 {M:.2e} c {c:.1f} kappa {h0["kappa"]:.2e} r_F {xi0:.3g} pc | dchi2(b/r_F='
              f'0.3,1,3,10) ' + ' '.join(f'{out["dchi2"][iz, i, k]:.3g}' for k in (2, 5, 8, 11))
              + f'  [{time.time() - t0:.0f} s]', flush=True)
    np.savez(out_path, **out)
np.savez(out_path, **out)
print(f'wrote {out_path} in {time.time() - t_all:.0f} s', flush=True)
