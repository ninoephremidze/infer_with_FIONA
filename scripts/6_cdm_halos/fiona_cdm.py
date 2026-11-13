"""Delta chi^2 and Fisher errors for NFW halos on the CDM c-M relation (Diemer & Joyce 2019), with the full
wave-optics F(w) from FIONA (FresnelNUFHTLensBatched) instead of the Born approximation.

Units. Lengths are in units of the fiducial halo's Fresnel radius xi_0 = r_F = r_s / sqrt(w_s(f_*)), where
w_s is the dimensionless frequency in r_s units. Since w scales as xi_0^2 and M_Lz as r_s^2, the dimensionless
frequency in these units is w'(f) = f / f_* for EVERY halo, so the fiducial and all finite-difference halos
share one FIONA quadrature grid. A halo (M200, c200) then has
  psi'(x') = (kappa/2) h(x' xi_0 / r_s) (r_s/xi_0)^2   ->  NFWStable(psi0 = kappa (r_s/xi_0)^2, xs = r_s/xi_0),
and impact parameter y' = b / xi_0. FIONA's NFW is untruncated, as in Eq. (nfw_psi) of the paper.
The phase exp(-i w' psi'(y')) (a time shift, absorbed by t_c) is removed analytically.

Delta chi^2 = rho_L^2 (1 - match^2) maximized over amplitude, phase, time; Fisher on (ln M200, ln c200, ln b,
ln A, phi_0, t_c), central differences with two steps (EPS) as a check. The full 6x6 matrices are saved ('fish')
for post-processing (e.g. the error on rho_s r_s, or on c200 at fixed M200). With --born, the Born F (truncated and
untruncated) is evaluated at the same points for comparison.

  python fiona_cdm.py LISA SMBHB_1e4 [--idx 10,20] [--born] [--sched steep_core] [--out file.npz]
"""
import finufft  # noqa: F401 -- must precede any fiona import
import argparse, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('det'); ap.add_argument('name')
ap.add_argument('--idx', default=None, help='comma-separated M200 grid indices (default all)')
ap.add_argument('--born', action='store_true')
ap.add_argument('--sched', default='steep_core')
ap.add_argument('--out', default=None)
A = ap.parse_args()
sys.argv = sys.argv[:3]                    # scan_detect parses argv[1:3]

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
from astropy.cosmology import Planck18 as COSMO
cosmology.setCosmology('planck18')

NW = int(os.environ.get('SLURM_CPUS_ON_NODE', 8))
EPS = (1e-2, 3e-3)
B_RF = SD.B_RF
f, WGT, F_STAR = SD.f, SD.WGT, SD.F_STAR
WP = f / F_STAR                                            # w' = f / f_*
M200 = SD.M200
CC = np.array([concentration.concentration(m * COSMO.h, '200c', SD.Z_L, model='diemer19') for m in M200])
IDX = range(len(M200)) if A.idx is None else [int(i) for i in A.idx.split(',')]


def lens_of(M, c, xi0_pc):
    h = H.halo(M, c, SD.Z_L, SD.z_s)
    r = h['rs_pc'] / xi0_pc
    return NFWStable(psi0=h['kappa'] * r * r, xs=r), h


def mismatch_dchi2(F1, F2):
    """rho^2 (1 - match^2) of F2 h0 against F1 h0, match maximized over amplitude, phase, time."""
    X = F1 * np.conj(F2) * WGT
    span = 2.0 / F_STAR
    ts = np.linspace(-span, span, 801)
    m = lambda t: np.abs(np.exp(2j * np.pi * np.outer(t, f)) @ X)
    v = m(ts); t0, half = ts[np.argmax(v)], ts[1] - ts[0]
    for _ in range(3):
        ts = np.linspace(t0 - half, t0 + half, 41); v = m(ts); t0, half = ts[np.argmax(v)], half / 15
    n1, n2 = np.sum(np.abs(F1) ** 2 * WGT), np.sum(np.abs(F2) ** 2 * WGT)
    return n2 * (1 - min(v.max() / np.sqrt(n1 * n2), 1.0) ** 2)


def fisher_mat(F0, dF):
    """6x6 Fisher matrix in (ln M200, ln c200, ln b, ln A, phi_0, t_c)."""
    der = list(dF) + [F0, 1j * F0, 2j * np.pi * f * F0]
    return np.array([[np.sum(np.real(np.conj(a) * b) * WGT) for b in der] for a in der])


def fisher_sig(F0, dF, keep=None):
    Fi = fisher_mat(F0, dF)
    if keep is not None:
        keep[...] = Fi
    try:
        return np.sqrt(np.abs(np.diag(np.linalg.inv(Fi))[:3]))
    except np.linalg.LinAlgError:
        return np.full(3, np.nan)


n, nb, ne = len(M200), len(B_RF), len(EPS)
out = dict(M200=M200, c200=CC, b_rF=B_RF, f_star=F_STAR, z_L=SD.Z_L, z_s=SD.z_s, snr=SD.SNR, eps=np.array(EPS),
           kappa=np.full(n, np.nan), w_star_rs=np.full(n, np.nan),
           dchi2=np.full((n, nb), np.nan), sig=np.full((n, nb, ne, 3), np.nan), snrL=np.full((n, nb), np.nan),
           fish=np.full((n, nb, ne, 6, 6), np.nan), done=np.zeros(n, bool))
if A.born:
    for k in ('dchi2_born_tr', 'dchi2_born_un', 'mm_fiona_born_un', 'mm_fiona_born_tr', 'maxdF_born_un'):
        out[k] = np.full((n, nb), np.nan)
    out['sig_born_tr'] = np.full((n, nb, 3), np.nan)
    out['fish_born_tr'] = np.full((n, nb, 6, 6), np.nan)
out_path = A.out or os.path.join(os.environ.get('CDM_RAW', str(paths.OUTPUTS_DIR / 'cdm')), f'fiona_cdm_{SD.DET}_{SD.NAME}.npz')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
print(f'{SD.DET} {SD.NAME}: {len(f)} freqs, w\' {WP[0]:.3g}-{WP[-1]:.3g}, sched {A.sched}, {NW} workers', flush=True)

t_all = time.time()
for i in IDX:
    t0 = time.time()
    M, c = M200[i], CC[i]
    h0 = H.halo(M, c, SD.Z_L, SD.z_s)
    ws = H.w_of_f(h0['M_Lz'], F_STAR)
    xi0 = h0['rs_pc'] / np.sqrt(ws)                          # r_F [pc]
    out['kappa'][i], out['w_star_rs'][i] = h0['kappa'], ws
    lenses = [lens_of(M, c, xi0)[0]]
    for e in EPS:
        lenses += [lens_of(M * np.exp(e), c, xi0)[0], lens_of(M * np.exp(-e), c, xi0)[0],
                   lens_of(M, c * np.exp(e), xi0)[0], lens_of(M, c * np.exp(-e), xi0)[0]]
    ys = []
    for bb in B_RF:
        ys += [bb] + [bb * np.exp(s * e) for e in EPS for s in (1, -1)]
    ys = np.array(ys)
    solver = FresnelNUFHTLensBatched(lenses, gl_dir=str(paths.GL_RULES_DIR), n_workers=NW, print_timing=False)
    raw = solver(WP, ys)                                     # (n_f, n_lens, n_y)
    psi_y = np.array([[L.psi_r(np.array([y]))[0] for y in ys] for L in lenses])   # (n_lens, n_y)
    Fall = raw * np.exp(1j * WP[:, None, None] * psi_y[None])
    ny_b = 1 + 2 * ne
    for k, bb in enumerate(B_RF):
        jb = k * ny_b
        F0 = Fall[:, 0, jb]
        nL = np.sum(np.abs(F0) ** 2 * WGT)
        out['snrL'][i, k] = np.sqrt(nL)
        out['dchi2'][i, k] = mismatch_dchi2(np.ones_like(F0), F0)
        for ie, e in enumerate(EPS):
            l0 = 1 + 4 * ie
            dM = (Fall[:, l0, jb] - Fall[:, l0 + 1, jb]) / (2 * e)
            dc = (Fall[:, l0 + 2, jb] - Fall[:, l0 + 3, jb]) / (2 * e)
            db = (Fall[:, 0, jb + 1 + 2 * ie] - Fall[:, 0, jb + 2 + 2 * ie]) / (2 * e)
            out['sig'][i, k, ie] = fisher_sig(F0, (dM, dc, db), keep=out['fish'][i, k, ie])
        if A.born:
            b_pc = bb * xi0
            Fu = 1 + H.born_delta(H.w_of_f(h0['M_Lz'], f), b_pc / h0['rs_pc'], h0['kappa_s'], None,
                                  a_max=b_pc / h0['rs_pc'] * np.sqrt(2 * H.w_of_f(h0['M_Lz'], f).max()))
            Ft = SD.F_born(M, c, b_pc)
            out['dchi2_born_un'][i, k] = mismatch_dchi2(np.ones_like(Fu), Fu)
            out['dchi2_born_tr'][i, k] = mismatch_dchi2(np.ones_like(Ft), Ft)
            out['mm_fiona_born_un'][i, k] = mismatch_dchi2(Fu, F0)
            out['mm_fiona_born_tr'][i, k] = mismatch_dchi2(Ft, F0)
            out['maxdF_born_un'][i, k] = np.abs(F0 - Fu).max()
            ee = 1e-3
            dF = [(SD.F_born(M * np.exp(ee), c, b_pc) - SD.F_born(M * np.exp(-ee), c, b_pc)) / (2 * ee),
                  (SD.F_born(M, c * np.exp(ee), b_pc) - SD.F_born(M, c * np.exp(-ee), b_pc)) / (2 * ee),
                  (SD.F_born(M, c, b_pc * np.exp(ee)) - SD.F_born(M, c, b_pc * np.exp(-ee))) / (2 * ee)]
            out['sig_born_tr'][i, k] = fisher_sig(Ft, dF, keep=out['fish_born_tr'][i, k])
    out['done'][i] = True
    msg = (f'M200 {M:.2e} c {c:.1f} kappa {h0["kappa"]:.2e} w*_rs {ws:.2e} | dchi2 '
           + ' '.join(f'{v:.3g}' for v in out['dchi2'][i]) + ' | sig_lnc(b=rF) '
           + ' '.join(f'{v:.3g}' for v in out['sig'][i, 1, :, 1]))
    if A.born:
        msg += (' | born_tr ' + ' '.join(f'{v:.3g}' for v in out['dchi2_born_tr'][i])
                + ' born_un ' + ' '.join(f'{v:.3g}' for v in out['dchi2_born_un'][i])
                + ' | mm(F,born_un) ' + ' '.join(f'{v:.2g}' for v in out['mm_fiona_born_un'][i])
                + f' | sig_lnc born_tr {out["sig_born_tr"][i, 1, 1]:.3g}')
    print(msg + f'  [{time.time() - t0:.0f} s]', flush=True)
    np.savez(out_path, **out)
print(f'wrote {out_path} in {time.time() - t_all:.0f} s', flush=True)
