"""Post-processing of fiona_cdm_rate_<NAME>_<sched>.npz (both quadrature schedules) for NFW halos on the CDM
c-M relation lensing a LISA source: intrinsic-parameter marginalization, two NFW-specific detection criteria,
and the optical depth for detectable lensing.

1. Intrinsic parameters. Chirp mass, symmetric mass ratio and aligned spin (chi_1 = chi_2 = chi; the antisymmetric
   spin is degenerate at equal masses) enter through d h/d theta = h g_theta(f), g_theta = d ln h/d theta from
   pycbc IMRPhenomD (the paper's LISA waveform), as ln[h(theta+e)/h(theta-e)]/2e on pycbc's grid, interpolated;
   eta is differenced one-sidedly from 0.25.
   * Fisher: (ln M200, ln c200, ln b, ln A, phi_0, t_c, ln Mc, eta, chi).
   * Delta chi^2_int: residual r = F0 - T between the lensed signal and the best unlensed template T (best
     amplitude, phase, time), minus its projection onto {T, iT, 2 pi i f T, T g_theta} (linearized refit of the
     intrinsic parameters). This is the expected (noise-free) log-likelihood-ratio statistic 2 ln(L_NFW/L_unl).

2. Detection criteria (NFW-specific; the PBH-calibrated bracket is no longer used).
   (a) Likelihood-ratio test, Wilks with k = 3 extra parameters (M200, c200, b):
       Delta chi^2_int >= 14.2 (3 sigma, p = 2.7e-3) or 31.8 (5 sigma, p = 5.7e-7).
   (b) Bayes factor, Laplace/Fisher approximation with an explicit NFW prior:
         B = exp(Delta chi^2_int / 2) (2 pi)^{3/2} |C_L|^{1/2} E_{theta ~ N(theta_hat, C_L)}[pi(theta)],
       C_L = marginal 3x3 covariance of (ln M200, ln c200, ln b) from the 9x9 Fisher matrix (the nuisance
       parameters' Occam factors cancel between the two models in this approximation), expectation by Monte
       Carlo (N_MC samples). Prior pi: ln M200 uniform on [ln 1e3, ln 1e12]; ln c200 Gaussian about the
       Diemer & Joyce (2019) relation at z_L with SIG_LNC (0.16 dex); b uniform in area within B_OUT r_F,
       i.e. p(ln b) = 2 b^2 / B_OUT^2. Detection: B >= 10 (strong) or >= 100 (decisive).

3. Optical depth. b_max(z_L, M200) = largest b meeting a criterion (interpolated on the b grid: log Delta chi^2
   or ln B linear in ln b). tau = int dchi int dlnM dn/dlnM pi [b_max (1+z_L)]^2 (comoving), Tinker et al. (2008)
   M200c mass function (colossus), chi_L/chi_S in {0, 0.1, 0.3, 0.5, 0.7, 0.9, 1} with zero integrand at the
   ends (trapezoid). Only points where the two schedules agree (Delta chi^2 to TOL) and b <= B_OK r_F are used.
   flag: 0 ok, 1 criterion met at the largest usable b (b_max a lower limit), -1 met nowhere, 9 < 2 usable b.

  python fiona_cdm_rate_post.py SMBHB_1e4 [SMBHB_1e5 ...]
"""
import os
import sys
import numpy as np
from scipy.stats import chi2, norm
from pycbc.waveform import get_fd_waveform
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from astropy.cosmology import Planck18 as COSMO

cosmology.setCosmology('planck18')
from infer_with_fiona import paths
RAW = os.environ.get('CDM_RAW', str(paths.OUTPUTS_DIR / 'cdm'))      # raw FIONA runs (regenerable)
D = str(paths.DATA_DIR / 'cdm')                 # post-processed results (shipped)
MASSES = {'SMOKE': 1e4, 'SMBHB_1e4': 1e4, 'SMBHB_1e5': 1e5, 'SMBHB_1e6': 1e6, 'SMBHB_1e7': 1e7}
K_LENS = 3
LRT = {'3sig': chi2.isf(2 * norm.sf(3), K_LENS), '5sig': chi2.isf(2 * norm.sf(5), K_LENS)}
LNB = {'B10': np.log(10.0), 'B100': np.log(100.0)}
TOL = 0.02
B_OK = 20.0          # b = 30 r_F lies outside FIONA's integration domain at the highest frequencies
B_OUT = float(os.environ.get('B_OUT', 10.0))   # prior on b: uniform in area within B_OUT r_F
SIG_LNC = float(os.environ.get('SIG_DEX', 0.16)) * np.log(10)
TAG = os.environ.get('TAG', '')
IN_TAG = os.environ.get('IN_TAG', '')   # '_imr': inputs from the full-band, native-SNR runs
LNM_PRIOR = (np.log(1e3), np.log(1e12))
N_MC = 200000
RNG = np.random.default_rng(1)
Z_STD = RNG.standard_normal((N_MC, 3))


def hfd(m1, m2, chi, f_lo, df):
    hp, _ = get_fd_waveform(approximant='IMRPhenomD', mass1=m1, mass2=m2, spin1z=chi, spin2z=chi,
                            f_lower=float(f_lo), delta_f=df, distance=1000.0)
    return hp.sample_frequencies.numpy(), hp.numpy()


def g_theta(f, m_src, z_s, df=5e-8):
    """g = d ln h / d (ln Mc, eta, chi) on the frequency grid f (complex, shape (3, n_f))."""
    M = 2 * m_src * (1 + z_s)                              # detector-frame total mass
    eta0, Mc0 = 0.25, (0.25 ** 0.6) * M
    f_lo = f[0] * 0.98

    def masses(Mc, eta):
        Mt = Mc / eta ** 0.6
        d = np.sqrt(max(1 - 4 * eta, 0.0))
        return Mt * (1 + d) / 2, Mt * (1 - d) / 2

    def ratio(pa, pb, step):
        fa, ha = hfd(*pa, f_lo, df)
        fb, hb = hfd(*pb, f_lo, df)
        n = min(len(ha), len(hb))
        ok = (np.abs(ha[:n]) > 0) & (np.abs(hb[:n]) > 0)
        g = np.zeros(n, complex)
        g[ok] = np.log(ha[:n][ok] / hb[:n][ok]) / step
        return np.interp(f, fa[:n], g.real) + 1j * np.interp(f, fa[:n], g.imag)

    e = 1e-7
    gMc = ratio((*masses(Mc0 * np.exp(e), eta0), 0.0), (*masses(Mc0 * np.exp(-e), eta0), 0.0), 2 * e)
    e = 1e-6
    gEta = ratio((*masses(Mc0, eta0), 0.0), (*masses(Mc0, eta0 - e), 0.0), e)
    gChi = ratio((*masses(Mc0, eta0), e), (*masses(Mc0, eta0), -e), 2 * e)
    return np.array([gMc, gEta, gChi])


def ip(a, b, W):
    return np.sum(np.real(np.conj(a) * b) * W)


def gram_project(r, basis, W):
    G = np.array([[ip(a, b, W) for b in basis] for a in basis])
    v = np.array([ip(a, r, W) for a in basis])
    return ip(r, r, W) - v @ np.linalg.solve(G, v)


def ln_prior(th, lnc_of_lnM, ln_bout):
    """ln pi(ln M200, ln c200, ln b) for samples th (N, 3)."""
    lnM, lnc, lnb = th.T
    lp = np.full(len(th), -np.inf)
    ok = (lnM >= LNM_PRIOR[0]) & (lnM <= LNM_PRIOR[1]) & (lnb <= ln_bout)
    mu = lnc_of_lnM(lnM[ok])
    lp[ok] = (-np.log(LNM_PRIOR[1] - LNM_PRIOR[0]) - 0.5 * ((lnc[ok] - mu) / SIG_LNC) ** 2
              - np.log(np.sqrt(2 * np.pi) * SIG_LNC) + np.log(2.0) + 2 * (lnb[ok] - ln_bout))
    return lp


U_PRIOR = RNG.random((N_MC, 3))


def prior_samples(lnc_of_lnM, ln_bout):
    """N_MC samples from the NFW prior (fixed random numbers, so ln B is smooth in the parameters)."""
    lnM = LNM_PRIOR[0] + (LNM_PRIOR[1] - LNM_PRIOR[0]) * U_PRIOR[:, 0]
    lnc = lnc_of_lnM(lnM) + SIG_LNC * norm.ppf(np.clip(U_PRIOR[:, 1], 1e-12, 1 - 1e-12))
    lnb = ln_bout + 0.5 * np.log(np.clip(U_PRIOR[:, 2], 1e-300, 1.0))     # b^2 uniform on [0, b_out^2]
    return np.column_stack([lnM, lnc, lnb])


def ln_bayes(d, C, th0, lnc_of_lnM, ln_bout):
    """Laplace/Fisher ln B for Delta chi^2 d, lens covariance C (3x3), MLE th0, with the NFW prior:
    ln B = d/2 + ln[(2 pi)^{3/2} |C|^{1/2}] + ln int pi(theta) N(theta; th0, C) dtheta. The integral is estimated by
    importance sampling from the equal mixture of N and pi, which is robust both when the likelihood is much
    narrower than the prior and when it is much broader (degenerate directions)."""
    try:
        Lc = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        return np.nan
    Ci = np.linalg.inv(C)
    ln_norm = -1.5 * np.log(2 * np.pi) - np.sum(np.log(np.diag(Lc)))
    th = np.vstack([th0 + Z_STD @ Lc.T, prior_samples(lnc_of_lnM, ln_bout)])
    dd = th - th0
    lnN = ln_norm - 0.5 * np.einsum('ij,jk,ik->i', dd, Ci, dd)
    lnP = ln_prior(th, lnc_of_lnM, ln_bout)
    lnq = np.logaddexp(lnN, lnP) - np.log(2.0)
    lw = lnP + lnN - lnq
    m = np.max(lw)
    if not np.isfinite(m):
        return -np.inf
    ln_E = m + np.log(np.mean(np.exp(lw - m)))
    return d / 2 - ln_norm + ln_E


def crossing(b, s, thr, log):
    """Largest b with score s >= thr; (b, flag)."""
    above = s >= thr
    if not above.any():
        return 0.0, -1
    if above[-1]:
        return b[-1], 1
    j = np.max(np.nonzero(above)[0])
    y0, y1, t = s[j], s[j + 1], thr
    if log:
        y0, y1, t = np.log(y0), np.log(max(y1, 1e-300)), np.log(thr)
    x0, x1 = np.log(b[j]), np.log(b[j + 1])
    return float(np.exp(x0 + (t - y0) * (x1 - x0) / (y1 - y0))), 0


for name in sys.argv[1:]:
    A = dict(np.load(f'{RAW}/fiona_cdm_rate_{name}_steep_core{IN_TAG}.npz', allow_pickle=True))
    Bc = dict(np.load(f'{RAW}/fiona_cdm_rate_{name}_cis_x2{IN_TAG}.npz', allow_pickle=True))
    f, W, zs = A['f'], A['wgt'], float(A['z_s'])
    if os.environ.get('NATIVE_SNR'):                    # rescale from SNR = 1e3 to the native SNR (W sums to SNR^2)
        snr_nat = float(np.load(f'{D}/native_snr.npz')[name])
        W = W * (snr_nat / float(A['snr'])) ** 2
        print(f'{name}: native SNR {snr_nat:.1f} (W rescaled by {(snr_nat / float(A["snr"])) ** 2:.3g})', flush=True)
    G = g_theta(f, MASSES[name], zs)
    nz, nm, nb = A['dchi2'].shape
    conv = A['done'][..., None] & Bc['done'][..., None] & (np.abs(A['dchi2'] - Bc['dchi2'])
                                                        <= TOL * np.maximum(np.abs(Bc['dchi2']), 1e-12))
    conv &= (A['b_rF'] <= B_OK)[None, None, :]
    lnM = np.log(A['M200'])
    d_int = np.full((nz, nm, nb), np.nan)
    lnB = np.full((nz, nm, nb), np.nan)
    sig = np.full((nz, nm, nb, 2, 3), np.nan)              # [fixed, marginalized intrinsic] x (lnc, lnSig, lnc|M)
    for iz in range(nz):
        okc = np.isfinite(A['c200'][iz])
        lnc_of_lnM = (lambda x, a=lnM[okc], c=np.log(A['c200'][iz][okc]): np.interp(x, a, c))
        for i in range(nm):
            if not A['done'][iz, i]:
                continue
            c = A['c200'][iz, i]
            dlnm = c * c / ((1 + c) ** 2 * (np.log1p(c) - c / (1 + c)))
            gS = np.array([1 / 3, 2 - dlnm] + [0] * 7)
            for k in range(nb):
                F0, dF, t = A['F0'][iz, i, k], A['dF'][iz, i, k], A['tbest'][iz, i, k]
                T0 = np.exp(-2j * np.pi * f * t)
                T = T0 * np.sum(np.conj(T0) * F0 * W) / np.sum(W)
                basis = [T, 1j * T, 2j * np.pi * f * T] + [T * g for g in G]
                d_int[iz, i, k] = gram_project(F0 - T, basis, W)
                der = list(dF) + [F0, 1j * F0, 2j * np.pi * f * F0] + [F0 * g for g in G]
                Fi = np.array([[ip(a, b, W) for b in der] for a in der])
                for j, n in enumerate((6, 9)):
                    try:
                        C = np.linalg.inv(Fi[:n, :n])
                        Cm = np.linalg.inv(Fi[1:n, 1:n])
                        sig[iz, i, k, j] = [np.sqrt(abs(C[1, 1])), np.sqrt(abs(gS[:n] @ C @ gS[:n])),
                                            np.sqrt(abs(Cm[0, 0]))]
                        if n == 9:
                            CL = C[:3, :3]
                            th0 = np.array([lnM[i], np.log(c), np.log(A['b_rF'][k])])
                            lnB[iz, i, k] = ln_bayes(d_int[iz, i, k], CL, th0, lnc_of_lnM, np.log(B_OUT))
                    except np.linalg.LinAlgError:
                        pass
    # convergence of sigma(ln c200 | M200) (intrinsic fixed, 6x6): both schedules and both steps within 5%
    def sig_cM(Fi):
        try:
            return np.sqrt(abs(np.linalg.inv(Fi[1:, 1:])[0, 0]))
        except np.linalg.LinAlgError:
            return np.nan
    conv_sig = np.zeros((nz, nm, nb), bool)
    for iz in range(nz):
        for i in range(nm):
            for k in range(nb):
                if not conv[iz, i, k]:
                    continue
                v = np.array([sig_cM(A['fish'][iz, i, k, 0]), sig_cM(A['fish'][iz, i, k, 1]),
                              sig_cM(Bc['fish'][iz, i, k, 0])])
                conv_sig[iz, i, k] = np.all(np.isfinite(v)) and np.ptp(v) <= 0.05 * v[0]
    # optical depth for each criterion
    chiS = COSMO.comoving_distance(zs).value
    xg = np.concatenate([[0.0], A['xfrac'], [1.0]])
    dndlnM = np.array([mass_function.massFunction(A['M200'] * COSMO.h, z, mdef='200c', model='tinker08',
                                                  q_out='dndlnM') * COSMO.h ** 3 for z in A['z_L']])
    crits = {f'LRT_{k}': (d_int, v, True) for k, v in LRT.items()}
    crits.update({f'Bayes_{k}': (lnB, v, False) for k, v in LNB.items()})
    res = {}
    for cname, (S, thr, lg) in crits.items():
        bm = np.zeros((nz, nm)); fl = np.zeros((nz, nm), int)
        for iz in range(nz):
            for i in range(nm):
                ok = conv[iz, i] & np.isfinite(S[iz, i])
                if ok.sum() < 2:
                    fl[iz, i] = 9; continue
                bm[iz, i], fl[iz, i] = crossing(A['b_rF'][ok], S[iz, i][ok], thr, lg)
        integ = np.where(fl == 9, 0.0, dndlnM * np.pi * (bm * A['rF_pc'] * 1e-6 * (1 + A['z_L'][:, None])) ** 2)
        integ = np.nan_to_num(integ)
        per_z = np.trapezoid(integ, lnM, axis=1)
        tau = np.trapezoid(np.concatenate([[0.0], per_z, [0.0]]), xg * chiS)
        dtau = np.trapezoid(np.vstack([np.zeros(nm), integ, np.zeros(nm)]), xg * chiS, axis=0)
        res[cname] = dict(b_max=bm, flag=fl, tau=tau, dtau_dlnM=dtau, thr=thr)
    np.savez(f'{D}/fiona_cdm_rate_post_{name}{TAG}.npz', d_int=d_int, lnB=lnB, sig=sig, conv=conv, conv_sig=conv_sig, G=G,
             M200=A['M200'], z_L=A['z_L'], b_rF=A['b_rF'], c200=A['c200'], kappa=A['kappa'], rF_pc=A['rF_pc'],
             dchi2_fixed=A['dchi2'],
             **{f'{c}__{k}': v for c, r in res.items() for k, v in r.items()})
    # summary
    print(f'\n===== {name}  (z_s {zs:.2f});  LRT thresholds (k={K_LENS}): '
          + ', '.join(f'{k} {v:.1f}' for k, v in LRT.items()))
    kz = int(np.argmin(np.abs(A['xfrac'] - 0.5))); kb = int(np.argmin(np.abs(A['b_rF'] - 1.0)))
    print(f'z_L = {A["z_L"][kz]:.3f}, b = r_F:  M200  conv  dchi2_fixed  dchi2_int  lnB | '
          'sig(lnc) sig(lnSig) sig(lnc|M): [intrinsic fixed] -> [marginalized]')
    for i in range(nm):
        if not A['done'][kz, i]:
            continue
        print(f'  {A["M200"][i]:8.2e} {"y" if conv[kz, i, kb] else "n"} {A["dchi2"][kz, i, kb]:10.3g} '
              f'{d_int[kz, i, kb]:10.3g} {lnB[kz, i, kb]:8.3g} | '
              + ' '.join(f'{v:7.3g}' for v in sig[kz, i, kb, 0]) + ' -> '
              + ' '.join(f'{v:7.3g}' for v in sig[kz, i, kb, 1]))
    for cname, r in res.items():
        top = int(np.argmax(r['dtau_dlnM'])) if r['tau'] > 0 else -1
        print(f'tau[{cname:12s}] = {r["tau"]:.3g}'
              + (f'  (peak dtau/dlnM at M200 = {A["M200"][top]:.1e}; flags over z_L there: {r["flag"][:, top]};'
                 f' unconverged (z,M) cells: {(r["flag"] == 9).sum()})' if top >= 0 else ''))
