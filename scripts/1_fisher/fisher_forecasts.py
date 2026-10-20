#!/usr/bin/env python
"""Fisher forecasts of the paper (the LISA and LIGO figures): Fisher ellipses of (Delta ln M_Lz, y, x_c) for a
cored isothermal sphere (CIS) lens with y = 0.3 and x_c = 0.01. LISA sources are three SMBH binaries at z_s = 3
(last month before merger), normalized to SNR 1e3. LIGO sources are five compact binaries at z_s = 0.1 with the
aLIGO O4 design sensitivity, normalized to SNR 1e2.

Lens mass. Each source is paired with the lens mass for which w = 10 (`--w-star`) at f_*, the frequency below which 90%
of the source's (unlensed) SNR^2 accumulates, so that its band runs from w ~ 0.4-3 to w ~ 10-50 and covers the wave-optics regime w ~ 1. `--m-lz` instead
fixes one lens mass for every source.

Numerics. F and its derivatives come from infer_with_fiona.fisher.compute_F_and_derivatives: FIONA with the
converged `cis` Gauss-Legendre schedule, blended into the beyond-geometric-optics (bGO) sum above the measured,
lens-dependent switch frequency w_s(x_c, y), with F referenced to the first-arriving image (T23's convention).
Distance and phase are marginalized.

  python scripts/1_fisher/fisher_forecasts.py plot      # fisher_lisa.png, fisher_ligo.png -> outputs/figures
  python scripts/1_fisher/fisher_forecasts.py compute   # rerun the Fisher matrices -> data/fisher_covariances.npz

`plot` needs only data/fisher_covariances.npz and reproduces the paper's figures exactly.
"""
import argparse
import os

import numpy as np

from infer_with_fiona import paths

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('mode', choices=['compute', 'plot'])
ap.add_argument('--snr-ligo', type=float, default=100.0)
ap.add_argument('--snr-lisa', type=float, default=1e3)
ap.add_argument('--m-lz', type=float, default=None, help='one lens mass for every source (default: w(f_*) = --w-star per source)')
ap.add_argument('--w-star', type=float, default=10.0, help='dimensionless frequency at f_* that sets each source\'s lens mass')
ap.add_argument('--gl-schedule', default='cis', help="quadrature schedule ('cis', 'steep_core', or 'default' for FIONA's own)")
ap.add_argument('--cache', default=str(paths.FISHER_COVARIANCES))
ap.add_argument('--out-dir', default=str(paths.FIGURES_DIR))
ap.add_argument('--suffix', default='')
args = ap.parse_args()

Y_FID, XC_FID, N_F, EPS = 0.3, 0.01, 256, 5e-6
LISA_SOURCES = [  # name, m1, m2, label, colour
    ('SMBHB_1e4', 1e4, 1e4, r'SMBHB ($10^4\,M_\odot+10^4\,M_\odot$)', '#4C72B0'),
    ('SMBHB_1e5', 1e5, 1e5, r'SMBHB ($10^5\,M_\odot+10^5\,M_\odot$)', '#DD8452'),
    ('SMBHB_1e6', 1e6, 1e6, r'SMBHB ($10^6\,M_\odot+10^6\,M_\odot$)', '#55A868'),
]
LIGO_SOURCES = [  # name, m1, m2, label, colour, pycbc delta_f
    ('BNS', 1.4, 1.4, r'BNS ($1.4\,M_\odot+1.4\,M_\odot$)', '#4C72B0', 1e-3),
    ('NSBH', 10.0, 1.4, r'NSBH ($10\,M_\odot+1.4\,M_\odot$)', '#DD8452', 1 / 256),
    ('BBH_10', 10.0, 10.0, r'BBH ($10\,M_\odot+10\,M_\odot$)', '#55A868', 1 / 64),
    ('BBH_30', 30.0, 30.0, r'BBH ($30\,M_\odot+30\,M_\odot$)', '#C44E52', 1 / 16),
    ('BBH_60', 60.0, 60.0, r'BBH ($60\,M_\odot+60\,M_\odot$)', '#8172B2', 1 / 4),
]
DETECTORS = {'LISA': (LISA_SOURCES, 3.0, 1e-4, 1.0), 'LIGO': (LIGO_SOURCES, 0.1, 10.0, 1000.0)}


def mass_label(m):
    e = int(np.floor(np.log10(m)))
    return rf'$M_{{Lz}}={m / 10 ** e:.1f}\times10^{{{e}}}\,M_\odot$'


def compute():
    import finufft  # noqa: F401 -- before FIONA
    from pycbc.waveform import get_fd_waveform
    from infer_with_fiona import quadrature
    if args.gl_schedule != 'default':
        quadrature.get_schedule(args.gl_schedule).apply()
    from infer_with_fiona.fisher import lisa_psd, ligo_o4_psd, f_isco, f_1month, luminosity_distance
    from infer_with_fiona.fisher.fisher import _inner_product
    from infer_with_fiona.fisher.lensing import compute_F_and_derivatives
    from infer_with_fiona.fisher.utils import G_over_c3_Msun

    n_workers = int(os.environ.get('SLURM_CPUS_ON_NODE', 8))
    psds = {'LISA': lisa_psd, 'LIGO': ligo_o4_psd}
    out = {}
    for det, (sources, z_s, fmin, fmax) in DETECTORS.items():
        D_L = luminosity_distance(z_s)
        for src in sources:
            name, m1, m2 = src[:3]
            dfreq = src[5] if det == 'LIGO' else 1e-7
            M_tot, eta = m1 + m2, m1 * m2 / (m1 + m2) ** 2
            f_lo, f_hi = max(f_1month(M_tot, eta, z_s), fmin), min(f_isco(M_tot, z_s), fmax)
            if f_hi < fmin:
                continue
            hp, _ = get_fd_waveform(approximant='IMRPhenomD', mass1=float(m1 * (1 + z_s)), mass2=float(m2 * (1 + z_s)),
                                    f_lower=float(f_lo), delta_f=float(dfreq), distance=float(D_L))
            fp, hpa = hp.sample_frequencies.numpy(), np.abs(hp.numpy())
            # f_*: 90% of the unlensed SNR^2 lies below it (T23's reference frequency)
            fg = np.logspace(np.log10(f_lo), np.log10(f_hi), 20000)
            dens = np.interp(fg, fp, hpa, left=0., right=0.) ** 2 / psds[det](fg)
            cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(fg))])
            f_star = float(np.interp(0.9 * cum[-1], cum, fg))
            M_Lz = args.m_lz if args.m_lz else args.w_star / (8 * np.pi * G_over_c3_Msun * f_star)
            f = np.logspace(np.log10(f_lo), np.log10(f_hi), N_F)
            h0 = np.interp(f, fp, hpa, left=0., right=0.)
            S = psds[det](f)
            F, dL, dy, dx = [a[:, 0] for a in compute_F_and_derivatives(
                M_Lz, np.array([Y_FID]), XC_FID, f, lens_model='CIS', n_workers=n_workers,
                eps_logM=EPS, eps_y=EPS, eps_xc=EPS)]
            h = F * h0
            der = [dL * h0, dy * h0, dx * h0, -h, 1j * h]          # ln M_Lz, y, x_c, ln D, phi_0
            Fish = np.array([[_inner_product(a, b, S, f) for b in der] for a in der])
            cov = np.linalg.inv(Fish)
            out[f'{det}_{name}_cov'] = cov[:3, :3]
            out[f'{det}_{name}_snr'] = np.sqrt(_inner_product(h, h, S, f))
            out[f'{det}_{name}_mlz'] = M_Lz
            out[f'{det}_{name}_fstar'] = f_star
            out[f'{det}_{name}_cond'] = np.linalg.cond(Fish)
            w = 8 * np.pi * G_over_c3_Msun * M_Lz * f
            snr_fid = args.snr_lisa if det == 'LISA' else args.snr_ligo
            sig = np.sqrt(np.diag(cov[:3, :3])) * out[f'{det}_{name}_snr'] / snr_fid
            print(f'{det} {name:9s} f_* {f_star:.3g} Hz, M_Lz {M_Lz:.3g} Msun, w {w[0]:.3g}-{w[-1]:.3g}: '
                  f'sigma(lnM, y, x_c) at SNR {snr_fid:g} = {sig[0]:.3g} {sig[1]:.3g} {sig[2]:.3g}, '
                  f'cond {out[f"{det}_{name}_cond"]:.2g}', flush=True)
    np.savez(args.cache, **out)
    print(f'wrote {args.cache}')


def corner_fig(covs, labels, colors, fname):
    """The paper's corner plot (infer_with_fiona.fisher.plotting), saved to fname."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from infer_with_fiona.fisher.plotting import corner_fig as _corner_fig
    plt.close(_corner_fig(covs, labels, colors, (Y_FID, XC_FID), fname))
    print(f'wrote {fname}')


def plot():
    d = np.load(args.cache)
    os.makedirs(args.out_dir, exist_ok=True)
    for det, (sources, *_) in DETECTORS.items():
        snr_fid = args.snr_lisa if det == 'LISA' else args.snr_ligo
        covs, labels, colors = [], [], []
        for src in sources:
            name, label, color = src[0], src[3], src[4]
            if f'{det}_{name}_cov' not in d:
                continue
            covs.append(d[f'{det}_{name}_cov'] * (float(d[f'{det}_{name}_snr']) / snr_fid) ** 2)
            labels.append(f'{label}, {mass_label(float(d[f"{det}_{name}_mlz"]))}')
            colors.append(color)
            print(f'{det} {name}: sigma at SNR {snr_fid:g} = {np.sqrt(np.diag(covs[-1]))}')
        corner_fig(covs, labels, colors, os.path.join(args.out_dir, f'fisher_{det.lower()}{args.suffix}.png'))


compute() if args.mode == 'compute' else plot()
