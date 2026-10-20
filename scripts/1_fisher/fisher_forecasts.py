#!/usr/bin/env python
"""Fisher forecasts of the paper (the LISA and LIGO figures): Fisher ellipses of (Delta ln M_Lz, y, x_c)
for a CIS lens at M_Lz = 1e5 Msun, y = 0.3, x_c = 0.01. LISA sources are three SMBH binaries at z_s = 3,
normalized to SNR 1e3. LIGO sources are five compact binaries at z_s = 0.1 with the aLIGO O4 design
sensitivity, normalized to SNR 1e2.

  python scripts/1_fisher/fisher_forecasts.py plot       # fisher_lisa.png, fisher_ligo.png -> outputs/figures
  python scripts/1_fisher/fisher_forecasts.py compute    # rerun the Fisher matrices with FIONA
  python scripts/1_fisher/fisher_forecasts.py plot-ref   # figures for all three time-reference conventions

`plot` needs only data/fisher_covariances.npz, so it runs in seconds and reproduces the paper's figures
exactly. `compute` recomputes the covariances with FIONA, overwrites that file, and also writes all three
conventions below to outputs/checks/fisher_time_reference.npz.

Time reference. FIONA's F carries the shape- and y-dependent phase e^{i w phi_min(x_c, y)} of the first image,
i.e. a lens-dependent overall time shift. The paper uses convention C, which removes it (F referenced to the
first-arriving image, as in T23), so that a pure time shift is not counted as information about the lens.
A keeps the shift, and D also adds the arrival time as a Fisher parameter. The derivatives from
infer_with_fiona.fisher (FIONA + bGO above w = 100) are converted exactly:
F_min = F e^{-i w phi}, dF_min/dtheta = (dF/dtheta - i w F dphi/dtheta) e^{-i w phi}, with dphi/d(ln M) = phi
(w is proportional to M) and dphi/dy, dphi/dx_c by central differences of phi_min.
"""
import argparse
import os

import numpy as np

from infer_with_fiona import paths

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['compute', 'plot', 'plot-ref'])
ap.add_argument('--snr-ligo', type=float, default=100.0, help='LIGO normalization SNR (caption: 100)')
ap.add_argument('--snr-lisa', type=float, default=1e3)
ap.add_argument('--out-dir', default=str(paths.FIGURES_DIR))
ap.add_argument('--suffix', default='')
args = ap.parse_args()

CACHE = str(paths.FISHER_COVARIANCES)
REF = str(paths.CHECKS_DIR / 'fisher_time_reference.npz')
REF_FIGS = str(paths.CHECKS_DIR / 'fisher_time_reference_figs')
M_LZ, Y_FID, XC_FID = 1e5, 0.3, 0.01

LISA_SOURCES = [  # name, m1, m2, label, colour
    ('SMBHB_1e4', 1e4, 1e4, r'SMBHB ($10^4\,M_\odot+10^4\,M_\odot$)', '#4C72B0'),
    ('SMBHB_1e5', 1e5, 1e5, r'SMBHB ($10^5\,M_\odot+10^5\,M_\odot$)', '#DD8452'),
    ('SMBHB_1e6', 1e6, 1e6, r'SMBHB ($10^6\,M_\odot+10^6\,M_\odot$)', '#55A868'),
    ('SMBHB_1e7', 1e7, 1e7, r'SMBHB ($10^7\,M_\odot+10^7\,M_\odot$)', '#C44E52'),
]
LIGO_SOURCES = [  # name, m1, m2, label, colour, pycbc delta_f
    ('BNS', 1.4, 1.4, r'BNS ($1.4\,M_\odot+1.4\,M_\odot$)', '#4C72B0', 1e-3),
    ('NSBH', 10.0, 1.4, r'NSBH ($10\,M_\odot+1.4\,M_\odot$)', '#DD8452', 1 / 256),
    ('BBH_10', 10.0, 10.0, r'BBH ($10\,M_\odot+10\,M_\odot$)', '#55A868', 1 / 64),
    ('BBH_30', 30.0, 30.0, r'BBH ($30\,M_\odot+30\,M_\odot$)', '#C44E52', 1 / 16),
    ('BBH_60', 60.0, 60.0, r'BBH ($60\,M_\odot+60\,M_\odot$)', '#8172B2', 1 / 4),
]


def corner_fig(covs, labels, colors, fname):
    """The paper's corner plot (infer_with_fiona.fisher.plotting), saved to fname."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from infer_with_fiona.fisher.plotting import corner_fig as _corner_fig
    plt.close(_corner_fig(covs, labels, colors, (Y_FID, XC_FID), fname))
    print(f'wrote {fname}')


def plot():
    d = np.load(CACHE)
    for det, sources, snr_fid in [('LISA', LISA_SOURCES, args.snr_lisa), ('LIGO', LIGO_SOURCES, args.snr_ligo)]:
        covs, labels, colors = [], [], []
        for src in sources:
            name, label, color = src[0], src[3], src[4]
            if f'{det}_{name}_cov' not in d:
                continue
            covs.append(d[f'{det}_{name}_cov'] * (float(d[f'{det}_{name}_snr']) / snr_fid) ** 2)
            labels.append(label); colors.append(color)
            print(f'{det} {name}: sigma at SNR {snr_fid:g} = {np.sqrt(np.diag(covs[-1]))}')
        corner_fig(covs, labels, colors, os.path.join(args.out_dir, f'fisher_{det.lower()}{args.suffix}.png'))


def compute_ref():
    import finufft  # noqa: F401
    from pycbc.waveform import get_fd_waveform
    from infer_with_fiona.fisher import lisa_psd, ligo_o4_psd, f_isco, f_1month, luminosity_distance
    from infer_with_fiona.fisher.fisher import _inner_product
    from infer_with_fiona.fisher.lensing import compute_F_and_derivatives
    from infer_with_fiona.fisher.utils import G_over_c3_Msun
    from infer_with_fiona.amplification import phi_min_image

    n_workers = int(os.environ.get('SLURM_CPUS_ON_NODE', 8))
    ph = lambda xc, y: phi_min_image(xc, y, 'CIS')
    h_ = 1e-4
    phi0 = ph(XC_FID, Y_FID)
    dphi_y = (ph(XC_FID, Y_FID + h_) - ph(XC_FID, Y_FID - h_)) / (2 * h_)
    dphi_x = (ph(XC_FID + h_, Y_FID) - ph(XC_FID - h_, Y_FID)) / (2 * h_)
    print(f'phi_min = {phi0:.6f}, dphi/dy = {dphi_y:.6f}, dphi/dx_c = {dphi_x:.6f}', flush=True)
    out = dict(phi_min=phi0, dphi_dy=dphi_y, dphi_dxc=dphi_x)
    for det, sources, z_s, psd, fmin, fmax in [('LISA', LISA_SOURCES, 3.0, lisa_psd, 1e-4, 1.0),
                                               ('LIGO', LIGO_SOURCES, 0.1, ligo_o4_psd, 10.0, 1000.0)]:
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
            f = np.logspace(np.log10(f_lo), np.log10(f_hi), 256)          # as run_fisher(n_f=256)
            h0 = np.interp(f, fp, hpa, left=0., right=0.)
            S = psd(f)
            F, dL, dy, dx = [a[:, 0] for a in compute_F_and_derivatives(
                M_LZ, np.array([Y_FID]), XC_FID, f, lens_model='CIS', n_workers=n_workers,
                eps_logM=5e-6, eps_y=5e-6, eps_xc=None, w_high=100.0)]
            w = 8 * np.pi * G_over_c3_Msun * M_LZ * f
            e = np.exp(-1j * w * phi0)
            Fm = F * e
            dLm = (dL - 1j * w * phi0 * F) * e
            dym = (dy - 1j * w * dphi_y * F) * e
            dxm = (dx - 1j * w * dphi_x * F) * e
            for conv, (G, gL, gy, gx), t in [('A', (F, dL, dy, dx), False), ('C', (Fm, dLm, dym, dxm), False),
                                            ('D', (Fm, dLm, dym, dxm), True)]:
                h = G * h0
                der = [gL * h0, gy * h0, gx * h0, -h, 1j * h] + ([2j * np.pi * f * h] if t else [])
                Fish = np.array([[_inner_product(a, b, S, f) for b in der] for a in der])
                C = np.linalg.inv(Fish)
                out[f'{conv}_{det}_{name}_cov'] = C[:3, :3]
                out[f'{conv}_{det}_{name}_snr'] = np.sqrt(_inner_product(h, h, S, f))
            sn = lambda c: np.sqrt(np.diag(out[f'{c}_{det}_{name}_cov']))
            print(f'{det} {name:9s} raw sigma(lnM, y, x_c):  A {sn("A")}  C {sn("C")}  D {sn("D")}', flush=True)
    os.makedirs(os.path.dirname(REF), exist_ok=True)
    np.savez(REF, **out)
    print(f'wrote {REF} (all three conventions)')
    paper = {k[2:]: v for k, v in out.items() if k.startswith('C_')}      # minimum-image reference, as in T23
    np.savez(CACHE, **paper)
    print(f'wrote {CACHE} (minimum-image reference: the covariances behind the paper figures)')


def plot_ref():
    d = np.load(REF)
    for conv, tag in [('A', 'absref'), ('C', 'minimage_T23'), ('D', 'minimage_time')]:
        for det, sources, snr_fid in [('LISA', LISA_SOURCES, args.snr_lisa), ('LIGO', LIGO_SOURCES, args.snr_ligo)]:
            covs, labels, colors = [], [], []
            for src in sources:
                k = f'{conv}_{det}_{src[0]}'
                if f'{k}_cov' not in d:
                    continue
                covs.append(d[f'{k}_cov'] * (float(d[f'{k}_snr']) / snr_fid) ** 2)
                labels.append(src[3]); colors.append(src[4])
                print(f'{conv} {det} {src[0]:9s} sigma(lnM, y, x_c) at SNR {snr_fid:g}: '
                      + ' '.join(f'{v:.3g}' for v in np.sqrt(np.diag(covs[-1]))))
            corner_fig(covs, labels, colors, os.path.join(REF_FIGS, f'{det}_fisher_{tag}.png'))


if args.mode == 'compute':
    compute_ref()
elif args.mode == 'plot-ref':
    os.makedirs(REF_FIGS, exist_ok=True)
    plot_ref()
else:
    os.makedirs(args.out_dir, exist_ok=True)
    plot()
