#!/usr/bin/env python
"""Regenerate Figs. 1 and 2 of the PRD draft (prd_draft/figs/LISA_fisher.png, LIGO_fisher.png): Fisher
ellipses of (Delta ln M_Lz, y, x_c) for a CIS lens at M_Lz = 1e5 Msun, y = 0.3, x_c = 0.01.

Copied from the notebooks that made the original figures (read-only, not modified):
  ../0-fiona/realistic_fisher_CIS.ipynb          cells 5, 23, 25, 41  (LISA, z_s = 3, SNR_fid = 1e3)
  ../0-fiona/realistic_fisher_LIGO_CIS_O4.ipynb  cells 5, 25, 28, 34  (aLIGO O4, z_s = 0.1)
with run_fisher from the local fisherWithFIONA copy (same numerics as ../0-fiona's: it only batches the
x_c +- eps solves differently). Only change: legend labels give both component masses units.

  python regen_fisher_figures.py compute   # Fisher runs -> fisher_fig12_cov.npz (raw cov + SNR)
  python regen_fisher_figures.py plot [--snr-ligo 100] [--out-dir prd_draft/figs]
  python regen_fisher_figures.py compute-ref   # same sources, three time-reference conventions -> fisher_fig12_ref.npz
  python regen_fisher_figures.py plot-ref      # figures + table for each convention (not into prd_draft/figs)

compute-ref: FIONA's F carries the shape- and y-dependent phase e^{i w phi_min(x_c, y)} of the first image. The
published figures (A) use it as is, with no arrival-time parameter, so that pure time shift counts as lens
information. C removes it (min-image reference, T23's convention); D also adds the arrival time t_c, as PE
marginalizes over it. The CIS derivatives from fisherWithFIONA (FIONA + bGO above w = 100) are converted
exactly: F_min = F e^{-i w phi}, dF_min/dtheta = (dF/dtheta - i w F dphi/dtheta) e^{-i w phi}, with
dphi/d(ln M) = phi (w is proportional to M) and dphi/dy, dphi/dx_c by central differences of phi_min.
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['compute', 'plot', 'compute-ref', 'plot-ref'])
ap.add_argument('--snr-ligo', type=float, default=100.0, help='LIGO normalization SNR (caption: 100)')
ap.add_argument('--snr-lisa', type=float, default=1e3)
ap.add_argument('--out-dir', default='prd_draft/figs')
ap.add_argument('--suffix', default='')
args = ap.parse_args()

CACHE = 'fisher_fig12_cov.npz'
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


def compute():
    import finufft  # noqa: F401 -- before fiona
    from pycbc.waveform import get_fd_waveform
    from fisherWithFIONA import run_fisher, lisa_psd, ligo_o4_psd, f_isco, f_1month, luminosity_distance

    n_workers = int(os.environ.get('SLURM_CPUS_ON_NODE', 8))
    out = {}
    for det, sources, z_s, psd, fmin, fmax in [('LISA', LISA_SOURCES, 3.0, lisa_psd, 1e-4, 1.0),
                                               ('LIGO', LIGO_SOURCES, 0.1, ligo_o4_psd, 10.0, 1000.0)]:
        D_L = luminosity_distance(z_s)
        for src in sources:
            name, m1, m2 = src[:3]
            dfreq = src[5] if det == 'LIGO' else 1e-7
            M_tot, eta = m1 + m2, m1 * m2 / (m1 + m2) ** 2
            f_lo, f_hi = max(f_1month(M_tot, eta, z_s), fmin), min(f_isco(M_tot, z_s), fmax)
            if f_hi < fmin:
                print(f'{det} {name}: skipped (f_ISCO {f_hi:.2e} Hz below band)')
                continue
            hp, _ = get_fd_waveform(approximant='IMRPhenomD', mass1=float(m1 * (1 + z_s)),
                                    mass2=float(m2 * (1 + z_s)), f_lower=float(f_lo), delta_f=float(dfreq),
                                    distance=float(D_L))
            fp, hpa = hp.sample_frequencies.numpy(), np.abs(hp.numpy())

            def wfunc(f, fp=fp, hpa=hpa, lo=f_lo, hi=f_hi):
                h = np.interp(f, fp, hpa, left=0., right=0.)
                h[(f < lo) | (f > hi)] = 0.
                return h

            res = run_fisher(M_Lz_Msun=np.array([M_LZ]), y=np.array([Y_FID]), x_c=XC_FID, m1_Msun=m1, m2_Msun=m2,
                             z_s=z_s, lens_model='CIS', n_f=256, n_workers=n_workers, return_cov=True,
                             waveform_func=wfunc, psd_func=psd, f_min=fmin, f_max=fmax)
            out[f'{det}_{name}_cov'] = res['cov'][0, 0]
            out[f'{det}_{name}_snr'] = res['SNR'][0, 0]
            print(f'{det} {name}: SNR {res["SNR"][0, 0]:.3g}, raw sigma = {np.sqrt(np.diag(res["cov"][0, 0]))}',
                  flush=True)
    np.savez(CACHE, **out)
    print(f'wrote {CACHE}')


def draw_ellipse(ax, cov2, center, color, levels=(0.68, 0.95), alphas=(0.35, 0.12), lw=1.4):
    from matplotlib.patches import Ellipse
    chi2 = {0.68: 2.30, 0.95: 5.99}
    ev, evec = np.linalg.eigh(cov2)
    angle = np.degrees(np.arctan2(evec[1, -1], evec[0, -1]))
    for level, alpha in zip(levels, alphas):
        s = np.sqrt(chi2[level])
        ax.add_patch(Ellipse(xy=center, width=2 * s * np.sqrt(abs(ev[-1])), height=2 * s * np.sqrt(abs(ev[-2])),
                             angle=angle, facecolor=color, alpha=alpha, linewidth=0))
    s = np.sqrt(chi2[0.68])
    ax.add_patch(Ellipse(xy=center, width=2 * s * np.sqrt(abs(ev[-1])), height=2 * s * np.sqrt(abs(ev[-2])),
                         angle=angle, facecolor='none', edgecolor=color, linewidth=lw))


def corner_fig(covs, labels, colors, fname):
    """Publication corner plot, as in cells 41 (LISA) / 34 (LIGO) of the original notebooks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
    matplotlib.rcParams.update({'font.size': 12, 'figure.dpi': 110})

    plabels = [r'$\Delta\log M_{Lz}$', r'$y$', r'$x_c$']
    pfid = [0.0, Y_FID, XC_FID]
    n = 3
    max_sig = np.array([max(np.sqrt(abs(c[k, k])) for c in covs) for k in range(n)])
    half = 1.5 * np.sqrt(5.99) * max_sig
    lims = [(pfid[k] - half[k], pfid[k] + half[k]) for k in range(n)]
    fig, axes = plt.subplots(n, n, figsize=(6.8, 6.8))
    fig.subplots_adjust(hspace=0.08, wspace=0.08)
    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue
            if col == row:
                x = np.linspace(*lims[col], 500)
                for cov, c in zip(covs, colors):
                    sig = np.sqrt(abs(cov[row, row]))
                    ax.plot(x, np.exp(-0.5 * ((x - pfid[row]) / sig) ** 2), color=c, lw=1.8)
                ax.set_xlim(lims[col]); ax.set_ylim(0, 1.15); ax.set_yticks([])
            else:
                for cov, c in zip(covs, colors):
                    draw_ellipse(ax, cov[np.ix_([col, row], [col, row])], (pfid[col], pfid[row]), c)
                ax.set_xlim(lims[col]); ax.set_ylim(lims[row])
            ax.xaxis.set_major_locator(ticker.MaxNLocator(3, prune='both'))
            if row != col:
                ax.yaxis.set_major_locator(ticker.MaxNLocator(3, prune='both'))
            ax.tick_params(labelsize=10, length=3, width=0.7)
            if row == n - 1:
                ax.set_xlabel(plabels[col], fontsize=13)
            else:
                ax.tick_params(labelbottom=False)
            if col == 0 and row > 0:
                ax.set_ylabel(plabels[row], fontsize=13)
            elif col > 0:
                ax.tick_params(labelleft=False)
    fig.legend(handles=[Line2D([0], [0], color=c, lw=2.2, label=l) for l, c in zip(labels, colors)],
               loc='upper right', bbox_to_anchor=(0.95, 0.93), fontsize=11, framealpha=0.9, handlelength=1.6)
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
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
        corner_fig(covs, labels, colors, os.path.join(args.out_dir, f'{det}_fisher{args.suffix}.png'))


def compute_ref():
    import finufft  # noqa: F401
    from pycbc.waveform import get_fd_waveform
    from fisherWithFIONA import lisa_psd, ligo_o4_psd, f_isco, f_1month, luminosity_distance
    from fisherWithFIONA.fisher import _inner_product
    from fisherWithFIONA.lensing import compute_F_and_derivatives
    from fisherWithFIONA.utils import G_over_c3_Msun
    from fiona_pe_F import phi_min_image

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
    np.savez('fisher_fig12_ref.npz', **out)
    print('wrote fisher_fig12_ref.npz')


def plot_ref():
    d = np.load('fisher_fig12_ref.npz')
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
            corner_fig(covs, labels, colors, os.path.join('fisher_ref_figs', f'{det}_fisher_{tag}.png'))


if args.mode == 'compute-ref':
    compute_ref()
elif args.mode == 'plot-ref':
    os.makedirs('fisher_ref_figs', exist_ok=True)
    plot_ref()
else:
    compute() if args.mode == 'compute' else plot()
