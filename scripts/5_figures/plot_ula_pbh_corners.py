#!/usr/bin/env python
"""ULA soliton and PBH lens posteriors of the paper (run_inference_ula_pbh.py: F* on LAL waveforms,
log-uniform M_Lz prior, steep_core quadrature schedule), at y = 0.3, M_Lz = 100 Msun, and r_s = 0.1 for ULA.

  python scripts/5_figures/plot_ula_pbh_corners.py               # from the shipped posterior samples
  python scripts/5_figures/plot_ula_pbh_corners.py --from-runs   # from fresh bilby output

Writes outputs/figures/lensing_corner_<ULA|PBH>.png (lens parameters: r_s, y, log10 M_Lz) and
full_corner_<ULA|PBH>.png (all sampled parameters except the arrival time).
"""
import argparse
import os

import corner
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from infer_with_fiona import posteriors as P

args = P.add_source_arguments(argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)).parse_args()
SOURCE = 'bilby' if args.from_runs else 'shipped'
os.makedirs(args.out_dir, exist_ok=True)
RUNS = {  # tag: (lens, label, sampled parameters, lens parameters, colour, truths)
    'ULA': ('ULA', P.fiona_label('ULA', 0.3, 0.1), P.search_order('ULA'), ['lp', 'y', 'MLz'], 'tab:purple',
            dict(lp=0.1, y=0.3, MLz=2.0)),
    'PBH': ('PBH', P.fiona_label('PBH', 0.3), P.search_order('PBH'), ['y', 'MLz'], 'tab:brown',
            dict(y=0.3, MLz=2.0)),
}
LABEL = dict(lp=r'$r_s$', y=r'$y$', MLz=r'$\log_{10}(M_{Lz}/M_\odot)$', mass_ratio=r'$q$',
             chirp_mass=r'$\mathcal{M}$', luminosity_distance=r'$d_L$', theta_jn=r'$\theta_{JN}$',
             psi=r'$\psi$', phase=r'$\phi$', a_1=r'$a_1$', a_2=r'$a_2$', H1_time=r'$t_{\rm H1}$',
             zenith=r'$\kappa_z$', azimuth=r'$\epsilon_a$')
GW_TRUTH = dict(mass_ratio=1.0, chirp_mass=(30.0 * 30.0) ** 0.6 / 60.0 ** 0.2,
                luminosity_distance=500.0, theta_jn=np.pi / 3, psi=np.pi / 2, phase=np.pi / 2, a_1=0.0,
                a_2=0.0, zenith=2.3666, azimuth=2.9995)
LEVELS = (1 - np.exp(-0.5), 1 - np.exp(-2))
BINS_2D, BINS_1D = 25, 10
STYLE = dict(levels=LEVELS, show_titles=False, plot_density=False, plot_datapoints=False,
             max_n_ticks=4, bins=BINS_2D, smooth=0.9, hist_bin_factor=BINS_1D / BINS_2D)


def cols(post, keys):
    return np.column_stack([np.log10(post[k]) if k == 'MLz' else post[k] for k in keys])


for tag, (lens, label, order, lens_keys, col, truth) in RUNS.items():
    post, info = P.load_fiona(label, SOURCE, args.runs_dir)
    status = info['status']
    print(f'{tag}: {status}')
    lab = dict(LABEL, lp=r'$x_c$') if lens == 'CIS' else LABEL

    s = cols(post, lens_keys)
    t = [truth[k] for k in lens_keys]
    rng = [(a.min(), a.max()) for a in s.T]
    fig = corner.corner(s, labels=[lab[k] for k in lens_keys], color=col, fill_contours=True, range=rng,
                        label_kwargs=dict(fontsize=15), hist_kwargs=dict(density=True, lw=1.8, color=col),
                        **STYLE)
    axes = np.array(fig.axes).reshape(len(lens_keys), len(lens_keys))
    for i in range(len(lens_keys)):   # filled 1D marginals, as in the CIS figure
        axes[i, i].hist(s[:, i], bins=BINS_1D, range=rng[i], density=True, histtype='stepfilled',
                        color=col, alpha=0.25, lw=0)
        axes[i, i].set_ylim(0, 1.05 * max(pp.get_xy()[:, 1].max() for pp in axes[i, i].patches))
    corner.overplot_lines(fig, t, color='black', lw=1.2)
    corner.overplot_points(fig, [t], marker='s', color='black', ms=6)
    fig.savefig(os.path.join(args.out_dir, f'lensing_corner_{tag}.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)

    keys = [k for k in order if k != 'H1_time']
    sf = cols(post, keys)
    tf = [truth.get(k, GW_TRUTH.get(k)) for k in keys]
    fig = corner.corner(sf, labels=[lab[k] for k in keys], color=col, fill_contours=True,
                        label_kwargs=dict(fontsize=13), hist_kwargs=dict(density=True, lw=1.4, color=col),
                        **STYLE)
    corner.overplot_lines(fig, tf, color='black', lw=0.8)
    fig.savefig(os.path.join(args.out_dir, f'full_corner_{tag}.png'), bbox_inches='tight', dpi=110)
    plt.close(fig)

    for i, k in enumerate(lens_keys):
        q = np.percentile(s[:, i], [5, 16, 50, 84, 95])
        print(f'    {k:4s} median {q[2]:7.3f}  68% [{q[1]:.3f}, {q[3]:.3f}]  90% [{q[0]:.3f}, {q[4]:.3f}]'
              f'  truth {t[i]:g}  (median - truth)/std = {(q[2] - t[i]) / s[:, i].std():+.2f}')
