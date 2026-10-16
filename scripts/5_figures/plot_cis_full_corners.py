#!/usr/bin/env python
"""Appendix figures: full 14-parameter posteriors of this pipeline (FIONA, coloured) and glworia (gray) for
the four C24 CIS benchmarks, overlaid, with large labels for the two-column journal layout. Both use the
final runs: F* applied to the LAL waveforms and the log-uniform M_Lz prior on [0.1, 2000] Msun.
Case colours follow the main-text figures (C24 Fig. 7): strong red, y=1 blue, y=2 orange, y=3 green.

  python scripts/5_figures/plot_cis_full_corners.py               # from the shipped posterior samples
  python scripts/5_figures/plot_cis_full_corners.py --from-runs   # from fresh bilby/glworia output

Writes outputs/figures/full_corner_fiona_vs_glworia_<case>.png.
"""
import argparse
import os

import bilby
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
bilby.core.utils.logger.setLevel('WARNING')
CASES = [  # short, y, x_c, colour
    ('xc0.02_y0.3', 0.3, 0.02, 'tab:red'),
    ('xc0.2_y1', 1.0, 0.2, 'tab:blue'),
    ('xc0.2_y2', 2.0, 0.2, 'tab:orange'),
    ('xc0.2_y3', 3.0, 0.2, 'tab:green'),
]
PARAMS = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'luminosity_distance', 'theta_jn', 'psi', 'phase',
          'MLz', 'y', 'lp', 'H1_time', 'zenith', 'azimuth']
LABELS = [r'$\mathcal{M}\,[M_\odot]$', r'$q$', r'$a_1$', r'$a_2$', r'$d_L\,[{\rm Mpc}]$', r'$\theta_{JN}$',
          r'$\psi$', r'$\phi$', r'$\log_{10}M_{Lz}$', r'$y$', r'$x_c$', r'$t_{\rm H1}\,[{\rm ms}]$',
          r'$\kappa_z$', r'$\epsilon_a$']
GLW_COLOR = '#7F7F7F'
LEVELS = (1 - np.exp(-0.5), 1 - np.exp(-2))
H1 = bilby.gw.detector.get_empty_interferometer('H1')


INJ = dict(P.INJECTION)
INJ['H1_time'] = INJ['geocent_time'] + H1.time_delay_from_geocenter(INJ['ra'], INJ['dec'], INJ['geocent_time'])


def load(post, y, xc):
    """Samples as columns of PARAMS (log10 M_Lz; H1 arrival time in ms from the injected one), and truths."""
    cols, truths = [], []
    for k in PARAMS:
        v = np.asarray(post[k], dtype=float)
        if k == 'MLz':
            v, tr = np.log10(v), np.log10(INJ['MLz'])
        elif k == 'H1_time':
            v, tr = (v - INJ['H1_time']) * 1e3, 0.0
        else:
            tr = dict(INJ, y=y, lp=xc)[k]
        cols.append(v)
        truths.append(tr)
    return np.column_stack(cols), truths


plt.rcParams.update({'font.size': 18})
for short, y, xc, col in CASES:
    sg, truths = load(P.load_glworia(P.glworia_name(y, xc), SOURCE, args.glworia_dir)[0], y, xc)
    sf, _ = load(P.load_fiona(P.fiona_label('CIS', y, xc), SOURCE, args.runs_dir)[0], y, xc)
    rng = [(min(np.percentile(sg[:, i], 0.3), np.percentile(sf[:, i], 0.3)),
            max(np.percentile(sg[:, i], 99.7), np.percentile(sf[:, i], 99.7))) for i in range(len(PARAMS))]
    fig = plt.figure(figsize=(2.3 * len(PARAMS), 2.3 * len(PARAMS)))
    for s, c in ((sg, GLW_COLOR), (sf, col)):
        fig = corner.corner(s, fig=fig, range=rng, color=c, labels=LABELS, levels=LEVELS, bins=20, smooth=0.9,
                            plot_density=False, plot_datapoints=False, fill_contours=True, max_n_ticks=2,
                            label_kwargs=dict(fontsize=38), labelpad=0.32,
                            hist_kwargs=dict(density=True, lw=3.0, color=c),
                            contour_kwargs=dict(linewidths=2.4))
    corner.overplot_lines(fig, truths, color='black', lw=2.2)
    from matplotlib.ticker import MaxNLocator
    K = len(PARAMS)
    axes = np.array(fig.axes).reshape(K, K)
    for i in range(K):                                  # keep ticks away from panel edges
        for j in range(i + 1):
            axes[i, j].xaxis.set_major_locator(MaxNLocator(3, prune='both'))
            if j < i:
                axes[i, j].yaxis.set_major_locator(MaxNLocator(3, prune='both'))
    for ax in fig.axes:
        ax.tick_params(labelsize=30, width=2.0, length=10)
    fig.legend(handles=[plt.Line2D([], [], color='none', label=rf'$y={y:g},\ x_c={xc:g}$'),
                        plt.Line2D([], [], color=GLW_COLOR, lw=6, label='glworia'),
                        plt.Line2D([], [], color=col, lw=6, label='this work')],
               loc='upper right', bbox_to_anchor=(0.97, 0.97), fontsize=46, frameon=False)
    out = os.path.join(args.out_dir, f'full_corner_fiona_vs_glworia_{short}.png')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('wrote', out)
