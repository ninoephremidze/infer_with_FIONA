#!/usr/bin/env python
"""CIS lens-parameter posteriors of the paper (four C24 benchmarks): this pipeline (FIONA, filled, Fig. 7
colours of C24) against glworia (gray), both with F* applied to the LAL waveforms and the log-uniform M_Lz
prior on [0.1, 2000] Msun, on identical data. M_Lz is shown as log10, with 10-bin marginals. The strong case
is zoomed to x_c <= 0.6, 0.1 <= y <= 0.5.

  python scripts/5_figures/plot_cis_lens_corners.py               # from the shipped posterior samples
  python scripts/5_figures/plot_cis_lens_corners.py --from-runs   # from fresh bilby/glworia output

Writes outputs/figures/lensing_corner_fiona_lal_loguniformMLz_<case>.{png,pdf}.
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
CASES = [  # short, y, x_c, colour (C24 Fig. 7)
    ('xc0.02_y0.3', 0.3, 0.02, 'tab:red'),
    ('xc0.2_y1', 1.0, 0.2, 'tab:blue'),
    ('xc0.2_y2', 2.0, 0.2, 'tab:orange'),
    ('xc0.2_y3', 3.0, 0.2, 'tab:green'),
]
GLW_COLOR = 'tab:gray'
PARAMS = ['lp', 'y', 'MLz']
LABELS = [r'$x_c$', r'$y$', r'$\log_{10}(M_{Lz}/M_\odot)$']
LEVELS = (1 - np.exp(-0.5), 1 - np.exp(-2))
BINS_2D, BINS_1D = 25, 10
RANGE_WEAK = [(0.0, 1.0), (0.1, 5.0), (0.0, np.log10(2000.0))]
STYLE = dict(labels=LABELS, levels=LEVELS, show_titles=False, label_kwargs=dict(fontsize=16),
             plot_density=False, plot_datapoints=False, max_n_ticks=4, bins=BINS_2D, smooth=0.9,
             hist_bin_factor=BINS_1D / BINS_2D)


def lens_cols(post):
    return np.column_stack([np.asarray(post['lp'], float), np.asarray(post['y'], float),
                            np.log10(np.asarray(post['MLz'], float))])


for short, y, xc, col in CASES:
    try:
        post, info = P.load_fiona(P.fiona_label('CIS', y, xc), SOURCE, args.runs_dir)
    except FileNotFoundError as e:
        print(f'{short}: {e}, skipped')
        continue
    s = lens_cols(post)
    t = [xc, y, np.log10(P.INJECTION['MLz'])]
    try:
        pg, ginfo = P.load_glworia(P.glworia_name(y, xc), SOURCE, args.glworia_dir)
        sg, glw_status = lens_cols(pg), ginfo['status']
    except FileNotFoundError:
        sg, glw_status = None, 'not available'

    rng = list(RANGE_WEAK)
    if short == 'xc0.02_y0.3':
        both = [s] if sg is None else [s, sg]
        rng = [(0.0, 0.6), (0.1, 0.5), (min(np.percentile(a[:, 2], 0.1) for a in both),
                                        max(np.percentile(a[:, 2], 99.9) for a in both))]
    fig = corner.corner(s, color=col, fill_contours=True, range=rng,
                        hist_kwargs=dict(density=True, lw=1.8, color=col), **STYLE)
    if sg is not None:
        fig = corner.corner(sg, fig=fig, color=GLW_COLOR, fill_contours=False, no_fill_contours=True,
                            range=rng, contour_kwargs=dict(linewidths=1.6),
                            hist_kwargs=dict(density=True, lw=1.8, color=GLW_COLOR), **STYLE)
    axes = np.array(fig.axes).reshape(3, 3)
    for i in range(3):
        axes[i, i].hist(s[:, i], bins=BINS_1D, range=rng[i], density=True, histtype='stepfilled',
                        color=col, alpha=0.25, lw=0)
        axes[i, i].set_ylim(0, 1.05 * max(pp.get_xy()[:, 1].max() for pp in axes[i, i].patches))
    corner.overplot_lines(fig, t, color='black', lw=1.2)
    corner.overplot_points(fig, [t], marker='s', color='black', ms=6)
    fig.legend(handles=[plt.Line2D([], [], color='none', label=rf'$y={y:g},\ x_c={xc:g}$'),
                        plt.Line2D([], [], color=col, lw=3, label='FIONA')]
               + ([] if sg is None else [plt.Line2D([], [], color=GLW_COLOR, lw=3, label='glworia')]),
               loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=15, frameon=False)
    for ax in fig.axes:
        ax.tick_params(labelsize=12)
    out = os.path.join(args.out_dir, f'lensing_corner_fiona_lal_loguniformMLz_{short}')
    fig.savefig(f'{out}.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out}.pdf', bbox_inches='tight')
    plt.close(fig)

    q = np.percentile(s, [16, 50, 84], axis=0)
    sd = s.std(axis=0)
    print(f'{short}: {len(s)} samples ({info["status"]}), ln Z {info["log_evidence"]:.1f};  '
          f'glworia: {glw_status}')
    for i, k in enumerate(PARAMS):
        gs = '' if sg is None else f'   glworia median {np.median(sg[:, i]):7.3f}'
        print(f'    {k:4s} median {q[1, i]:7.3f}  68% [{q[0, i]:.3f}, {q[2, i]:.3f}]  '
              f'(median - truth)/std = {(q[1, i] - t[i]) / sd[i]:+.2f}' + gs)
