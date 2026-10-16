#!/usr/bin/env python
"""Appendix figures of the PRD draft: full 14-parameter posteriors of the FIONA and glworia
wide-prior CIS runs (uniform M_Lz prior, as sampled; no reweighting), overlaid, with large
labels for the two-column journal layout. Case colours as in the main-text figures
(Cheung+2024 Fig. 7): strong red, y=1 blue, y=2 orange, y=3 green; glworia gray.

Writes prd_draft/figs/full_corner_fiona_wide_tmin_vs_glworia_<case>.png (the files main.tex uses).
Run: /n/home02/nephremidze/.conda/envs/glow311/bin/python plot_full_corners_appendix.py
"""
import json
from pathlib import Path

import bilby
import corner
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

bilby.core.utils.logger.setLevel('WARNING')
OUT = Path('prd_draft/figs')
CASES = [  # short, regime, y, x_c, FIONA tag, colour
    ('xc0.02_y0.3', 'strong', 0.3, 0.02, 'xc0.02_y0.3', 'tab:red'),
    ('xc0.2_y1', 'weak', 1.0, 0.2, 'xc0.2_y1.0', 'tab:blue'),
    ('xc0.2_y2', 'weak', 2.0, 0.2, 'xc0.2_y2.0', 'tab:orange'),
    ('xc0.2_y3', 'weak', 3.0, 0.2, 'xc0.2_y3.0', 'tab:green'),
]
PARAMS = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'luminosity_distance', 'theta_jn', 'psi', 'phase',
          'MLz', 'y', 'lp', 'H1_time', 'zenith', 'azimuth']
LABELS = [r'$\mathcal{M}\,[M_\odot]$', r'$q$', r'$a_1$', r'$a_2$', r'$d_L\,[{\rm Mpc}]$', r'$\theta_{JN}$',
          r'$\psi$', r'$\phi$', r'$\log_{10}M_{Lz}$', r'$y$', r'$x_c$', r'$t_{\rm H1}\,[{\rm ms}]$',
          r'$\kappa_z$', r'$\epsilon_a$']
INJ_ZENITH, INJ_AZIMUTH = 2.3666, 2.9995      # injection is in (ra, dec); see build_fiona_wide_notebook.py
GLW_COLOR = '#7F7F7F'
LEVELS = (1 - np.exp(-0.5), 1 - np.exp(-2))
H1 = bilby.gw.detector.get_empty_interferometer('H1')


def path(code, regime, y, xc, tag):
    if code == 'glworia':
        g = f'MLz_100.0_dL_500.0_y_{y:.4f}_xc_{xc:.4f}'
        return Path('glworia_repo/outdir') / f'CIS_PE_{regime}_prod_7' / f'CIS_PE_{regime}_prod_7' / g / f'{g}_result.json'
    n = f'CIS_glworia_repro_mlz100_{tag}_high_mlzprior0.1-2000.0_uniform_wtabv4tmin_0'
    return Path('full_parameter_inference/CIS_glworia_repro') / n / f'{n}_result.json'


def load(p):
    d = json.load(open(p))
    post, inj = d['posterior']['content'], dict(d['injection_parameters'])
    t_h1 = inj['geocent_time'] + H1.time_delay_from_geocenter(inj['ra'], inj['dec'], inj['geocent_time'])
    m1, m2 = inj['mass_1'], inj['mass_2']                 # injection stores component masses
    inj.setdefault('chirp_mass', (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2)
    inj.setdefault('mass_ratio', min(m1, m2) / max(m1, m2))
    cols, truths = [], []
    for k in PARAMS:
        v = np.asarray(post[k], dtype=float)
        if k == 'MLz':
            v, tr = np.log10(v), np.log10(inj['MLz'])
        elif k == 'H1_time':
            v, tr = (v - t_h1) * 1e3, 0.0                   # ms relative to the injected H1 arrival
        else:
            tr = {'zenith': INJ_ZENITH, 'azimuth': INJ_AZIMUTH}.get(k, inj.get(k))
        cols.append(v)
        truths.append(tr)
    return np.column_stack(cols), truths


plt.rcParams.update({'font.size': 18})
for short, regime, y, xc, tag, col in CASES:
    sg, truths = load(path('glworia', regime, y, xc, tag))
    sf, _ = load(path('fiona', regime, y, xc, tag))
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
    out = OUT / f'full_corner_fiona_wide_tmin_vs_glworia_{short}.png'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('wrote', out)
