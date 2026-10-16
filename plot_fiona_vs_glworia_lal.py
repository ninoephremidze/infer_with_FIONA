#!/usr/bin/env python
"""FIONA vs glworia lensing corners for the corrected CIS runs: both with the LAL Fourier
convention and sampled under the log-uniform M_Lz prior on [0.1, 2000] Msun (no reweighting).

  FIONA:   full_parameter_inference/CIS_glworia_repro/..._log-uniform_wtabv4lal_0 (final results)
  glworia: glworia_repo/outdir/CIS_PE_{strong,weak}_lal_loguniform/... (final result if it exists,
           otherwise the dynesty checkpoint, importance-resampled, marked provisional with its dlogz)

Same style as plot_final_cis_lensing_corners.py (FIONA filled, glworia dashed).
Outputs (this directory, not copied to prd_draft): lensing_corner_fiona_vs_glworia_lal_<case>.{png,pdf}

Run: /n/home02/nephremidze/.conda/envs/glow311/bin/python plot_fiona_vs_glworia_lal.py
"""
import json
from pathlib import Path

import corner
import dill
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dynesty.utils import resample_equal

FIONA = Path('full_parameter_inference/CIS_glworia_repro')
GLW = Path('glworia_repo/outdir')
CASES = [  # short, y, x_c, FIONA tag, glworia regime, colour (Fig. 7)
    ('xc0.02_y0.3', 0.3, 0.02, 'xc0.02_y0.3', 'strong', 'tab:red'),
    ('xc0.2_y1', 1.0, 0.2, 'xc0.2_y1.0', 'weak', 'tab:blue'),
    ('xc0.2_y2', 2.0, 0.2, 'xc0.2_y2.0', 'weak', 'tab:orange'),
    ('xc0.2_y3', 3.0, 0.2, 'xc0.2_y3.0', 'weak', 'tab:green'),
]
SEARCH_ORDER = ['mass_ratio', 'chirp_mass', 'luminosity_distance', 'theta_jn', 'psi', 'phase',
                'a_1', 'a_2', 'MLz', 'y', 'lp', 'H1_time', 'zenith', 'azimuth']   # same for both codes
LENS = ['lp', 'y', 'MLz']
LABELS = [r'$x_c$', r'$y$', r'$\log_{10}(M_{Lz}/M_\odot)$']
LEVELS = (1 - np.exp(-0.5), 1 - np.exp(-2))
BINS_2D, BINS_1D = 25, 10
STYLE = dict(labels=LABELS, levels=LEVELS, show_titles=False, label_kwargs=dict(fontsize=16),
             plot_density=False, plot_datapoints=False, max_n_ticks=4, bins=BINS_2D, smooth=0.9,
             hist_bin_factor=BINS_1D / BINS_2D)


def lens_cols(post):
    return np.column_stack([np.asarray(post['lp'], float), np.asarray(post['y'], float),
                            np.log10(np.asarray(post['MLz'], float))])


def fiona(tag):
    n = f'CIS_glworia_repro_mlz100_{tag}_high_mlzprior0.1-2000.0_log-uniform_wtabv4lal_0'
    p = FIONA / n / f'{n}_result.json'
    if not p.exists():
        return None, None
    d = json.load(open(p))
    return lens_cols(d['posterior']['content']), 'final'


def glworia(y, xc, regime):
    n = f'MLz_100.0_dL_500.0_y_{y:.4f}_xc_{xc:.4f}_lal_loguniform'
    ext = f'CIS_PE_{regime}_lal_loguniform'
    d = GLW / ext / ext / n
    if (d / f'{n}_result.json').exists():
        r = json.load(open(d / f'{n}_result.json'))
        return lens_cols(r['posterior']['content']), 'final'
    if not (d / f'{n}_resume.pickle').exists():
        return None, None
    with open(d / f'{n}_resume.pickle', 'rb') as f:
        obj = dill.load(f)
    sampler = next(o for o in (obj if isinstance(obj, tuple) else (obj,)) if hasattr(o, 'results'))
    res = sampler.results
    w = np.exp(res.logwt - res.logz[-1]); w /= w.sum()
    eq = resample_equal(res.samples, w, rstate=np.random.default_rng(0))
    logl_max = np.max(sampler.live_logl) if hasattr(sampler, 'live_logl') else np.max(res.logl)
    dlogz = float(np.logaddexp(0, logl_max + res.logvol[-1] - res.logz[-1]))
    post = {k: eq[:, i] for i, k in enumerate(SEARCH_ORDER)}
    return lens_cols(post), f'provisional, dlogz {dlogz:.2g}'


for short, y, xc, tag, regime, col in CASES:
    sf, stf = fiona(tag)
    sg, stg = glworia(y, xc, regime)
    if sf is None or sg is None:
        print(f'{short}: FIONA {stf}, glworia {stg} -- skipped')
        continue
    t = [xc, y, 2.0]
    if short == 'xc0.02_y0.3':
        lo = min(np.percentile(s[:, 2], 0.1) for s in (sf, sg))
        hi = max(np.percentile(s[:, 2], 99.9) for s in (sf, sg))
        rng = [(0.0, 0.6), (0.1, 0.5), (lo, hi)]
    else:
        rng = [(0.0, 1.0), (0.1, 5.0), (0.0, np.log10(2000.0))]

    fig = corner.corner(sf, color=col, fill_contours=True, range=rng,
                        hist_kwargs=dict(density=True, lw=1.8, color=col), **STYLE)
    fig = corner.corner(sg, fig=fig, color='black', fill_contours=False, no_fill_contours=True, range=rng,
                        contour_kwargs=dict(linestyles='--', linewidths=1.4),
                        hist_kwargs=dict(density=True, lw=1.6, color='black', ls='--'), **STYLE)
    axes = np.array(fig.axes).reshape(3, 3)
    for i in range(3):
        axes[i, i].hist(sf[:, i], bins=BINS_1D, range=rng[i], density=True, histtype='stepfilled',
                        color=col, alpha=0.25, lw=0)
        axes[i, i].set_ylim(0, 1.05 * max(pp.get_xy()[:, 1].max() for pp in axes[i, i].patches))
    corner.overplot_lines(fig, t, color='black', lw=1.0, alpha=0.6)
    corner.overplot_points(fig, [t], marker='s', color='black', ms=6)
    glw_label = 'glworia' + ('' if stg == 'final' else f' ({stg})')
    fig.legend(handles=[plt.Line2D([], [], color='none', label=rf'$y={y:g},\ x_c={xc:g}$'),
                        plt.Line2D([], [], color=col, lw=3, label='FIONA'),
                        plt.Line2D([], [], color='black', lw=1.6, ls='--', label=glw_label)],
               loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=14, frameon=False)
    for ax in fig.axes:
        ax.tick_params(labelsize=12)
    out = f'lensing_corner_fiona_vs_glworia_lal_{short}'
    fig.savefig(f'{out}.png', bbox_inches='tight', dpi=200)
    fig.savefig(f'{out}.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f'{short}: FIONA {stf} ({len(sf)}), glworia {stg} ({len(sg)})')
    for i, k in enumerate(['xc', 'y', 'logM']):
        mf, mg = np.median(sf[:, i]), np.median(sg[:, i])
        print(f'    {k:4s} FIONA {mf:7.3f} +/- {sf[:, i].std():.3f}   glworia {mg:7.3f} +/- {sg[:, i].std():.3f}'
              f'   (difference / FIONA std = {(mg - mf) / sf[:, i].std():+.2f})')
