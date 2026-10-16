#!/usr/bin/env python
"""Paper figures (Figs. 3-4 of prd_draft): lensing corner plots of the final FIONA CIS runs, made
with F* applied to the LAL waveforms (Eq. lensed_waveform) and sampled directly under the
log-uniform M_Lz prior on [0.1, 2000] Msun, so no reweighting. glworia (gray, as in the captions) is
overlaid from its corrected runs (LAL convention, log-uniform M_Lz): the final result if it exists,
otherwise the dynesty checkpoint, importance-resampled (status printed, not shown in the figure).

Same style as the fiona_vs_glworia figures of plot_fig7style_reweighted.py: Fig. 7 colours,
M_Lz as log10, 10-bin marginals, strong case zoomed to x_c <= 0.6, 0.1 <= y <= 0.5.

Outputs (also copied to prd_draft/figs/):
  lensing_corner_fiona_lal_loguniformMLz_<case>.{png,pdf}   for each case whose FIONA run has finished
  (FIONA alone if glworia has no output yet)

Run: /n/home02/nephremidze/.conda/envs/glow311/bin/python plot_final_cis_lensing_corners.py
"""
import json
import shutil
from pathlib import Path

import corner
import dill
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dynesty.utils import resample_equal

BASE = Path('full_parameter_inference/CIS_glworia_repro')
GLW = Path('glworia_repo/outdir')
FIGS = Path('prd_draft/figs')
CASES = [  # short, y, x_c, FIONA tag, colour (Fig. 7)
    ('xc0.02_y0.3', 0.3, 0.02, 'xc0.02_y0.3', 'tab:red'),
    ('xc0.2_y1', 1.0, 0.2, 'xc0.2_y1.0', 'tab:blue'),
    ('xc0.2_y2', 2.0, 0.2, 'xc0.2_y2.0', 'tab:orange'),
    ('xc0.2_y3', 3.0, 0.2, 'xc0.2_y3.0', 'tab:green'),
]
GLW_COLOR = 'tab:gray'
SEARCH_ORDER = ['mass_ratio', 'chirp_mass', 'luminosity_distance', 'theta_jn', 'psi', 'phase',
                'a_1', 'a_2', 'MLz', 'y', 'lp', 'H1_time', 'zenith', 'azimuth']   # glworia sampler order
PARAMS = ['lp', 'y', 'MLz']
LABELS = [r'$x_c$', r'$y$', r'$\log_{10}(M_{Lz}/M_\odot)$']
LEVELS = (1 - np.exp(-0.5), 1 - np.exp(-2))
BINS_2D, BINS_1D = 25, 10
RANGE_WEAK = [(0.0, 1.0), (0.1, 5.0), (0.0, np.log10(2000.0))]
STYLE = dict(labels=LABELS, levels=LEVELS, show_titles=False, label_kwargs=dict(fontsize=16),
             plot_density=False, plot_datapoints=False, max_n_ticks=4, bins=BINS_2D, smooth=0.9,
             hist_bin_factor=BINS_1D / BINS_2D)


def result_path(tag):
    n = f'CIS_glworia_repro_mlz100_{tag}_high_mlzprior0.1-2000.0_log-uniform_wtabv4lal_0'
    return BASE / n / f'{n}_result.json'


def lens_cols(post):
    return np.column_stack([np.asarray(post['lp'], float), np.asarray(post['y'], float),
                            np.log10(np.asarray(post['MLz'], float))])


def glworia(y, xc):
    n = f'MLz_100.0_dL_500.0_y_{y:.4f}_xc_{xc:.4f}_lal_loguniform'
    ext = f'CIS_PE_{"strong" if y < 0.5 else "weak"}_lal_loguniform'
    d = GLW / ext / ext / n
    if (d / f'{n}_result.json').exists():
        return lens_cols(json.load(open(d / f'{n}_result.json'))['posterior']['content']), 'final'
    if not (d / f'{n}_resume.pickle').exists():
        return None, 'not started'
    with open(d / f'{n}_resume.pickle', 'rb') as f:
        obj = dill.load(f)
    sampler = next(o for o in (obj if isinstance(obj, tuple) else (obj,)) if hasattr(o, 'results'))
    res = sampler.results
    w = np.exp(res.logwt - res.logz[-1]); w /= w.sum()
    eq = resample_equal(res.samples, w, rstate=np.random.default_rng(0))
    logl_max = np.max(sampler.live_logl) if hasattr(sampler, 'live_logl') else np.max(res.logl)
    dlogz = float(np.logaddexp(0, logl_max + res.logvol[-1] - res.logz[-1]))
    return lens_cols({k: eq[:, i] for i, k in enumerate(SEARCH_ORDER)}), f'checkpoint, dlogz {dlogz:.2g}'


for short, y, xc, tag, col in CASES:
    p = result_path(tag)
    if not p.exists():
        print(f'{short}: run not finished, skipped')
        continue
    d = json.load(open(p))
    post, inj = d['posterior']['content'], d['injection_parameters']
    s = np.column_stack([np.asarray(post['lp']), np.asarray(post['y']), np.log10(np.asarray(post['MLz']))])
    t = [inj['lp'], inj['y'], np.log10(inj['MLz'])]
    sg, glw_status = glworia(y, xc)

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
    out = f'lensing_corner_fiona_lal_loguniformMLz_{short}'
    fig.savefig(f'{out}.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{out}.pdf', bbox_inches='tight')
    plt.close(fig)
    for ext in ('png', 'pdf'):
        shutil.copy(f'{out}.{ext}', FIGS / f'{out}.{ext}')

    q = np.percentile(s, [16, 50, 84], axis=0)
    sd = s.std(axis=0)
    print(f'{short}: {len(s)} samples, sampling time {d["sampling_time"] / 3600:.1f} h, '
          f'ln BF {d["log_bayes_factor"]:.1f} +/- {d["log_evidence_err"]:.1f};  glworia: {glw_status}')
    for i, k in enumerate(PARAMS):
        gs = '' if sg is None else f'   glworia median {np.median(sg[:, i]):7.3f}'
        print(f'    {k:4s} median {q[1, i]:7.3f}  68% [{q[0, i]:.3f}, {q[2, i]:.3f}]  '
              f'(median - truth)/std = {(q[1, i] - t[i]) / sd[i]:+.2f}' + gs)
