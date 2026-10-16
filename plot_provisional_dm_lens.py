#!/usr/bin/env python
"""Provisional posteriors of the ULA and PBH runs (PE_DM_lens.py: F* on LAL waveforms, log-uniform
M_Lz prior, GL v4x4; label ..._log-uniform_wtabv4x4lal_0), read from the final result if it exists,
otherwise from the dynesty checkpoint (dead points, importance-resampled).

Outputs (this directory):
  lensing_corner_provisional_<tag>.png   lens parameters (ULA: r_s, CIS: x_c, then y, log10 M_Lz)
  full_corner_provisional_<tag>.png       all sampled parameters

Run: /n/home02/nephremidze/.conda/envs/glow311/bin/python plot_provisional_dm_lens.py
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

BASE = Path('full_parameter_inference')
GW = ['mass_ratio', 'chirp_mass', 'luminosity_distance', 'theta_jn', 'psi', 'phase', 'a_1', 'a_2']
RUNS = {  # tag: (lens, label, search order (from the bilby log), lens params, colour, truths)
    'ULA': ('ULA', 'ULA_glworia_repro_mlz100_rs0.1_y0.3_high_mlzprior0.1-2000.0_log-uniform_wtabv4x4lal_0',
            GW + ['MLz', 'y', 'lp', 'H1_time', 'zenith', 'azimuth'], ['lp', 'y', 'MLz'], 'tab:purple',
            dict(lp=0.1, y=0.3, MLz=2.0)),
    'PBH': ('PBH', 'PBH_glworia_repro_mlz100_y0.3_high_mlzprior0.1-2000.0_log-uniform_wtabv4x4lal_0',
            GW + ['MLz', 'y', 'H1_time', 'zenith', 'azimuth'], ['y', 'MLz'], 'tab:brown',
            dict(y=0.3, MLz=2.0)),
    # mass scan (M_Lz = 3)
    'CIS_mlz3': ('CIS', 'CIS_glworia_repro_mlz3_xc0.02_y0.3_high_mlzprior0.1-2000.0_log-uniform_wtabv4lal_0',
                 GW + ['MLz', 'y', 'lp', 'H1_time', 'zenith', 'azimuth'], ['lp', 'y', 'MLz'], 'tab:blue',
                 dict(lp=0.02, y=0.3, MLz=np.log10(3.0))),
    'PBH_mlz3': ('PBH', 'PBH_glworia_repro_mlz3_y0.3_high_mlzprior0.1-2000.0_log-uniform_wtabv4x4lal_0',
                 GW + ['MLz', 'y', 'H1_time', 'zenith', 'azimuth'], ['y', 'MLz'], 'tab:brown',
                 dict(y=0.3, MLz=np.log10(3.0))),
}
LABEL = dict(lp=r'$r_s$', y=r'$y$', MLz=r'$\log_{10}(M_{Lz}/M_\odot)$', mass_ratio=r'$q$',
             chirp_mass=r'$\mathcal{M}$', luminosity_distance=r'$d_L$', theta_jn=r'$\theta_{JN}$',
             psi=r'$\psi$', phase=r'$\phi$', a_1=r'$a_1$', a_2=r'$a_2$', H1_time=r'$t_{\rm H1}$',
             zenith=r'$\kappa_z$', azimuth=r'$\epsilon_a$')
GW_TRUTH = dict(mass_ratio=1.0, chirp_mass=(30.0 * 30.0) ** 0.6 / 60.0 ** 0.2,
                luminosity_distance=500.0, theta_jn=np.pi / 3, psi=np.pi / 2, phase=np.pi / 2, a_1=0.0,
                a_2=0.0, zenith=2.3666, azimuth=2.9995)
LEVELS = (1 - np.exp(-0.5), 1 - np.exp(-2))
STYLE = dict(levels=LEVELS, show_titles=False, plot_density=False, plot_datapoints=False,
             max_n_ticks=4, bins=25, smooth=0.9, hist_bin_factor=10 / 25)


def load(tag):
    lens, label, order = RUNS[tag][:3]
    d = BASE / f'{lens}_glworia_repro' / label
    if (d / f'{label}_result.json').exists():
        r = json.load(open(d / f'{label}_result.json'))
        return {k: np.asarray(v, float) for k, v in r['posterior']['content'].items() if k in order}, 'final'
    with open(d / f'{label}_resume.pickle', 'rb') as f:
        obj = dill.load(f)
    sampler = next(o for o in (obj if isinstance(obj, tuple) else (obj,)) if hasattr(o, 'results'))
    res = sampler.results
    w = np.exp(res.logwt - res.logz[-1]); w /= w.sum()
    eq = resample_equal(res.samples, w, rstate=np.random.default_rng(0))
    logl_max = np.max(sampler.live_logl) if hasattr(sampler, 'live_logl') else np.max(res.logl)
    dlogz = float(np.logaddexp(0, logl_max + res.logvol[-1] - res.logz[-1]))
    ess = 1.0 / np.sum(w ** 2)
    return ({k: eq[:, i] for i, k in enumerate(order)},
            f'provisional: {len(res.logl)} iterations, dlogz {dlogz:.3g}, ESS {ess:.0f}')


def cols(post, keys):
    return np.column_stack([np.log10(post[k]) if k == 'MLz' else post[k] for k in keys])


for tag, (lens, label, order, lens_keys, col, truth) in RUNS.items():
    post, status = load(tag)
    print(f'{tag}: {status}')
    lab = dict(LABEL, lp=r'$x_c$') if lens == 'CIS' else LABEL

    s = cols(post, lens_keys)
    t = [truth[k] for k in lens_keys]
    fig = corner.corner(s, labels=[lab[k] for k in lens_keys], color=col, fill_contours=True,
                        label_kwargs=dict(fontsize=15), hist_kwargs=dict(density=True, lw=1.8, color=col),
                        **STYLE)
    corner.overplot_lines(fig, t, color='black', lw=1.2)
    corner.overplot_points(fig, [t], marker='s', color='black', ms=6)
    fig.savefig(f'lensing_corner_provisional_{tag}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)

    keys = [k for k in order if k != 'H1_time']
    sf = cols(post, keys)
    tf = [truth.get(k, GW_TRUTH.get(k)) for k in keys]
    fig = corner.corner(sf, labels=[lab[k] for k in keys], color=col, fill_contours=True,
                        label_kwargs=dict(fontsize=13), hist_kwargs=dict(density=True, lw=1.4, color=col),
                        **STYLE)
    corner.overplot_lines(fig, tf, color='black', lw=0.8)
    fig.savefig(f'full_corner_provisional_{tag}.png', bbox_inches='tight', dpi=110)
    plt.close(fig)

    for i, k in enumerate(lens_keys):
        q = np.percentile(s[:, i], [5, 16, 50, 84, 95])
        print(f'    {k:4s} median {q[2]:7.3f}  68% [{q[1]:.3f}, {q[3]:.3f}]  90% [{q[0]:.3f}, {q[4]:.3f}]'
              f'  truth {t[i]:g}  (median - truth)/std = {(q[2] - t[i]) / s[:, i].std():+.2f}')
