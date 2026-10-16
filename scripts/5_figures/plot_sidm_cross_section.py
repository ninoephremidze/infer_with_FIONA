#!/usr/bin/env python
"""SIDM cross-section posteriors from the final FIONA CIS runs (F* on LAL waveforms, log-uniform M_Lz
prior): each posterior sample (x_c, M_Lz) -> <sigma v>/m with dm_conversions (paper Eq. sidm_sigma,
per-sample M_Lz and M_vir), at the injection's source distance (d_L = 500 Mpc) and T23's pessimistic z_L.

Outputs (outputs/figures, or --out-dir):
  sidm_corner_<case>.png      x_c, log10 M_Lz, log10 M_vir, log10 sigma/m, truths marked
  sidm_sigma_posteriors.png   cumulative log10 sigma/m posteriors (upper limits) of the y = 0.3 and y = 1 cases (solid: their
                              injected values), plus dashed lines at the true sigma/m of the proposed CIS scan
                              injections, M_Lz in {10, 100, 1000} x x_c in {0.02, 0.2}

Run: python scripts/5_figures/plot_sidm_cross_section.py [--from-runs]
"""
import argparse
import os

import corner
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from infer_with_fiona import dm_conversions as D
from infer_with_fiona import posteriors as P

args = P.add_source_arguments(argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)).parse_args()
SOURCE = 'bilby' if args.from_runs else 'shipped'
os.makedirs(args.out_dir, exist_ok=True)
CASES = [  # short, FIONA tag, y, x_c, colour (Fig. 7)
    ('xc0.02_y0.3', 'xc0.02_y0.3', 0.3, 0.02, 'tab:red'),
    ('xc0.2_y1', 'xc0.2_y1.0', 1.0, 0.2, 'tab:blue'),
]   # y = 2, 3 left out: their x_c posteriors fill the prior, so sigma/m there is prior-driven
M_TRUE = 100.0
SCAN_M = {10.0: 'tab:purple', 100.0: 'tab:gray', 1000.0: 'goldenrod'}   # proposed scan, y = 0.3
SCAN_XC = (0.02, 0.2)
X_MIN, X_MAX = -16.0, -6.0
EXCL_Y = {'xc0.02_y0.3': 0.30, 'xc0.2_y1': 0.18}   # heights of the exclusion arrows
# Cores too small to resolve, from the run's own x_c posterior: with a flat x_c prior, p(x_c)/p(0) is the
# marginal likelihood ratio against no core (y, M_Lz, ... free to compensate). Below the x_c where it drops
# to e^-1/2 (marginal Delta chi^2 = 1) a core cannot be told from no core, so any true sigma/m there gives
# the same data and the same upper limit. (The fixed-parameter rho_diff of core_resolvability_test.py,
# job 52305406, overstates the resolving power by 3-7x in x_c because it ignores this degeneracy.)


def xc_unresolved(xc):
    k = gaussian_kde(np.concatenate([xc, -xc]), bw_method=0.08)    # reflected at the x_c = 0 prior edge
    g = np.linspace(0.0, 1.0, 1001)
    r = 2 * k(g) / (2 * k(g[:5])).mean()
    return float(g[np.argmax(r < np.exp(-0.5))])


SAME_Y = {'xc0.02_y0.3': 0.66, 'xc0.2_y1': 0.56}   # heights of the 'same limit' bars
zS = D.z_source(500.0)
zL = D.zL_pessimistic(zS)
LABELS = [r'$x_c$', r'$\log_{10}(M_{Lz}/M_\odot)$', r'$\log_{10}(M_{\rm vir}/M_\odot)$',
          r'$\log_{10}\,\langle\sigma v\rangle/m\ [{\rm cm^2\,g^{-1}}]$']
LEVELS = (1 - np.exp(-0.5), 1 - np.exp(-2))
STYLE = dict(labels=LABELS, levels=LEVELS, show_titles=False, label_kwargs=dict(fontsize=13),
             plot_density=False, plot_datapoints=False, max_n_ticks=4, bins=25, smooth=0.9,
             hist_bin_factor=10 / 25)

fig1, ax1 = plt.subplots(figsize=(7.5, 4.5))
print(f'd_L = 500 Mpc -> z_S = {zS:.4f}, pessimistic z_L = {zL:.4f}, d_eff = {D.d_eff_Gpc(zL, zS):.4f} Gpc')
for short, tag, y, xc_t, col in CASES:
    post = P.load_fiona(P.fiona_label('CIS', y, xc_t), SOURCE, args.runs_dir)[0]
    xc, mlz = np.asarray(post['lp']), np.asarray(post['MLz'])
    sig = D.sigma_over_m(np.maximum(xc, 1e-6), mlz, zL, zS)
    s = np.column_stack([xc, np.log10(mlz), np.log10(D.Mvir_sidm(mlz, zL, zS)), np.log10(sig)])
    t = [xc_t, np.log10(M_TRUE), np.log10(float(D.Mvir_sidm(M_TRUE, zL, zS))),
         np.log10(float(D.sigma_over_m(xc_t, M_TRUE, zL, zS)))]
    rng = [(0, min(1.0, np.percentile(xc, 99.5) * 1.1))] + [tuple(np.percentile(s[:, i], [0.5, 99.5])) for i in (1, 2, 3)]
    fig = corner.corner(s, color=col, fill_contours=True, range=rng,
                        hist_kwargs=dict(density=True, lw=1.8, color=col), **STYLE)
    corner.overplot_lines(fig, t, color='black', lw=1.2)
    corner.overplot_points(fig, [t], marker='s', color='black', ms=6)
    fig.suptitle(rf'SIDM from CIS: injected $y={y:g}$, $x_c={xc_t:g}$, $M_{{Lz}}=100\,M_\odot$'
                 rf' ($d_L=500$ Mpc, $z_L={zL:.3f}$)', fontsize=12, y=1.02)
    fig.savefig(os.path.join(args.out_dir, f'sidm_corner_{short}.png'), bbox_inches='tight', dpi=180)
    plt.close(fig)

    # x_c is bounded only from above (posterior flat down to the x_c = 0 prior edge), so a density in
    # log sigma/m would show a spurious lower side from the Jacobian (prior ∝ sigma^(1/2) in log sigma/m).
    # Show the cumulative posterior instead and mark the 95% upper limit.
    srt = np.sort(s[:, 3])
    cdf = np.arange(1, len(srt) + 1) / len(srt)
    ul = np.percentile(s[:, 3], 95)
    ax1.step(srt, cdf, where='post', color=col, lw=1.8,
             label=rf'$y={y:g},\ x_c={xc_t:g}$')
    ax1.plot([ul], [0.95], marker='<', color=col, ms=9)
    # everything above the 95% limit is excluded: larger sigma/m means a larger core, and the data reject
    # x_c above the limit (and x_c > 1/2 gives no multiple images), so the exclusion extends past the prior edge
    ax1.axvspan(ul, X_MAX, color=col, alpha=0.10, lw=0)
    ax1.annotate('', xy=(X_MAX - 0.05, EXCL_Y[short]), xytext=(ul, EXCL_Y[short]),
                 arrowprops=dict(arrowstyle='->', color=col, lw=1.6))
    ax1.text(ul + 0.1, EXCL_Y[short] + 0.015, rf'excluded at 95%: $\sigma/m > 10^{{{ul:.1f}}}$',
             color=col, fontsize=8, va='bottom')
    ax1.axvline(t[3], color=col, ls='-', lw=1.0, alpha=0.8)
    print(f'{short}: true log10 sigma/m = {t[3]:.2f};  95% upper limit {ul:.2f} '
          f'(x_c 95% UL {np.percentile(xc, 95):.3f}; x_c posterior has no lower bound)')
    xc_same = xc_unresolved(xc)
    s_same = np.log10(float(D.sigma_over_m(xc_same, M_TRUE, zL, zS)))
    ax1.annotate('', xy=(X_MIN + 0.05, SAME_Y[short]), xytext=(s_same, SAME_Y[short]),
                 arrowprops=dict(arrowstyle='->', color=col, lw=1.6, ls='-'))
    ax1.plot([s_same], [SAME_Y[short]], marker='|', color=col, ms=12, mew=2)
    ax1.text(X_MIN + 0.15, SAME_Y[short] + 0.015,
             rf'true $\sigma/m<10^{{{s_same:.1f}}}$ ($x_c<{xc_same:.2f}$): same signal, same limit',
             color=col, fontsize=8, va='bottom')
    print(f'    unresolved below x_c = {xc_same:.3f} (marginal Delta chi^2 = 1) -> true log10 sigma/m < {s_same:.2f} gives the same limit')
ax1.axhline(0.95, color='black', ls=':', lw=0.8)
ax1.set_ylim(0, 1.18)
print('\nproposed scan injections (y = 0.3): true log10 sigma/m')
for m, c in SCAN_M.items():
    for xc_s in SCAN_XC:
        v = np.log10(float(D.sigma_over_m(xc_s, m, zL, zS)))
        ax1.axvline(v, color=c, ls='--', lw=1.0)
        ax1.text(v, 1.17, rf'$x_c={xc_s:g}$', rotation=90, ha='right', va='top', fontsize=7, color=c)
        print(f'    M_Lz = {m:6g}, x_c = {xc_s:4g}: {v:.2f}')
handles, labels = ax1.get_legend_handles_labels()
handles += [plt.Line2D([], [], color='black', lw=1.0), plt.Line2D([], [], color='black', ls=':', lw=0.8)]
labels += ['injected value (solid)', '95%']
handles += [plt.Line2D([], [], color=c, ls='--', lw=1.0) for c in SCAN_M.values()]
labels += [rf'scan: $M_{{Lz}}={m:g}\,M_\odot$' for m in SCAN_M]
ax1.set(xlabel=LABELS[3], ylabel=r'cumulative posterior $P(<\sigma/m)$', xlim=(X_MIN, X_MAX),
        title=r'SIDM cross section from the CIS runs ($M_{Lz}=100\,M_\odot$): shaded = excluded at 95%,'
              '\nleft arrows = true values the data cannot tell apart (dashed: proposed scan)')
ax1.legend(handles, labels, fontsize=7.5, loc='upper right', bbox_to_anchor=(0.995, 0.86), framealpha=0.9)
fig1.tight_layout()
fig1.savefig(os.path.join(args.out_dir, 'sidm_sigma_posteriors.png'), dpi=180)
