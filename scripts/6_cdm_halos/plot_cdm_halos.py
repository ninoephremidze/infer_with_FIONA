"""Paper figure (FIONA version): NFW halos on the CDM c-M relation (Diemer & Joyce 2019).
Inputs: fiona_cdm_LIGO_BBH_30_{steep_core,cis_x2}.npz (fiona_cdm.py) and fiona_cdm_rate_post_<NAME>.npz
(fiona_cdm_rate_post.py). Writes cdm_nfw_fisher.{png,pdf} here (copy to prd_draft/figs/).

(a) Delta chi^2 vs M200 for a halo halfway to the source (comoving): LISA with the binary's intrinsic parameters
    refit, LIGO with amplitude, phase and time refit; lines b = r_F, bands 0.3-3 r_F; dashed lines: the
    likelihood-ratio thresholds 14.2 (3 sigma) and 31.8 (5 sigma) for three lens parameters.
(b) Largest detectable impact parameter b_max / r_F (LISA, halfway), likelihood-ratio 3 sigma (solid) and Bayes
    factor B >= 100 (dashed).
(c) sigma(ln c200) at fixed M200, intrinsic parameters marginalized, LISA, b = r_F; drawn only where detected (3 sigma);
    dashed line: CDM c-M scatter, sigma(ln c200) = 0.16 ln 10 = 0.37.
Only points converged between the two FIONA quadrature schedules are drawn.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from infer_with_fiona import paths
D = str(paths.DATA_DIR / 'cdm')                 # post-processed results
FIG = str(paths.FIGURES_DIR)
PUB = {'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'xtick.direction': 'in',
       'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True}
LISA = [('SMBHB_1e4', r'LISA, $10^4+10^4\,M_\odot$', 'tab:blue'),
        ('SMBHB_1e5', r'LISA, $10^5+10^5\,M_\odot$', 'tab:green'),
        ('SMBHB_1e6', r'LISA, $10^6+10^6\,M_\odot$', 'tab:cyan' if os.environ.get('TAG', '').startswith('_imr') else 'tab:red')]
if os.environ.get('TAG', '') == '_imr':
    LISA = LISA + [('SMBHB_1e7', r'LISA, $10^7+10^7\,M_\odot$', 'navy')]   # colours as in Fig. sidm_probes
M_BC = 3.3e10 if os.environ.get('TAG', '') == '_imr' else np.inf   # panels (b), (c): M200 range quoted in the text
THR3, THR5 = 14.2, 31.8
import os
IMR_SNR = {'SMBHB_1e4': 51, 'SMBHB_1e5': 301, 'SMBHB_1e6': 1041, 'SMBHB_1e7': 158}  # native SNR, full IMR band (_imr run)
TAG = os.environ.get('TAG', '')               # '_native': LISA at its native SNR (NATIVE_SNR run of the post step)
L1 = np.load(f'{D}/fiona_cdm_LIGO_BBH_30_steep_core.npz')
L2 = np.load(f'{D}/fiona_cdm_LIGO_BBH_30_cis_x2.npz')
lconv = np.abs(L1['dchi2'] - L2['dchi2']) <= 0.02 * np.abs(L2['dchi2'])

with plt.rc_context(PUB):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.0, 2.4))
    M = L1['M200']; d = np.where(lconv, L1['dchi2'], np.nan)
    a1.fill_between(M, d[:, 2], d[:, 0], color='0.25', alpha=0.18, lw=0)
    a1.plot(M, d[:, 1], color='0.25', lw=1.5, label=r'LIGO, $30+30\,M_\odot$' + (r' ($\varrho=32$)' if TAG in ('_native', '_imr') else ''))
    for name, lab, col in LISA:
        P = np.load(f'{D}/fiona_cdm_rate_post_{name}{TAG}.npz')
        if TAG in ('_native', '_imr'):
            snr = IMR_SNR[name] if TAG == '_imr' else float(np.load(f"{D}/native_snr.npz")[name])
            lab = lab + rf' ($\varrho={snr:.0f}$)'
        M, b = P['M200'], list(np.round(P['b_rF'], 3))
        iz = int(np.argmin(np.abs(P['z_L'] - P['z_L'][len(P['z_L']) // 2])))
        k1, k0, k3 = b.index(1.0), b.index(0.3), b.index(3.0)
        dd = np.where(P['conv'][iz], P['d_int'][iz], np.nan)
        a1.fill_between(M, dd[:, k3], dd[:, k0], color=col, alpha=0.18, lw=0)
        a1.plot(M, dd[:, k1], color=col, lw=1.5, label=lab)
        for crit, ls in (('LRT_3sig', '-'), ('Bayes_B100', '--')):
            bm = P[f'{crit}__b_max'][iz].astype(float)
            fl = P[f'{crit}__flag'][iz]
            bm = np.where((fl == 9) | (bm <= 0) | (M > M_BC), np.nan, bm)
            a2.plot(M, bm, color=col, lw=1.5 if ls == '-' else 1.1, ls=ls)
        s = np.where(P['conv_sig'][iz, :, k1], P['sig'][iz, :, k1, 1, 2], np.nan)
        det = dd[:, k1] >= THR3
        if (det & (M <= M_BC)).any():   # converged trend below 3 sigma, from 1e9 Msun (the range of panel b)
            a3.plot(M, np.where((M <= M_BC) & (M >= 1e9), s, np.nan), color=col, lw=1.1, ls=':')
        a3.plot(M, np.where(det & (M <= M_BC), s, np.nan), color=col, lw=1.8, marker='o', ms=2.5, label=lab)
    for t, tl in ((THR3, r'$3\sigma$'), (THR5, r'$5\sigma$')):
        a1.axhline(t, color='k', lw=0.7, ls='--')
        a1.text(6.0, t / 1.4 if t == THR3 else t * 1.4, tl, fontsize=6.5, va='top' if t == THR3 else 'bottom')
    a1.set(xscale='log', yscale='log', xlim=(1, 1e12), ylim=(1e-8, 1e3) if TAG == '_imr' else (1e-8, 1e5), xlabel=r'$M_{200}\ [M_\odot]$',
           ylabel=r'$\Delta\chi^2$')
    a1.set_xticks([1, 1e3, 1e6, 1e9, 1e12])
    a1.plot([], [], color='0.25', lw=1.5)
    a2.plot([], [], color='k', ls='-', label=r'$3\sigma$ likelihood ratio')
    a2.plot([], [], color='k', ls='--', lw=1.1, label=r'Bayes factor $\geq100$')
    a2.set(xscale='log', yscale='log', xlim=(5e8, 2e11) if TAG == '_imr' else (1e7, 1e12), ylim=(0.1, 5) if TAG == '_imr' else (0.1, 20), xlabel=r'$M_{200}\ [M_\odot]$',
           ylabel=r'$b_{\max}/r_F$')
    a2.legend(frameon=False, loc='lower right' if TAG == '_imr' else 'upper left', handlelength=1.8, bbox_to_anchor=(1.0, 0.1) if TAG == '_imr' else None)
    a3.axhline(0.16 * np.log(10), color='k', lw=0.7, ls='--', label='_thr')   # CDM c-M scatter, 0.16 in log10 (Diemer & Joyce 2019)
    a3.set(xscale='log', yscale='log', xlim=(5e8, 2e11) if TAG == '_imr' else (1e8, 1e12), ylim=(0.1, 1) if TAG == '_imr' else (0.02, 1), xlabel=r'$M_{200}\ [M_\odot]$',
           ylabel=r'$\sigma(\ln c_{200})$ at fixed $M_{200}$')
    if TAG == '_imr':   # one decade: label 0.1, 0.2, 0.5, 1 instead of every minor tick
        from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
        a3.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1.0]))
        a3.yaxis.set_major_formatter(FixedFormatter(['0.1', '0.2', '0.5', '1']))
        a3.yaxis.set_minor_formatter(NullFormatter())
    if TAG == '_imr':   # axis ranges from the plotted data (log padding 15%)
        def fit(ax, axis='both'):
            xs, ys = [], []
            for ln in ax.get_lines():
                x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
                ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                if ok.sum() and ln.get_label() != '_thr':
                    xs += list(x[ok]); ys += list(y[ok])
            pad = lambda lo, hi: (lo / (hi / lo) ** 0.08, hi * (hi / lo) ** 0.08)
            if axis in ('both', 'x'): ax.set_xlim(*pad(min(xs), max(xs)))
            if axis in ('both', 'y'): ax.set_ylim(*pad(min(ys), max(ys)))
        fit(a2); fit(a3)
        lo = min(a2.get_xlim()[0], a3.get_xlim()[0]); hi = max(a2.get_xlim()[1], a3.get_xlim()[1])
        a2.set_xlim(lo, hi); a3.set_xlim(lo, hi)
        from matplotlib.ticker import FixedLocator as FL, FixedFormatter as FF, NullFormatter as NF
        for ax_, tk in ((a2, [0.2, 0.3, 0.5, 1.0, 1.5]), (a3, [0.2, 0.3, 0.5, 1.0, 2.0])):
            y0, y1 = ax_.get_ylim(); tk = [t for t in tk if y0 <= t <= y1]
            ax_.yaxis.set_major_locator(FL(tk)); ax_.yaxis.set_major_formatter(FF([f'{t:g}' for t in tk]))
            ax_.yaxis.set_minor_formatter(NF())
        a1.set_ylim(1e-8, 3e2)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=c, lw=1.8) for _, _, c in LISA]
    ls_ = [h.get_label() for h in a3.get_lines() if not h.get_label().startswith('_')]
    fig.legend([Line2D([], [], color='0.25', lw=1.5)] + hs, [r'LIGO, $30+30\,M_\odot$' + (r' ($\varrho=32$)' if TAG in ('_native', '_imr') else '')] + ls_, frameon=False,
               loc='upper center', ncol=3 if len(LISA) > 3 else 4, bbox_to_anchor=(0.5, 1.12 if len(LISA) > 3 else 1.07), handlelength=1.6)
    for ax, t in ((a1, '(a)'), (a2, '(b)'), (a3, '(c)')):
        up = TAG == '_imr' and t != '(a)'   # (b), (c): lines fill the lower right
        left = up and t == '(b)'
        ax.text(0.03 if left else 0.97, 0.95 if up else 0.05, t, transform=ax.transAxes,
                ha='left' if left else 'right', va='top' if up else 'bottom', fontsize=8)
    fig.tight_layout(w_pad=1.0)
    os.makedirs(FIG, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{FIG}/cdm_nfw_fisher{TAG}.{ext}', dpi=300, bbox_inches='tight')
print(f'wrote {FIG}/cdm_nfw_fisher{TAG}.png')
