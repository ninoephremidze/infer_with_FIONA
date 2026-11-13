#!/usr/bin/env python
"""Publication figure: lensed and unlensed waveforms of the Sec. III Fisher-forecast sources (full IMRPhenomD band).

Left LISA (z_S = 3, last month before merger), right LIGO (z_S = 0.1). Characteristic strain 2 f |h(f)| at native
amplitude, unlensed (dashed) and lensed (solid) by the CIS (y = 0.3, x_c = 0.01) at each source's paired lens mass
(w = 10 at f_*), against the detector noise sqrt(f S_n). Lens masses from data/fisher_covariances.npz.

  python scripts/5_figures/plot_fisher_waveforms.py compute   (cluster node: FIONA) -> data/fisher_waveforms.npz
  python scripts/5_figures/plot_fisher_waveforms.py plot      -> outputs/figures/fisher_waveforms_imr.{png,pdf}
"""
import os
import sys
import numpy as np

from infer_with_fiona import paths
DATA = str(paths.DATA_DIR / 'fisher_waveforms.npz')
D = str(paths.FIGURES_DIR)
Y, XC = 0.3, 0.01
FMIN = {'SMBHB_1e8': 1e-5}   # 10^8+10^8 merges below 1e-4 Hz: band floor lowered to 1e-5 Hz for it only
SRC = {   # det: (z_S, f_min, f_max, sources: name, m1, m2, label, colour (as in the Fisher figures), pycbc delta_f)
    'LISA': (3.0, 1e-4, 1.0, [
        ('SMBHB_1e4', 1e4, 1e4, r'$10^4+10^4\,M_\odot$', '#EEA06C', 1e-7),
        ('SMBHB_1e5', 1e5, 1e5, r'$10^5+10^5\,M_\odot$', '#E0727F', 1e-7),
        ('SMBHB_1e6', 1e6, 1e6, r'$10^6+10^6\,M_\odot$', '#B95E99', 1e-7),
        ('SMBHB_1e7', 1e7, 1e7, r'$10^7+10^7\,M_\odot$', '#7D5AA6', 1e-8),
        ('SMBHB_1e8', 1e8, 1e8, r'$10^8+10^8\,M_\odot$', '#3F5791', 1e-8)]),
    'LIGO': (0.1, 10.0, 1000.0, [
        ('BNS', 1.4, 1.4, r'$1.4+1.4\,M_\odot$', '#EEA06C', 1e-3),
        ('NSBH', 10.0, 1.4, r'$10+1.4\,M_\odot$', '#E0727F', 1 / 256),
        ('BBH_10', 10.0, 10.0, r'$10+10\,M_\odot$', '#B95E99', 1 / 64),
        ('BBH_30', 30.0, 30.0, r'$30+30\,M_\odot$', '#7D5AA6', 1 / 16),
        ('BBH_60', 60.0, 60.0, r'$60+60\,M_\odot$', '#3F5791', 1 / 4)]),
}


def compute():
    import finufft  # noqa: F401 -- before FIONA
    from pycbc.waveform import get_fd_waveform
    from infer_with_fiona import quadrature
    quadrature.get_schedule('cis').apply()
    from infer_with_fiona.fisher import lisa_psd, ligo_o4_psd, f_1month, luminosity_distance
    from infer_with_fiona.fisher.lensing import compute_F_and_derivatives
    from infer_with_fiona.fisher.utils import G_over_c3_Msun as T
    C = np.load(paths.FISHER_COVARIANCES)
    nw = int(os.environ.get('SLURM_CPUS_ON_NODE', 8))
    psds = {'LISA': lisa_psd, 'LIGO': ligo_o4_psd}
    out = {}
    for det, (z, fmin, fmax, srcs) in SRC.items():
        fn = np.geomspace(1e-5, 1, 600) if det == 'LISA' else np.geomspace(10, 1000, 2000)
        out[f'{det}_fn'], out[f'{det}_hn'] = fn, np.sqrt(fn * psds[det](fn))
        for name, m1, m2, *_ , df in srcs:
            M = m1 + m2
            flo = max(f_1month(M, m1 * m2 / M ** 2, z), FMIN.get(name, fmin))
            if f'{det}_{name}_mlz' in C:                                  # sources of the Fisher forecast
                M_Lz = float(C[f'{det}_{name}_mlz'])
            else:                                                         # same rule: w = 10 at f_* (90% of SNR^2)
                fb = min(0.2 / (M * (1 + z) * T), fmax)
                hq, _ = get_fd_waveform(approximant='IMRPhenomD', mass1=m1 * (1 + z), mass2=m2 * (1 + z),
                                        f_lower=flo, delta_f=df, distance=luminosity_distance(z))
                fg = np.geomspace(flo, fb, 20000)
                dn = np.interp(fg, hq.sample_frequencies.numpy(), np.abs(hq.numpy()), left=0, right=0) ** 2 / psds[det](fg)
                cm = np.concatenate([[0.0], np.cumsum(0.5 * (dn[1:] + dn[:-1]) * np.diff(fg))])
                M_Lz = 10.0 / (8 * np.pi * T * float(np.interp(0.9 * cm[-1], cm, fg)))
            fhi = min(0.2 / (M * (1 + z) * T), fmax, 250.0 / (8 * np.pi * T * M_Lz))   # as in the forecast
            hp, _ = get_fd_waveform(approximant='IMRPhenomD', mass1=m1 * (1 + z), mass2=m2 * (1 + z), f_lower=flo,
                                    delta_f=df, distance=luminosity_distance(z))
            f = np.geomspace(flo, fhi, 2048)
            h = np.interp(f, hp.sample_frequencies.numpy(), np.abs(hp.numpy()), left=0, right=0)
            F = compute_F_and_derivatives(M_Lz, np.array([Y]), XC, f, lens_model='CIS', n_workers=nw)[0][:, 0]
            out[f'{det}_{name}_f'], out[f'{det}_{name}_hc'] = f, 2 * f * h
            out[f'{det}_{name}_absF'], out[f'{det}_{name}_mlz'] = np.abs(F), M_Lz
            if det == 'LISA' and flo > 1.01e-5:   # earlier inspiral, outside the analyzed band (display only)
                hx, _ = get_fd_waveform(approximant='IMRPhenomD', mass1=m1 * (1 + z), mass2=m2 * (1 + z),
                                        f_lower=1e-5, delta_f=min(df, 1e-8), distance=luminosity_distance(z))
                fx = np.geomspace(1e-5, flo, 512)
                hxa = np.interp(fx, hx.sample_frequencies.numpy(), np.abs(hx.numpy()), left=0, right=0)
                Fx = compute_F_and_derivatives(M_Lz, np.array([Y]), XC, fx, lens_model='CIS', n_workers=nw)[0][:, 0]
                out[f'{det}_{name}_fx'], out[f'{det}_{name}_hcx'], out[f'{det}_{name}_absFx'] = fx, 2 * fx * hxa, np.abs(Fx)
            print(det, name, f'M_Lz {M_Lz:.3g}', flush=True)
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez(DATA, **out)
    print(f'wrote {DATA}')


def sci(x):
    e = int(np.floor(np.log10(x)))
    return rf'{x / 10 ** e:.1f}\times10^{{{e}}}'


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d = np.load(DATA)
    PUB = {'font.size': 7, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 5.5, 'xtick.direction': 'in', 'ytick.direction': 'in',
           'xtick.top': True, 'ytick.right': True, 'axes.linewidth': 0.6}
    with plt.rc_context(PUB):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
        for ax, (det, (z, fmin, fmax, srcs)) in zip(axes, SRC.items()):
            ax.loglog(d[f'{det}_fn'], d[f'{det}_hn'], color='0.55', lw=1.0)
            hs, ls_ = [], []
            for name, m1, m2, lab, col, df in srcs:
                f, hc, aF = d[f'{det}_{name}_f'], d[f'{det}_{name}_hc'], d[f'{det}_{name}_absF']
                ax.loglog(f, hc, color=col, lw=1.0, ls='--')
                ax.loglog(f, hc * aF, color=col, lw=1.3)
                if f'{det}_{name}_fx' in d:   # outside the analyzed band (before the last month, or below 1e-4 Hz)
                    ax.loglog(d[f'{det}_{name}_fx'], d[f'{det}_{name}_hcx'], color=col, lw=0.9, ls='--')
                    ax.loglog(d[f'{det}_{name}_fx'], d[f'{det}_{name}_hcx'] * d[f'{det}_{name}_absFx'], color=col,
                              lw=1.0, ls=':')
                hs.append(Line2D([], [], color=col, lw=1.4))
                ls_.append(rf'{lab}, $M_{{Lz}}={sci(float(d[f"{det}_{name}_mlz"]))}\,M_\odot$')
            ax.set_xlabel(r'$f\ [{\rm Hz}]$')
            ax.legend(hs, ls_, frameon=False, loc='upper right', ncol=1, handlelength=1.4, borderaxespad=0.4,
                      labelspacing=0.3)
            if det == 'LISA':
                ax.set(xlim=(1e-5, 1), ylim=(1e-21, 3e-15))
                ax.set_ylabel(r'strain $2f\,|\tilde h(f)|$')
            else:   # shared line-style key in the empty lower left of the LIGO panel
                ax.set(xlim=(10, 1000), ylim=(1e-24, 1e-18))
                key = [Line2D([], [], color='0.25', lw=1.1), Line2D([], [], color='0.25', lw=0.9, ls='--'),
                       Line2D([], [], color='0.25', lw=1.0, ls=':'), Line2D([], [], color='0.55', lw=1.0)]
                ax.add_artist(ax.get_legend())
                ax.legend(key, ['lensed', 'unlensed', 'lensed, outside the band', 'noise'], frameon=False, loc='lower left', handlelength=1.6)
        fig.subplots_adjust(wspace=0.25)
        os.makedirs(D, exist_ok=True)
        for ext in ('png', 'pdf'):
            fig.savefig(f'{D}/fisher_waveforms_imr.{ext}', dpi=300, bbox_inches='tight')
    print('wrote fisher_waveforms_imr.png/pdf')


compute() if sys.argv[1] == 'compute' else plot()
