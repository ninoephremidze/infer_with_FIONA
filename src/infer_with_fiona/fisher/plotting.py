"""Corner plot of Fisher-forecast ellipses, as in the paper's LISA and LIGO figures.

Used by scripts/1_fisher/fisher_forecasts.py and notebooks/01_fisher_forecasts.ipynb, so both produce the
same figure. Each source is drawn as 68% and 95% confidence ellipses (filled, with a 68% outline) for every
pair of (Delta ln M_Lz, y, x_c), with the 1D Gaussian marginals on the diagonal.
"""
import numpy as np

PARAM_LABELS = [r'$\Delta\log M_{Lz}$', r'$y$', r'$x_c$']
CHI2 = {0.68: 2.30, 0.95: 5.99}          # Delta chi^2 of the 68% and 95% contours in two dimensions


def draw_ellipse(ax, cov2, center, color, levels=(0.68, 0.95), alphas=(0.35, 0.12), lw=1.4):
    """Filled confidence ellipses of a 2x2 covariance, plus an outline of the 68% one."""
    from matplotlib.patches import Ellipse
    ev, evec = np.linalg.eigh(cov2)
    angle = np.degrees(np.arctan2(evec[1, -1], evec[0, -1]))
    for level, alpha in zip(levels, alphas):
        s = np.sqrt(CHI2[level])
        ax.add_patch(Ellipse(xy=center, width=2 * s * np.sqrt(abs(ev[-1])), height=2 * s * np.sqrt(abs(ev[-2])),
                             angle=angle, facecolor=color, alpha=alpha, linewidth=0))
    s = np.sqrt(CHI2[0.68])
    ax.add_patch(Ellipse(xy=center, width=2 * s * np.sqrt(abs(ev[-1])), height=2 * s * np.sqrt(abs(ev[-2])),
                         angle=angle, facecolor='none', edgecolor=color, linewidth=lw))


def corner_fig(covs, labels, colors, fiducial, fname=None):
    """Draw the corner plot for a list of 3x3 covariances of (ln M_Lz, y, x_c).

    fiducial = (y, x_c) of the lens. If fname is given the figure is saved there (300 dpi, tight bounding
    box), exactly as the paper's figures were made. Returns the figure.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D

    with matplotlib.rc_context({'font.size': 12, 'figure.dpi': 110}):
        pfid = [0.0, fiducial[0], fiducial[1]]
        n = 3
        max_sig = np.array([max(np.sqrt(abs(c[k, k])) for c in covs) for k in range(n)])
        half = 1.5 * np.sqrt(5.99) * max_sig
        lims = [(pfid[k] - half[k], pfid[k] + half[k]) for k in range(n)]
        fig, axes = plt.subplots(n, n, figsize=(6.8, 6.8))
        fig.subplots_adjust(hspace=0.08, wspace=0.08)
        for row in range(n):
            for col in range(n):
                ax = axes[row, col]
                if col > row:
                    ax.set_visible(False)
                    continue
                if col == row:
                    x = np.linspace(*lims[col], 500)
                    for cov, c in zip(covs, colors):
                        sig = np.sqrt(abs(cov[row, row]))
                        ax.plot(x, np.exp(-0.5 * ((x - pfid[row]) / sig) ** 2), color=c, lw=1.8)
                    ax.set_xlim(lims[col]); ax.set_ylim(0, 1.15); ax.set_yticks([])
                else:
                    for cov, c in zip(covs, colors):
                        draw_ellipse(ax, cov[np.ix_([col, row], [col, row])], (pfid[col], pfid[row]), c)
                    ax.set_xlim(lims[col]); ax.set_ylim(lims[row])
                ax.xaxis.set_major_locator(ticker.MaxNLocator(3, prune='both'))
                if row != col:
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(3, prune='both'))
                ax.tick_params(labelsize=10, length=3, width=0.7)
                if row == n - 1:
                    ax.set_xlabel(PARAM_LABELS[col], fontsize=13)
                else:
                    ax.tick_params(labelbottom=False)
                if col == 0 and row > 0:
                    ax.set_ylabel(PARAM_LABELS[row], fontsize=13)
                elif col > 0:
                    ax.tick_params(labelleft=False)
        fig.legend(handles=[Line2D([0], [0], color=c, lw=2.2, label=l) for l, c in zip(labels, colors)],
                   loc='upper right', bbox_to_anchor=(0.95, 0.93), fontsize=11, framealpha=0.9, handlelength=1.6)
        if fname is not None:
            fig.savefig(fname, bbox_inches='tight', dpi=300)
    return fig
