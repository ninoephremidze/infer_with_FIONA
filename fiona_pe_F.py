"""F(w; x_c, y) for the CIS PE scripts: cached FIONA table + GO residual + lens-dependent bGO
switch, as validated in validate_w_table.py (job 48205398) and analyze_w_table_validation.ipynb.

Used by PE_CIS_glworia_reproduction.py and PE_CIS_glworia_reproduction_simplepriors.py.
Does NOT modify anything under 0-fiona/.

WHAT IT DOES
------------
1. Applies the v4 Gauss-Legendre schedule (fiona_adaptive_gl_v4) and seeds its node cache.
2. Builds the (x_c, y, w) grids. w runs from 2e-4 (below the prior's lowest w, M_Lz = 0.1 at
   20 Hz = 2.48e-4) to just above the prior's highest w (M_Lz_max at f_max).
3. Loads the table from the on-disk cache (fiona_w_table_cache/, keyed by a hash of the full
   configuration) into shared memory, or builds it (~46 min on 112 cores at 201^2), turns it
   into the GO residual, and saves it.
4. F(w, x_c, y) = fiona_w_table.F_hybrid: table (bilinear in x_c, y; linear in w; exact GO
   added back) blended smoothly into bGO above w_s(x_c, y).

ACCURACY (validation, 130 test combinations outside the prior's caustic mask, w <= 254)
------------------------------------------------------------------------------------
worst 1.02%, 95th percentile 0.21%, median 0.032% of median |F|, against direct converged
FIONA. INSIDE the caustic band |y - y_crit(x_c)| < 0.15 the GO residual is not valid (errors
up to ~26% at the masked test points), because the geometric-optics magnification diverges
there. Both CIS PE scripts exclude that band with glworia's exact mask
(fiona_w_table.cis_prior_masked): the production script via glworia's joint prior, the
_simplepriors variant via a bilby Constraint. The table's row trimming uses the same mask,
so a query outside it never reaches a trimmed entry.

TIME REFERENCE (added 2026-09-28)
---------------------------------
FIONA's F carries the absolute Fermat-potential phase of the images, so the whole lensed
signal is shifted in time by dt = 4 G M_Lz phi_min(y, x_c) / c^3, which depends on the lens
parameters. glworia measures times from the minimum (first-arriving) image instead. In the PE
(which samples H1_time explicitly: bilby cannot marginalise over a non-geocentre time) the
absolute convention made H1_time exactly degenerate with dt(M_Lz, y, x_c) -- ~3.5-4x more
likelihood calls per iteration, and the +-0.1 s H1_time prior cut off lens configurations with
|dt| > 100 ms (analyze_sampler_efficiency.ipynb). By default HybridF therefore returns
F * exp(-i w phi_min), glworia's convention; time_reference='absolute' keeps the old one.
Physically nothing changes (the absolute lensing delay is unobservable). The table itself is
unchanged, so cached tables and their validation still apply.

FOURIER CONVENTION (added 2026-10-07)
-------------------------------------
FIONA computes F in the lensing literature's convention, F = (w/2 pi i) Int d^2x e^{+i w T}
(images: sqrt|mu| e^{+i w T_j - i pi n_j}). LAL/bilby frequency-domain waveforms use
h(f) = Int h(t) e^{-2 pi i f t} dt, in which a delay dt multiplies h by e^{-2 pi i f dt}. Since
the lensing kernel is real in the time domain, F in LAL's convention is the complex conjugate
F*. Multiplying a LAL waveform by F itself (as glworia does, and as these scripts did before
this date) places later images EARLIER and flips the Morse phase -- a self-consistent but
unphysical lensed signal (pointed out by M. Cheung, 2026-10-07; check_fourier_convention.py).
HybridF therefore returns conj(F) by default (fourier_convention='lal'); 'fiona' returns F as
computed, for comparisons with glworia's (uncorrected) convention. The time-reference factor
is applied before the conjugation, so it conjugates with F, as it must.

PHASE / CONSTANT CONVENTION
---------------------------
The table is indexed by true w. Callers pass w computed with glworia's Mtow (the PE scripts'
MTOW_GLWORIA), so the w = 8 pi G M_Lz f / c^3 convention matches glworia exactly; FIONA's own
G/c^3 (1.8e-4 relative difference) is used only to generate table nodes at the right w.
"""
import importlib
import os

import numpy as np

import fiona_w_table as FWT

# Defaults used by the validated, cached table (fiona_w_table_cache/07cd8bb159006ece).
DEFAULTS = dict(gl_schedule='v4', dw=0.02, n_per_decade=100, w_split=100.5, dw_high=0.05,
                go_w_ref=1.0, switch_path='bgo_switch_fine.npz', switch_eps=1e-3,
                cache_dir='fiona_w_table_cache', time_reference='min_image',
                fourier_convention='lal', lens_model='CIS')


def phi_min_image(xc, y, lens_model='CIS'):
    """Fermat potential of the minimum (first-arriving, Morse index 0) image of the lens
    (x_c is the lens's shape parameter: CIS x_c, ULA r_s, point-lens softening)."""
    phi, mu, n = FWT.image_set(lens_model, xc, y)
    return float(phi[n == 0].min()) if np.any(n == 0) else float(phi.min())


def time_reference_factor(w, xc, y, time_reference, lens_model='CIS'):
    """Phase factor that moves F to the requested time reference ('min_image' or 'absolute')."""
    if time_reference == 'absolute':
        return 1.0
    if time_reference != 'min_image':
        raise ValueError(f'unknown time_reference {time_reference!r}')
    return np.exp(-1j * np.asarray(w, dtype=float) * phi_min_image(xc, y, lens_model))


def to_fourier_convention(F, fourier_convention):
    """Express F (computed in FIONA's e^{+i w T} convention) in the requested Fourier
    convention: 'lal' -> conj(F), for multiplying LAL/bilby waveforms (h(f) = Int h e^{-2 pi i f t});
    'fiona' -> F unchanged."""
    if fourier_convention == 'lal':
        return np.conj(F)
    if fourier_convention != 'fiona':
        raise ValueError(f'unknown fourier_convention {fourier_convention!r}')
    return F


def table_w_max(mlz_max, f_max):
    """Highest w the sampler can request (FIONA convention, >= glworia's), rounded up with
    >= 0.1 margin. For M_Lz_max = 2000 and f_max = 1024 Hz this is 253.7, the cached value."""
    return float(np.ceil(FWT.W_TO_MF * mlz_max * f_max * 10.0 + 1.0) / 10.0)


def table_w_min(mlz_min, f_min=20.0):
    """Lowest tabulated w. 2e-4 for glworia's full prior (M_Lz >= 0.1, lowest w 2.48e-4), so
    the default runs keep hitting the validated cache; for narrower priors (lowest w above
    1e-3) the table starts just below the lowest reachable w, skipping the very expensive
    low-w GL tiers (up to 1,024,000 nodes) that the prior can never use."""
    w_lo = FWT.W_TO_MF * mlz_min * f_min
    return 2.0e-4 if w_lo < 1.0e-3 else 0.8 * w_lo


class HybridF:
    """Callable F(w_vec, x_c, y) -> complex array, built once in the parent process (before
    bilby forks its sampler pool, so workers share the table's shared-memory pages).

    lens_model (default 'CIS') selects the lens: 'ULA' (x_c axis = soliton core radius r_s,
    caustic band masked as for CIS, see fiona_w_table.ula_prior_masked) or 'PointLens' (a
    single shape node at the fixed softening xc_min = xc_max; no mask). The CIS configuration,
    and so its cache key, is unchanged."""

    def __init__(self, xc_min, xc_max, y_min, y_max, n_grid, mlz_min, mlz_max, f_max, gl_dir,
                 n_workers, alloc=None, root='.', **kw):
        o = dict(DEFAULTS, **kw)
        self.opts = o
        gl = importlib.import_module(f'fiona_adaptive_gl_{o["gl_schedule"]}')
        gl.seed_gl_cache(gl_dir, n_values=tuple(gl.REQUIRED_N_GL))
        gl.apply()      # before make_w_grid / table_config: both read the active schedule

        self.lens_model = lm = o['lens_model']
        if lm == 'PointLens':
            self.xc_grid = np.array([float(xc_min)])
        else:
            self.xc_grid = np.linspace(max(xc_min, 1.0e-6), xc_max, n_grid)
        self.y_grid = np.linspace(y_min, y_max, n_grid)
        self.w_max = table_w_max(mlz_max, f_max)
        self.w_grid = FWT.make_w_grid(w_min=table_w_min(mlz_min), dw=o['dw'],
                                      n_per_decade=o['n_per_decade'], w_max=self.w_max,
                                      w_split=o['w_split'], dw_high=o['dw_high'])
        self.switch = FWT.BgoSwitch(os.path.join(root, o['switch_path']), eps=o['switch_eps'])
        w_max_rows = FWT.table_row_wmax(self.xc_grid, y_min, y_max, self.switch, self.w_max,
                                        masked=FWT.prior_masked_fn(lm))
        cfg = FWT.table_config(self.xc_grid, self.y_grid, self.w_grid, go_w_ref=o['go_w_ref'],
                               w_max_rows=w_max_rows,
                               extra=None if lm == 'CIS' else dict(lens_model=lm))
        cache = os.path.join(root, o['cache_dir']) if o['cache_dir'] else None
        print(f'[F] {lm} table {len(self.xc_grid)}x{len(self.y_grid)}x{len(self.w_grid)}, '
              f'w in [{self.w_grid[0]:.3g}, {self.w_max:g}], GL {o["gl_schedule"]}, '
              f'GO residual w_ref={o["go_w_ref"]}, switch {o["switch_path"]} '
              f'(eps={o["switch_eps"]:g}), time reference {o["time_reference"]}, '
              f'Fourier convention {o["fourier_convention"]}, '
              f'cache key {cfg["key"]}', flush=True)

        interp = FWT.load_table(cache, cfg, alloc=alloc) if cache else None
        if interp is None:
            print('[F] no cached table for this configuration: building it '
                  '(~46 min on 112 cores at 201^2)', flush=True)
            table = FWT.build_F_w_table(self.xc_grid, self.y_grid, self.w_grid, gl_dir,
                                        n_workers=n_workers, alloc=alloc,
                                        w_max_rows=w_max_rows, lens_model=lm)
            interp = FWT.FwInterpolator(self.xc_grid, self.y_grid, self.w_grid, table,
                                        lens_model=lm)
            if o['go_w_ref'] is not None:
                FWT.subtract_go_reference(interp, w_ref=o['go_w_ref'], n_workers=n_workers)
            if cache:
                FWT.save_table(cache, interp, cfg)
        self.interp = interp

    def __call__(self, w, xc, y):
        F = (FWT.F_hybrid(w, xc, y, self.interp, self.switch)
             * time_reference_factor(w, xc, y, self.opts['time_reference'], self.lens_model))
        return to_fourier_convention(F, self.opts['fourier_convention'])


def F_hybrid_warn_unmasked():   # for a prior that does NOT exclude the caustic band
    print('[F] WARNING: this prior does not mask the caustic band |y - y_crit(x_c)| < 0.15. '
          'The F table is validated only outside that band (errors up to ~26% inside it).',
          flush=True)


def self_check(F, M_Lz, xc, y, freqs, mtow, gl_dir, n_workers, warn_above=0.01):
    """Compare F (a HybridF) with direct FIONA at one (M_Lz, x_c, y) over `freqs`, using the
    same w values (w = mtow * M_Lz * f) for both. Returns max |dF| / median |F_direct|.
    Direct FIONA uses the GL schedule HybridF applied (v4)."""
    import time
    from fiona.axisym import FresnelNUFHTLensBatched
    freqs = np.asarray(freqs, dtype=float)
    w = mtow * M_Lz * freqs
    w = w[(w > 0) & (w <= F.w_max)]
    t0 = time.time()
    lm = getattr(F, 'lens_model', 'CIS')
    solver = FresnelNUFHTLensBatched([FWT.make_lens(lm, xc)], gl_dir=gl_dir,
                                     n_workers=n_workers, print_timing=False)
    F_dir = solver(w, np.array([float(y)]))[:, 0, 0]
    F_dir = to_fourier_convention(F_dir * time_reference_factor(w, xc, y, F.opts['time_reference'], lm),
                                  F.opts['fourier_convention'])                # same convention
    err = float(np.max(np.abs(F(w, xc, y) - F_dir)) / np.median(np.abs(F_dir)))
    print(f'[F] self-check vs direct FIONA ({lm}) at M_Lz={M_Lz:g}, shape={xc:g}, y={y:g} '
          f'({len(w)} bins, w <= {w.max():.3g}): max |dF| / median |F| = {err:.3%} '
          f'({time.time()-t0:.0f}s)', flush=True)
    if err > warn_above:
        print(f'[F] WARNING: self-check error {err:.2%} exceeds {warn_above:.0%}', flush=True)
    return err
