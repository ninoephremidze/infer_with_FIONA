"""F(w, x_c, y) interpolation table for the CIS lens, tabulated directly in w.

Prototype replacement for the (logM, x_c, y, f) table built in PE_CIS_glworia_reproduction.py.
Does NOT modify anything under 0-fiona/.

WHY
---
FIONA's solver only ever sees w = 8 pi (G/c^3) M_Lz f: compute_F_param_batch forms w_vec
from (M_Lz, f) and never uses either separately. The production table nonetheless stores
F on the 2D (logM, f) grid -- 121 x 4097 = 495,737 w-samples per (x_c, y) node, 298 GiB at
grid-res=high -- and each logM row is essentially the same F(w) curve shifted in log w.

It also interpolates between logM rows at fixed f, i.e. in steps of dln(w) = ln(2000)/120
= 0.063. Near w ~ 100 that is dw ~ 6, comparable to the fringe period of F(w), which is the
likely origin of the ~650% M_Lz-interpolation error interp_error_scan.py found near
M_Lz ~ 800.

Tabulating F on a single w-grid instead gives a table of shape (n_xc, n_y, n_w) with
n_w ~ a few thousand: ~3 GiB at 201 x 201, and no M_Lz interpolation at all. The only
remaining interpolation along w is 1D and its step dw is chosen directly.

W GRID
------
Log-spaced from w_min to w_switch (F is smooth and slowly varying there), then uniform with
step dw from w_switch to w_max, plus a node pair straddling each GL-schedule tier boundary
(F from FIONA is slightly discontinuous there; see schedule_breakpoints). Default w_max slightly exceeds W_HIGH=100 so that queries at
w=100 are bracketed; above W_HIGH production uses the closed-form bGO sum instead.

INTERPOLATION
-------------
Bilinear in (x_c, y) over the four bracketing grid nodes -> one F(w) vector, then linear
in w (np.interp, real and imaginary parts). Equivalent to trilinear interpolation in
(x_c, y, w). Queries outside the table are clamped to its edges; callers must route
w > W_HIGH to bGO themselves.

CONSTANT CONVENTION
-------------------
The table is indexed by FIONA's own w (fisherWithFIONA.utils.G_over_c3_Msun). glworia's
Mtow differs by -1.76e-4 relative. `w_of(M_Lz, f)` below uses FIONA's convention, which is
what the old (logM, f) table effectively stored too (compute_F_param_batch recomputes w
from M_Lz internally).

IMPORTANT: `finufft` must be imported before any fisherWithFIONA/fiona import.
"""
import json
import time

import finufft  # noqa: F401 -- must precede any fisherWithFIONA/fiona import
import numpy as np

from fisherWithFIONA import compute_F_param_batch
from fisherWithFIONA.utils import G_over_c3_Msun

W_TO_MF = 8.0 * np.pi * G_over_c3_Msun   # w = W_TO_MF * M_Lz[Msun] * f[Hz]


def w_of(M_Lz, f):
    """FIONA-convention dimensionless frequency."""
    return W_TO_MF * M_Lz * np.asarray(f, dtype=float)


def schedule_breakpoints(w_min=2.0e-4, w_max=100.5, n_gl_fn=None, n_scan=20000):
    """w values where the active adaptive GL schedule changes node count.

    FIONA's F(w) jumps slightly (at the ~1% self-convergence level) where n_gl changes tier,
    so a linear-interpolation cell straddling a tier boundary blends two different
    solutions. make_w_grid places a node on each side of every breakpoint to avoid that.
    Found by scanning a log grid and bisecting each change, so it follows whatever schedule
    is active (default: fiona.axisym._adaptive_n_gl, e.g. after fiona_adaptive_gl_v3.apply()).
    """
    if n_gl_fn is None:
        import fiona.axisym as _axisym
        n_gl_fn = _axisym._adaptive_n_gl
    ws = np.logspace(np.log10(w_min), np.log10(w_max), n_scan)
    ns = np.array([n_gl_fn(w) for w in ws])
    bps = []
    for i in np.nonzero(np.diff(ns))[0]:
        lo, hi = ws[i], ws[i + 1]
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if n_gl_fn(mid) == ns[i] else (lo, mid)
        bps.append(hi)                      # smallest w with the new node count (to ~1 ulp)
    return np.array(bps)


def make_w_grid(w_min=2.0e-4, w_switch=1.0, w_max=100.5, n_per_decade=100, dw=0.02,
                breakpoints='schedule', w_split=None, dw_high=None):
    """Log-spaced grid on [w_min, w_switch), then uniform with step dw on [w_switch, w_max],
    plus a node pair (b*(1-1e-9), b) at every GL-schedule breakpoint b.

    If w_split and dw_high are given, the uniform step becomes dw_high above w_split (used
    above w = 100, where only strongly lensed lenses with short time delays, i.e. long fringe
    periods in w, still use the FIONA table; the rest have switched to bGO).

    Uniform nodes are placed at w_switch + k*dw, so a grid with step dw contains every node of
    a grid with step m*dw (same w_switch): coarser grids are exact subsets of finer ones.
    `breakpoints`: 'schedule' (from schedule_breakpoints()), None, or an array.
    """
    n_log = max(2, int(np.ceil(np.log10(w_switch / w_min) * n_per_decade)))
    w_log = np.logspace(np.log10(w_min), np.log10(w_switch), n_log + 1)[:-1]
    w_top = w_max if w_split is None else min(w_split, w_max)
    n_lin = int(np.ceil((w_top - w_switch) / dw - 1e-9))
    w_lin = w_switch + dw * np.arange(n_lin + 1)
    parts = [w_log, w_lin]
    if w_split is not None and dw_high is not None and w_max > w_lin[-1]:
        n_hi = int(np.ceil((w_max - w_lin[-1]) / dw_high - 1e-9))
        parts.append(w_lin[-1] + dw_high * np.arange(1, n_hi + 1))
    if isinstance(breakpoints, str) and breakpoints == 'schedule':
        breakpoints = schedule_breakpoints(w_min, w_max)
    if breakpoints is not None and len(breakpoints):
        b = np.asarray(breakpoints, dtype=float)
        parts += [b, b * (1.0 - 1e-9)]
    return np.unique(np.concatenate(parts))


def _mem_budget_bytes(mem_budget_gb):
    import os
    if mem_budget_gb is not None:
        return mem_budget_gb * 2 ** 30
    slurm_mb = os.environ.get('SLURM_MEM_PER_NODE')
    if slurm_mb:
        return 0.5 * float(slurm_mb) * 2 ** 20             # half the job's memory
    return 32 * 2 ** 30


# Rough per-worker working set of FIONA's batched solver, in units of n_lens * (n_gl/2) * 16 B
# (psi_stack, phase, cos/sin, integrand, free chirp, ck and its packed copy, NUFHT output).
_FIONA_ARRAYS_PER_NODE = 10


def build_F_w_table(xc_grid, y_grid, w_grid, gl_dir, n_workers=1, chunk=500,
                    alloc=None, verbose=True, w_max_rows=None, mem_budget_gb=None,
                    lens_model='CIS'):
    """Tabulate FIONA's F(w, x_c, y) on the outer product of the three grids.

    Returns an array of shape (n_xc, n_y, n_w), complex128. `alloc(shape, dtype)` may be
    given to control where the table lives (e.g. PE_CIS_glworia_reproduction's
    _alloc_shared_array); default is np.empty.

    The adaptive GL schedule in effect in fiona.axisym (e.g. fiona_adaptive_gl_v3.apply())
    is used, exactly as in a direct compute_F call at the same w.

    `w_max_rows` (optional, one value per x_c): skip computing row i above w_max_rows[i]
    (those entries are set to NaN and must never be queried -- see table_row_wmax, which
    derives them from the bGO switch).

    Chunks are formed WITHIN a GL tier (one n_gl per call). FIONA's batched solver splits its
    workers evenly among the distinct n_gl values present in one call, so a call mixing many
    tiers leaves the most expensive tier with a small fraction of the workers.

    Lenses (x_c rows) are split into sub-batches so that all workers' working sets together
    stay within `mem_budget_gb` (default: half of SLURM_MEM_PER_NODE, else 32 GB). Without
    this, 112 workers at n_gl = 512,000 and 21 lenses already need ~80 GB (OOM seen in job
    48193517).

    `lens_model`: fisherWithFIONA lens name ('CIS', 'ULA' or 'PointLens'); xc_grid then holds
    that lens's shape parameter (x_c, r_s, or the point lens's fixed softening).
    """
    xc_grid = np.asarray(xc_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    w_grid = np.asarray(w_grid, dtype=float)
    shape = (len(xc_grid), len(y_grid), len(w_grid))
    table = (alloc or np.empty)(shape, np.complex128)

    # compute_F_param_batch takes (M_Lz, f); with M_Lz = 1 we pass f = w / W_TO_MF so that
    # its internal w_vec reproduces w_grid (to rounding).
    M_REF = 1.0
    w_max_rows = (np.full(len(xc_grid), np.inf) if w_max_rows is None
                  else np.asarray(w_max_rows, dtype=float))
    import fiona.axisym as _axisym
    n_gl = np.array([_axisym._adaptive_n_gl(w) for w in w_grid])
    chunks = []                                         # index arrays, each within one tier
    for n in np.unique(n_gl):
        idx = np.nonzero(n_gl == n)[0]
        # split into contiguous runs first: one n_gl can occur in separate w ranges (e.g. the
        # 512k tier at w ~ 3e-4 and w ~ 220), and row trimming uses each chunk's lowest w
        for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
            chunks += [run[k:k + chunk] for k in range(0, len(run), chunk)]
    chunks.sort(key=lambda ix: ix[0])
    budget = _mem_budget_bytes(mem_budget_gb)
    all_y = np.arange(len(y_grid))
    t_start = time.time()
    done = 0
    for ix in chunks:
        t0 = time.time()
        rows = np.nonzero(w_max_rows >= w_grid[ix[0]])[0]
        table[:, :, ix] = np.nan
        done += len(ix)
        if len(rows) == 0:
            continue
        busy = min(n_workers, len(ix))                      # workers active in one call
        per_lens = busy * _FIONA_ARRAYS_PER_NODE * 16 * (n_gl[ix[0]] // 2)
        max_lens = max(1, int(budget // per_lens))
        for r0 in range(0, len(rows), max_lens):
            rr = rows[r0:r0 + max_lens]
            F = compute_F_param_batch(M_REF, y_grid, xc_grid[rr], w_grid[ix] / (W_TO_MF * M_REF),
                                      lens_model=lens_model, gl_dir=gl_dir, n_workers=n_workers)
            # (n_w, n_rows, n_y) -> (n_rows, n_y, n_w)
            table[np.ix_(rr, all_y, ix)] = F.transpose(1, 2, 0)
        if verbose:
            print(f'  [w-table] w in [{w_grid[ix[0]]:.4g}, {w_grid[ix[-1]]:.4g}]  '
                  f'n_gl={n_gl[ix[0]]}  {done}/{len(w_grid)}  rows={len(rows)}/{len(xc_grid)}  '
                  f'lens batches={-(-len(rows) // max_lens)}  '
                  f'{time.time()-t0:.1f}s  elapsed={time.time()-t_start:.0f}s', flush=True)
    if verbose:
        print(f'[w-table] built {shape} ({table.nbytes/2**30:.2f} GiB) in '
              f'{time.time()-t_start:.0f}s', flush=True)
    return table


class FwInterpolator:
    """F(w; x_c, y) from a (n_xc, n_y, n_w) table: bilinear in (x_c, y), linear in w."""

    def __init__(self, xc_grid, y_grid, w_grid, table, lens_model='CIS'):
        self.lens_model = lens_model
        self.xc_grid = np.asarray(xc_grid, dtype=float)
        self.y_grid = np.asarray(y_grid, dtype=float)
        self.w_grid = np.asarray(w_grid, dtype=float)
        self.table = table
        assert table.shape == (len(self.xc_grid), len(self.y_grid), len(self.w_grid))

    @staticmethod
    def _bracket(grid, x):
        if len(grid) == 1:                  # single shape node (point lens): no interpolation
            return 0, 0.0
        x = min(max(x, grid[0]), grid[-1])
        i = int(np.clip(np.searchsorted(grid, x, side='right') - 1, 0, len(grid) - 2))
        t = (x - grid[i]) / (grid[i + 1] - grid[i])
        return i, t

    def curve(self, xc, y):
        """The bilinearly interpolated F on self.w_grid at (xc, y)."""
        i, a = self._bracket(self.xc_grid, float(xc))
        j, b = self._bracket(self.y_grid, float(y))
        T = self.table
        if len(self.xc_grid) == 1:
            return (1 - b) * T[0, j] + b * T[0, j + 1]
        return ((1 - a) * (1 - b) * T[i, j] + a * (1 - b) * T[i + 1, j]
                + (1 - a) * b * T[i, j + 1] + a * b * T[i + 1, j + 1])

    def __call__(self, w, xc, y):
        """F at the dimensionless frequencies `w` (any shape), clamped to the table's w range.

        If the table holds a residual (see subtract_go_reference), the exact GO reference at
        (xc, y) is added back, so the result is always F itself."""
        w = np.asarray(w, dtype=float)
        c = self.curve(xc, y)
        F = np.interp(w, self.w_grid, c.real) + 1j * np.interp(w, self.w_grid, c.imag)
        if getattr(self, 'go_w_ref', None) is not None:
            F = F + go_weight(w, self.go_w_ref) * F_go(w, *image_set(self.lens_model, xc, y))
        return F


# ─── Geometric-optics reference, for tabulating a smooth residual ──────────────────────
#
# At high w, F(w; x_c, y) is dominated by interfering images, F ~ sum_j |mu_j|^1/2
# exp(i(w phi_j - pi n_j)), whose phases w*phi_j(x_c, y) change by radians across one
# (x_c, y) grid cell -- the cause of the large bilinear (x_c, y) interpolation error at
# w > 30. Tabulating R = F - s(w) F_GO instead, and adding F_GO back exactly at the query
# point, leaves only the slowly varying wave-optics correction to be interpolated.
#
# F_GO sums ALL images, with no magnification cut and no bGO Delta/w term: production's
# F_bGO drops images with |mu| < 1e-2, which makes it discontinuous in (x_c, y) where a
# faint central image crosses the cut. The all-image GO sum is continuous except at the
# caustic itself (where two images merge and |mu| diverges), a band the prior masks out.
# Verified from the GL smoke test that FIONA and bGO share the same phase convention.

def cis_image_set(xc, y, x_max=20.0, n_half=2000):
    """All CIS images at (xc, y): arrays (phi, mu, n_morse). Robust to the faint central
    image near x=0 (geometric grid dense near 0), unlike a uniform sign-change scan."""
    from scipy.optimize import brentq
    from fisherWithFIONA.cis_go import _lens_eq, cis_image_props
    xc = max(float(xc), 1.0e-6)
    g = np.geomspace(1.0e-7, x_max, n_half)
    xs = np.concatenate([-g[::-1], g])
    f = _lens_eq(xs, float(y), xc)
    idx = np.nonzero(f[:-1] * f[1:] < 0)[0]
    phi, mu, nm = [], [], []
    for i in idx:
        x_I = brentq(_lens_eq, xs[i], xs[i + 1], args=(float(y), xc), xtol=1e-14, rtol=1e-14)
        p, m, n_I, _ = cis_image_props(x_I, float(y), xc)
        phi.append(p); mu.append(m); nm.append(n_I)
    return np.array(phi), np.array(mu), np.array(nm)


# ─── Other lenses (ULA soliton, point lens / PBH): exact deflections, same image search ─
#
# The table code takes a fisherWithFIONA lens name; its "x_c" axis then holds that lens's
# shape parameter (ULA: r_s; point lens: the fixed Plummer softening, a single node). The
# GO residual needs every image's (phi, mu, n) to high accuracy at w up to 254, so the
# deflection and its derivative are analytic here (generic_go's finite differences, with a
# step >= 1e-3, are too coarse for the ULA core at small r_s). psi itself is FIONA's own
# psi_r, so the image phases match the potential FIONA integrates.

POINT_LENS_SOFTENING = 1.0e-5     # as fisherWithFIONA's PBH runs (Eq. 9 of the paper draft)


def make_lens(lens_model, shape):
    """fiona lens object for a fisherWithFIONA lens name and shape parameter (psi0 = 1)."""
    from fiona.lenses import CIS, PointLens, ULASoliton
    if lens_model == 'CIS':
        return CIS(psi0=1.0, xc=max(float(shape), 1e-6))
    if lens_model == 'ULA':
        return ULASoliton(psi0=1.0, r_s=float(shape))
    if lens_model == 'PointLens':
        return PointLens(psi0=1.0, xc=float(shape))
    raise ValueError(f'unknown lens_model {lens_model!r}')


def _ula_alpha(r, rs):
    """ULA soliton deflection alpha(r), r > 0, normalised to alpha(1) = 1 exactly as
    fiona.lenses.ULASoliton: kappa ~ (1 + r^2/r_s^2)^-7.5 enclosed analytically."""
    r = np.asarray(r, dtype=float)
    norm = 1.0 - (1.0 + rs ** -2) ** -6.5
    return (1.0 - (1.0 + r * r / rs ** 2) ** -6.5) / (r * norm)


def _ula_dalpha(r, rs):
    r = np.asarray(r, dtype=float)
    norm = 1.0 - (1.0 + rs ** -2) ** -6.5
    u = 1.0 + r * r / rs ** 2
    return (13.0 * u ** -7.5 / rs ** 2 - (1.0 - u ** -6.5) / (r * r)) / norm


def _point_alpha(r, eps):
    r = np.asarray(r, dtype=float)
    return r / (r * r + eps * eps)


def _point_dalpha(r, eps):
    r = np.asarray(r, dtype=float)
    return (eps * eps - r * r) / (r * r + eps * eps) ** 2


def _alpha_fns(lens_model, shape):
    if lens_model == 'ULA':
        return (lambda r: _ula_alpha(r, shape)), (lambda r: _ula_dalpha(r, shape))
    if lens_model == 'PointLens':
        return (lambda r: _point_alpha(r, shape)), (lambda r: _point_dalpha(r, shape))
    raise ValueError(f'no analytic deflection for {lens_model!r}')


def axisym_image_set(lens_model, shape, y, x_max=40.0, n_half=4000):
    """All images of an axisymmetric lens with analytic deflection: (phi, mu, n_morse).
    Signed 1D coordinate x, alpha(x) = sign(x) alpha(|x|); dense geometric grid near 0 for
    the faint central image. The point lens's central image (|x| ~ y eps^2, |mu| ~ eps^4)
    lies below the grid and is negligible."""
    from scipy.optimize import brentq
    alpha, dalpha = _alpha_fns(lens_model, float(shape))
    psi = make_lens(lens_model, shape).psi_r
    y = float(y)
    lens_eq = lambda x: x - np.sign(x) * alpha(np.abs(x)) - y
    g = np.geomspace(1.0e-9, x_max, n_half)
    xs = np.concatenate([-g[::-1], g])
    f = lens_eq(xs)
    idx = np.nonzero(f[:-1] * f[1:] < 0)[0]
    phi, mu, nm = [], [], []
    for i in idx:
        if xs[i] < 0 < xs[i + 1]:
            continue                         # sign flip of alpha across x = 0, not a root
        x = brentq(lens_eq, xs[i], xs[i + 1], xtol=1e-15, rtol=1e-15)
        r = abs(x)
        f1 = 1.0 - float(alpha(r)) / r       # tangential eigenvalue
        f2 = 1.0 - float(dalpha(r))          # radial eigenvalue
        phi.append(0.5 * (x - y) ** 2 - float(psi(np.array([r]))[0]))
        mu.append(1.0 / (f1 * f2))
        nm.append(0.0 if (f1 > 0 and f2 > 0) else (1.0 if (f1 < 0 and f2 < 0) else 0.5))
    return np.array(phi), np.array(mu), np.array(nm)


_LAST_IMAGE_SET = [None, None]   # one-entry memo: a likelihood call asks twice per (shape, y)


def image_set(lens_model, shape, y):
    """All images at (shape, y) for the given lens: arrays (phi, mu, n_morse)."""
    key = (lens_model, float(shape), float(y))
    if _LAST_IMAGE_SET[0] == key:
        return _LAST_IMAGE_SET[1]
    res = cis_image_set(shape, y) if lens_model == 'CIS' else axisym_image_set(lens_model, shape, y)
    _LAST_IMAGE_SET[:] = [key, res]
    return res


def ula_y_crit(rs):
    """Radial caustic y_crit(r_s) of the ULA soliton (alpha' = 1 at r_crit, y = r - alpha).
    Exists for every r_s in the prior; it lies above y = 5 for r_s < ~0.3."""
    from scipy.optimize import brentq
    out = []
    for s in np.atleast_1d(np.asarray(rs, dtype=float)):
        rr = np.geomspace(1e-4 * s, 50.0, 20000)
        g = 1.0 - _ula_dalpha(rr, s)
        k = np.nonzero(g[:-1] * g[1:] < 0)[0]
        if len(k) == 0:
            out.append(-99.0)                 # no caustic: mask off (as glworia's sentinel)
            continue
        rc = brentq(lambda r: 1.0 - _ula_dalpha(r, s), rr[k[0]], rr[k[0] + 1], xtol=1e-14)
        out.append(abs(rc - float(_ula_alpha(rc, s))))
    out = np.array(out)
    return out if np.ndim(rs) else float(out[0])


_ULA_Y_CRIT_TAB = None


def ula_y_crit_fast(rs):
    """ula_y_crit from a fine tabulation (r_s in [1e-3, 2], 4000 log nodes; < 1e-6 error),
    for use inside the prior."""
    global _ULA_Y_CRIT_TAB
    if _ULA_Y_CRIT_TAB is None:
        g = np.geomspace(1e-3, 2.0, 4000)
        _ULA_Y_CRIT_TAB = (np.log(g), np.log(ula_y_crit(g)))
    lg, ly = _ULA_Y_CRIT_TAB
    return np.exp(np.interp(np.log(np.asarray(rs, dtype=float)), lg, ly))


def ula_prior_masked(rs, y, half_width=0.15):
    """True where the ULA PE prior excludes (r_s, y): the band |y - y_crit(r_s)| < 0.15 around
    the radial caustic, the same width as the CIS mask (glworia's caps 0.15/0.15)."""
    return np.abs(np.asarray(y, dtype=float) - ula_y_crit_fast(rs)) < half_width


def prior_masked_fn(lens_model):
    """The prior-mask function (shape, y) -> bool used for table row trimming."""
    if lens_model == 'CIS':
        return cis_prior_masked
    if lens_model == 'ULA':
        return ula_prior_masked
    return lambda shape, y: False         # point lens: caustic only at y = 0, outside the prior


def F_go(w, phi, mu, nm):
    """Plain geometric-optics sum over the given images."""
    w = np.asarray(w, dtype=float)
    F = np.zeros(w.shape, dtype=complex)
    for p, m, n_I in zip(phi, mu, nm):
        F += np.sqrt(abs(m)) * np.exp(1j * (w * p - np.pi * n_I))
    return F


def go_weight(w, w_ref):
    """Smooth 0 -> 1 turn-on of the GO reference around w_ref (any smooth s is exact at the
    nodes, since the same s is subtracted when building and added back when querying)."""
    return 0.5 * (1.0 + np.tanh((np.asarray(w, dtype=float) - w_ref) / (0.2 * w_ref)))


def _image_set_task(args):
    return image_set(*args)


def tabulate_image_sets(xc_grid, y_grid, n_workers=1, lens_model='CIS'):
    """image_set at every (shape, y) node, as a list of lists [i][j] -> (phi, mu, n)."""
    tasks = [(lens_model, xc, y) for xc in xc_grid for y in y_grid]
    if n_workers > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(n_workers) as pool:
            flat = pool.map(_image_set_task, tasks, chunksize=64)
    else:
        flat = [_image_set_task(t) for t in tasks]
    ny = len(y_grid)
    return [[flat[i * ny + j] for j in range(ny)] for i in range(len(xc_grid))]


def subtract_go_reference(interp, w_ref=1.0, n_workers=1, image_sets=None, verbose=True):
    """Turn an FwInterpolator's table into the residual R = F - s(w) F_GO, IN PLACE (no extra
    table-sized memory), and make the interpolator add s(w) F_GO back at query time."""
    t0 = time.time()
    if image_sets is None:
        image_sets = tabulate_image_sets(interp.xc_grid, interp.y_grid, n_workers,
                                         getattr(interp, 'lens_model', 'CIS'))
    s = go_weight(interp.w_grid, w_ref)
    on = s > 1e-12
    w_on = interp.w_grid[on]
    for i in range(len(interp.xc_grid)):
        for j in range(len(interp.y_grid)):
            interp.table[i, j, on] -= s[on] * F_go(w_on, *image_sets[i][j])
    interp.go_w_ref = float(w_ref)
    interp.image_sets = image_sets
    if verbose:
        print(f'[w-table] subtracted GO reference (w_ref={w_ref:g}) in '
              f'{time.time()-t0:.0f}s', flush=True)
    return interp


# ─── Lens-dependent FIONA -> bGO switch ────────────────────────────────────────────────
#
# derive_bgo_switch.py measures, on a fine (x_c, y) grid, w_s(x_c, y): the lowest w above
# which production's bGO (fisherWithFIONA.generic_go.F_bGO) stays within eps of converged
# FIONA up to w = 254. It ranges from ~0.5 (weak lensing, large y and x_c) to never (small
# x_c, small y; and along the caustic). This replaces the fixed W_HIGH = 100, which was too
# late for most weak-lensing lenses and too early for strongly lensed small-core ones.

class BgoSwitch:
    """Conservative lookup of w_s(x_c, y): the LARGEST value among the four surrounding
    grid nodes (inf if any is inf), so bGO is used only where every neighbour says it is
    accurate. Blend: b(w) rises smoothly from 0 at w_s to 1 at (1 + width) w_s."""

    def __init__(self, npz_path, eps=1e-3, width=0.2):
        d = np.load(npz_path)
        self.xc_grid = d['xc_grid']
        self.y_grid = d['y_grid']
        self.w_s = d[f'w_s_{eps:g}']
        self.eps = eps
        self.width = width

    def w_switch(self, xc, y):
        i = int(np.clip(np.searchsorted(self.xc_grid, xc, side='right') - 1,
                        0, len(self.xc_grid) - 2))
        j = int(np.clip(np.searchsorted(self.y_grid, y, side='right') - 1,
                        0, len(self.y_grid) - 2))
        return float(np.max(self.w_s[i:i + 2, j:j + 2]))

    def weight(self, w, xc, y):
        """bGO weight b(w) in [0, 1] (smoothstep between w_s and (1 + width) w_s)."""
        w_s = self.w_switch(xc, y)
        w = np.asarray(w, dtype=float)
        if not np.isfinite(w_s):
            return np.zeros(w.shape)
        t = np.clip((w / w_s - 1.0) / self.width, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)


def F_hybrid(w, xc, y, interp, switch):
    """Production F: FIONA table (FwInterpolator) blended into bGO above w_s(x_c, y)."""
    from fisherWithFIONA.generic_go import F_bGO
    w = np.asarray(w, dtype=float)
    b = switch.weight(w, xc, y)
    F = np.empty(w.shape, dtype=complex)
    lo = b < 1.0
    if lo.any():
        F[lo] = interp(w[lo], xc, y)
    hi = b > 0.0
    if hi.any():
        Fb = F_bGO(make_lens(getattr(interp, 'lens_model', 'CIS'), xc), float(y), w[hi])
        F[hi] = np.where(lo[hi], (1.0 - b[hi]) * F[hi] + b[hi] * Fb, Fb)
    return F


# ─── Prior caustic mask, and which table rows the switch lets us skip ──────────────────

def y_crit_cis(xc):
    """Analytic CIS critical curve (same as PE_CIS_glworia_reproduction.y_crit_cis)."""
    xc = np.asarray(xc, dtype=float)
    x = np.maximum(xc, 1.0e-12)
    s_crit = 0.5 * (-x + np.sqrt(x ** 2 + 4.0 * x))
    x_crit = np.sqrt(np.maximum(s_crit ** 2 - x ** 2, 0.0))
    y_crit = np.abs(x_crit - x_crit / (s_crit + x))
    return np.where(xc < 1.0e-10, 1.0, y_crit)


# glworia's own critical-curve interpolator, exactly as its released scripts/PE_full.py uses it
# for the CIS benchmarks (interpolator_settings crit_param_low/high = 0/1). It agrees with the
# analytic y_crit_cis to all printed digits for 0 < x_c < 0.49, but returns a sentinel of
# about -99 where CIS has no caustic (x_c >~ 0.49, and x_c = 0 exactly), so glworia's mask is
# OFF there.
GLWORIA_CRIT_PKL = 'glworia_repo/interpolation/crit_funcs/CIS_crit_funcs_0.000_1.000.pkl'
_GLW_Y_CRIT = None


def glworia_y_crit(xc):
    """y_crit(x_c) from glworia's pickled interp1d (loaded once)."""
    global _GLW_Y_CRIT
    if _GLW_Y_CRIT is None:
        import os
        import pickle
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), GLWORIA_CRIT_PKL)
        with open(path, 'rb') as f:
            _GLW_Y_CRIT = pickle.load(f)['lens_param_to_y_crit']
    return np.asarray(_GLW_Y_CRIT(xc), dtype=float)


def cis_prior_masked(xc, y, fac=1.0, cap_high=0.15, cap_low=0.15):
    """True where the CIS PE prior excludes (x_c, y). Identical to what glworia's released
    code does: glworia.pe.prior.Uniform2DMaskDist.check_mask with the benchmark configs'
    crit_mask_settings (fac 1.0, caps 0.15/0.15) and glworia's own y_crit interpolator, and
    NO mask box -- the configs list mask_boxes [[[0.7, 1.1], [0.0, 0.05]]], but
    scripts/PE_full.py never passes them to Uniform2DMaskDist, so glworia's runs don't use
    them. PE_CIS_glworia_reproduction.py builds its prior with the same settings; keep the two
    in sync (the table's row trimming relies on this definition)."""
    xc = np.asarray(xc, dtype=float)
    y = np.asarray(y, dtype=float)
    yc = glworia_y_crit(xc)
    half = np.maximum(np.minimum(fac * yc, cap_high), cap_low)
    return np.abs(y - yc) < half


def table_row_wmax(xc_grid, y_lo, y_hi, switch, w_max, n_y=400, masked=None):
    """Per table row i: the highest w at which any UNMASKED query between rows i-1 and i+1
    can still use the table, i.e. (1 + switch.width) * w_s, capped at w_max."""
    masked = cis_prior_masked if masked is None else masked
    xc_grid = np.asarray(xc_grid, dtype=float)
    ys = np.linspace(y_lo, y_hi, n_y)
    need = np.empty(len(xc_grid))
    for i in range(len(xc_grid)):
        xs = np.linspace(xc_grid[max(i - 1, 0)], xc_grid[min(i + 1, len(xc_grid) - 1)], 5)
        best = 0.0
        for xc in xs:
            for y in ys:
                if masked(xc, y):
                    continue
                best = max(best, switch.w_switch(xc, y))
        need[i] = min(w_max, (1.0 + switch.width) * best)
    return need


# ─── On-disk cache (the ~5 GB table; allowed -- unlike the old 300 GB 4D table) ─────────

def _sha(arr_or_bytes):
    import hashlib
    b = arr_or_bytes if isinstance(arr_or_bytes, bytes) else np.ascontiguousarray(arr_or_bytes).tobytes()
    return hashlib.sha256(b).hexdigest()


def table_config(xc_grid, y_grid, w_grid, go_w_ref=None, w_max_rows=None, extra=None):
    """Everything that determines the table's contents, reduced to a hashable dict: the
    grids, the GL node count actually used at every w (so any schedule change invalidates),
    the GO-residual setting, the row trimming, and the FIONA solver source."""
    import inspect
    import fiona.axisym as _axisym
    n_gl = np.array([_axisym._adaptive_n_gl(w) for w in w_grid], dtype=np.int64)
    cfg = dict(
        xc_grid=_sha(np.asarray(xc_grid, dtype=float)),
        y_grid=_sha(np.asarray(y_grid, dtype=float)),
        w_grid=_sha(np.asarray(w_grid, dtype=float)),
        n_gl=_sha(n_gl),
        go_w_ref=None if go_w_ref is None else float(go_w_ref),
        w_max_rows=None if w_max_rows is None else _sha(np.asarray(w_max_rows, dtype=float)),
        fiona_axisym=_sha(inspect.getsource(_axisym).encode()),
        shape=[len(xc_grid), len(y_grid), len(w_grid)],
    )
    if extra:
        cfg.update(extra)
    cfg['key'] = _sha(json.dumps(cfg, sort_keys=True).encode())[:16]
    return cfg


def save_table(cache_dir, interp, cfg):
    """Write the (possibly residual) table + grids + config atomically under cache_dir/key."""
    import os
    d = os.path.join(cache_dir, cfg['key'])
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    for name, arr in (('table', interp.table), ('xc_grid', interp.xc_grid),
                      ('y_grid', interp.y_grid), ('w_grid', interp.w_grid)):
        tmp = os.path.join(d, f'{name}.tmp{os.getpid()}.npy')   # unique per writer
        np.save(tmp, np.asarray(arr))
        os.replace(tmp, os.path.join(d, name + '.npy'))
    tmp = os.path.join(d, f'config.json.tmp{os.getpid()}')
    with open(tmp, 'w') as f:
        json.dump(cfg, f, indent=1)
    os.replace(tmp, os.path.join(d, 'config.json'))       # written last = marks complete
    print(f'[w-table] saved to {d} in {time.time()-t0:.0f}s', flush=True)
    return d


def load_table(cache_dir, cfg, alloc=None):
    """Return an FwInterpolator from cache_dir/key if a complete entry with an identical
    config exists, else None. The table is copied into `alloc` (e.g. shared memory) if
    given, otherwise memory-mapped read-only (forked workers then share the page cache)."""
    import os
    d = os.path.join(cache_dir, cfg['key'])
    path = os.path.join(d, 'config.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        saved = json.load(f)
    if saved != cfg:
        return None
    t0 = time.time()
    mm = np.load(os.path.join(d, 'table.npy'), mmap_mode='r')
    if alloc is not None:
        table = alloc(mm.shape, mm.dtype)
        table[...] = mm
    else:
        table = mm
    interp = FwInterpolator(np.load(os.path.join(d, 'xc_grid.npy')),
                            np.load(os.path.join(d, 'y_grid.npy')),
                            np.load(os.path.join(d, 'w_grid.npy')), table,
                            lens_model=cfg.get('lens_model', 'CIS'))
    if cfg.get('go_w_ref') is not None:
        interp.go_w_ref = cfg['go_w_ref']
    print(f'[w-table] loaded {mm.shape} from {d} in {time.time()-t0:.0f}s', flush=True)
    return interp
