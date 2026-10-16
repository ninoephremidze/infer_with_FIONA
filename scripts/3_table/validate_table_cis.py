#!/usr/bin/env python
"""Validate the F(w, x_c, y) table (table.py) against direct FIONA compute_F.

Reference ("truth") is FIONA itself, evaluated directly at every test point, with the same
v3 GL schedule. So this measures only the error the TABLE adds, not FIONA-vs-glworia
accuracy (settled separately in analyze_gl_schedule_v3_validation.ipynb). Since both sides
are FIONA, the comparison is on complex F -- stricter than magnitudes, and the phase matters
for the likelihood.

Test points: the 200-combo v3 validation grid (20 (x_c,y) points x 10 masses in
[0.1, 2000]), evaluated on bilby's production frequency array (4 s, 2048 Hz -> 0.25 Hz
spacing), f >= 20 Hz, restricted to w <= W_HIGH=100 (above that production uses bGO).

Stages
  A. w-interpolation error alone: F tabulated on candidate w-grids at each test point's
     EXACT (x_c, y), then interpolated in w only. Chooses dw / points-per-decade.
  O. The OLD scheme's M_Lz-interpolation error alone, for comparison: F at the two
     bracketing nodes of the production logM grid (log-uniform [1,2000], 121 nodes), exact
     (x_c, y), linearly interpolated in logM at fixed f. Masses in [1, 2000] only.
  B. Full pipeline: build the (n_grid x n_grid x n_w) table, query with FwInterpolator
     (bilinear in (x_c,y) + linear in w) and compare. This includes (x_c,y) interpolation
     error, which is identical in kind to what the old table has at the same n_grid.

Metric (per combo, over valid bins): max |F_table - F_direct| / median |F_direct|
(null-robust: fringe nulls do not inflate it), plus max |F_table - F_direct| itself.
"""
import argparse
import json
import os
import time

import finufft  # noqa: F401 -- must precede any FIONA import
from infer_with_fiona import paths, quadrature  # noqa: E402 -- after finufft
import numpy as np
from infer_with_fiona.fiona.axisym import FresnelNUFHTLensBatched
import infer_with_fiona.fiona.utils as _fiona_utils
from infer_with_fiona.fisher.lensing import _make_lens_3p

import importlib
from infer_with_fiona import table as FWT

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--n-grid', type=int, default=201,
                help='(x_c,y) grid size for stage B; 201 = production grid-res=high.')
ap.add_argument('--dw', type=float, default=0.02, help='Uniform w step for stage B.')
ap.add_argument('--n-per-decade', type=int, default=100, help='Log-part density for stage B.')
ap.add_argument('--dw-candidates', default='0.1,0.05,0.02,0.01,0.005')
ap.add_argument('--npd-candidates', default='50,100,200')
ap.add_argument('--stages', default='A,O,B')
ap.add_argument('--n-workers', type=int, default=112)
ap.add_argument('--quick', action='store_true',
                help='Smoke test: 3 (x_c,y) points, 3 masses, coarse candidates.')
ap.add_argument('--out', default='table_validation_cis')
ap.add_argument('--gl-schedule', default='cis', choices=quadrature.CHOICES,
                help='Gauss-Legendre node schedule (infer_with_fiona.quadrature). (default: cis)')
ap.add_argument('--go-residual', type=float, default=None, metavar='W_REF',
                help='Stage B: tabulate R = F - s(w) F_GO (all-image geometric optics) and '
                     'add F_GO back at query, with s turning on around W_REF. '
                     '(default: off, i.e. tabulate F itself)')
ap.add_argument('--switch', default=None, metavar='NPZ',
                help='Stage B: blend into bGO above w_s(x_c, y) from this derive_bgo_switch_cis.py '
                     'output (F_hybrid), and test up to --w-max instead of 100.')
ap.add_argument('--switch-eps', type=float, default=1e-3)
ap.add_argument('--w-max', type=float, default=100.5, help='Table w_max (and test range with --switch).')
ap.add_argument('--w-split', type=float, default=None, help='Uniform step becomes --dw-high above this.')
ap.add_argument('--dw-high', type=float, default=None)
ap.add_argument('--table-cache', default=None, metavar='DIR',
                help='Stage B: load the table from this cache if an identical config exists, '
                     'otherwise build it and save it there (table.save_table).')
args = ap.parse_args()
STAGES = set(args.stages.split(','))

ROOT = str(paths.REPO_ROOT)
CHECKS = str(paths.CHECKS_DIR); os.makedirs(CHECKS, exist_ok=True)   # outputs/checks
GL_DIR = str(paths.GL_RULES_DIR)
W_HIGH = 100.0
W_TEST_MAX = args.w_max if args.switch else W_HIGH   # without the switch, prod. uses bGO > 100

GLS = quadrature.get_schedule(args.gl_schedule)
GLS.seed_gl_cache(GL_DIR, n_values=tuple(GLS.REQUIRED_N_GL))
GLS.apply()

# ─── test points ───────────────────────────────────────────────────────────────────────
meta = json.load(open(paths.TABLE_VALIDATION_POINTS))
points = sorted({(m['xc'], m['y'], m['regime']) for m in meta})
masses = sorted({m['M_Lz'] for m in meta})
if args.quick:
    points = [points[0], points[8], points[15]]
    masses = [masses[3], masses[7], masses[9]]
xc_pts = np.array([p[0] for p in points])
y_pts = np.array([p[1] for p in points])
n_pts = len(points)

f_all = np.arange(4097) * 0.25                     # bilby: duration=4 s, fs=2048 Hz
f_test = f_all[f_all >= 20.0]

# Production logM grid of the failing runs (log-uniform [1,2000], n_logm=121)
logM_grid_old = np.linspace(np.log(1.0), np.log(2000.0), 121)

print(f'{n_pts} (x_c,y) points x {len(masses)} masses; {len(f_test)} test frequencies; '
      f'stages={sorted(STAGES)}', flush=True)


def direct_points(w_vec):
    """Direct FIONA F at every test point's exact (x_c, y): shape (n_w, n_pts).

    One batched call over all points (lenses = x_c values, outputs = y values), keeping the
    diagonal -- much cheaper than one call per point, since FIONA's per-w overhead
    dominates for small batches."""
    _fiona_utils._FIONA_GL2D_DIR = GL_DIR
    solver = FresnelNUFHTLensBatched([_make_lens_3p("CIS", xc) for xc in xc_pts], gl_dir=GL_DIR,
                                     n_workers=args.n_workers, print_timing=False)
    F = solver(np.asarray(w_vec, dtype=float), y_pts)       # (n_w, n_pts, n_pts)
    return F[:, np.arange(n_pts), np.arange(n_pts)]


W_BANDS = ((0.0, 1.0), (1.0, 10.0), (10.0, 30.0), (30.0, W_HIGH)) + \
          (((W_HIGH, W_TEST_MAX + 1.0),) if W_TEST_MAX > W_HIGH else ())


def err_metrics(F_tab, F_dir, w):
    """Null-robust max error overall and per w-band (all normalised by the same median
    |F_direct| of the combo), plus the w at which the worst error occurs."""
    d = np.abs(F_tab - F_dir)
    scale = np.median(np.abs(F_dir))
    m = dict(scaled=float(d.max() / scale), absmax=float(d.max()),
             w_worst=float(w[np.argmax(d)]))
    for lo, hi in W_BANDS:
        sel = (w >= lo) & (w < hi)
        m[f'band_{lo:g}_{hi:g}'] = float(d[sel].max() / scale) if sel.any() else float('nan')
    return m


# ─── truth: direct F at every (point, mass) on the test frequencies ────────────────────
t0 = time.time()
truth = {}
for M in masses:
    w = FWT.w_of(M, f_test)
    band = w <= W_TEST_MAX
    if not band.any():
        continue
    truth[M] = (w[band], direct_points(w[band]))
    print(f'  truth M_Lz={M:<8g} {band.sum():5d} bins  ({time.time()-t0:.0f}s)', flush=True)

results = dict(points=[list(p) for p in points], masses=masses, args=vars(args))


def summarize(name, rows, key='scaled'):
    """rows: list of dicts with 'M', 'regime', key."""
    def line(label, sel):
        a = np.array([r[key] for r in sel])
        if len(a) == 0:
            return
        print(f'    {label:22s} n={len(a):3d}  worst={a.max():9.3%}  95%={np.percentile(a,95):9.3%}'
              f'  median={np.median(a):9.3%}  >1%={int((a>0.01).sum()):3d}  >5%={int((a>0.05).sum()):3d}')
    # The sampler's prior is glworia's M_Lz in [0.1, 2000] (all test masses) with the caustic
    # band masked out, so "prior" = every test point OUTSIDE the mask.
    for r in rows:
        r.setdefault('masked', bool(FWT.cis_prior_masked(points[r['k']][0], points[r['k']][1])))
    print(f'  {name}')
    line('all (incl. masked)', rows)
    line('prior (outside mask)', [r for r in rows if not r['masked']])
    line('strong, outside mask', [r for r in rows if not r['masked'] and r['regime'] == 'strong'])
    line('weak,   outside mask', [r for r in rows if not r['masked'] and r['regime'] == 'weak'])
    prior = [r for r in rows if not r['masked']]
    for lo, hi in W_BANDS:
        a = np.array([r[f'band_{lo:g}_{hi:g}'] for r in prior])
        a = a[np.isfinite(a)]
        if len(a):
            print(f'      w in [{lo:g},{hi:g}) outside mask  n={len(a):3d}  worst={a.max():9.3%}  '
                  f'median={np.median(a):9.3%}')


# ─── Stage A: w-interpolation error at exact (x_c, y) ──────────────────────────────────
if 'A' in STAGES:
    dws = [float(x) for x in args.dw_candidates.split(',')]
    npds = [int(x) for x in args.npd_candidates.split(',')]
    if args.quick:
        dws, npds = [0.05, 0.02], [100]
    grids = {}
    for dw in dws:
        grids[('dw', dw)] = FWT.make_w_grid(dw=dw, n_per_decade=max(npds))
    for npd in npds:
        grids[('npd', npd)] = FWT.make_w_grid(dw=min(dws), n_per_decade=npd)
    # same as dw=0.02 but WITHOUT nodes straddling the GL tier boundaries, for comparison
    grids[('nobp', 0.02)] = FWT.make_w_grid(dw=0.02, n_per_decade=max(npds), breakpoints=None)
    w_union = np.unique(np.concatenate(list(grids.values())))
    print(f'\n[A] direct F on {len(w_union)} union w-nodes at exact (x_c,y) ...', flush=True)
    t0 = time.time()
    F_union = direct_points(w_union)                     # (n_union, n_pts)
    print(f'[A] done in {time.time()-t0:.0f}s', flush=True)
    results['A'] = {}
    print('\n=== Stage A: w-interpolation error only (exact x_c, y) ===')
    for gkey, wg in grids.items():
        idx = np.searchsorted(w_union, wg)
        assert np.allclose(w_union[idx], wg, rtol=0, atol=1e-12)
        Fg = F_union[idx]
        rows = []
        for M, (w, Fd) in truth.items():
            for k in range(n_pts):
                Ft = np.interp(w, wg, Fg[:, k].real) + 1j * np.interp(w, wg, Fg[:, k].imag)
                rows.append(dict(M=M, k=k, regime=points[k][2], **err_metrics(Ft, Fd[:, k], w)))
        results['A'][f'{gkey[0]}={gkey[1]}'] = dict(n_w=len(wg), rows=rows)
        label = {'dw': f'dw={gkey[1]:<6g} (npd={max(npds)})',
                 'npd': f'npd={gkey[1]} (dw={min(dws):g})',
                 'nobp': f'dw={gkey[1]:g}, NO tier-boundary nodes'}[gkey[0]]
        summarize(f'{label}  n_w={len(wg)}', rows)

# ─── Stage O: the old scheme's logM interpolation at exact (x_c, y) ────────────────────
if 'O' in STAGES:
    print('\n=== Stage O: OLD (logM, f) table -- logM-interpolation error only ===', flush=True)
    rows = []
    node_cache = {}
    for M, (w, Fd) in truth.items():
        if not (1.0 <= M <= 2000.0):
            continue
        lm = np.log(M)
        i = int(np.clip(np.searchsorted(logM_grid_old, lm, side='right') - 1, 0, 119))
        t = (lm - logM_grid_old[i]) / (logM_grid_old[i + 1] - logM_grid_old[i])
        f_band = w / FWT.W_TO_MF / M                  # the same test frequencies
        Fn = []
        for ii in (i, i + 1):
            if ii not in node_cache:
                node_cache[ii] = direct_points(FWT.w_of(np.exp(logM_grid_old[ii]), f_test))
            Fn.append(node_cache[ii][:len(f_band)])   # f_band is a prefix of f_test
        Ft = (1 - t) * Fn[0] + t * Fn[1]
        for k in range(n_pts):
            rows.append(dict(M=M, k=k, regime=points[k][2], **err_metrics(Ft[:, k], Fd[:, k], w)))
    results['O'] = rows
    summarize('old logM-interp (121 nodes, log-uniform [1,2000])', rows)

# ─── Stage B: full table + FwInterpolator ──────────────────────────────────────────────
if 'B' in STAGES:
    N = 21 if args.quick else args.n_grid
    xc_grid = np.linspace(1.0e-6, 1.0, N)          # same grids as the PE script
    y_grid = np.linspace(0.1, 5.0, N)
    w_grid = FWT.make_w_grid(dw=args.dw, n_per_decade=args.n_per_decade, w_max=args.w_max,
                             w_split=args.w_split, dw_high=args.dw_high)
    switch = FWT.BgoSwitch(args.switch, eps=args.switch_eps) if args.switch else None
    w_max_rows = (FWT.table_row_wmax(xc_grid, 0.1, 5.0, switch, args.w_max)
                  if switch is not None else None)
    print(f'\n[B] building {N}x{N}x{len(w_grid)} table '
          f'({N*N*len(w_grid)*16/2**30:.2f} GiB) ...', flush=True)
    cfg = FWT.table_config(xc_grid, y_grid, w_grid, go_w_ref=args.go_residual,
                           w_max_rows=w_max_rows)
    interp = FWT.load_table(args.table_cache, cfg) if args.table_cache else None
    t_build = t_go = 0.0
    if interp is None:
        t0 = time.time()
        table = FWT.build_F_w_table(xc_grid, y_grid, w_grid, GL_DIR, n_workers=args.n_workers,
                                    w_max_rows=w_max_rows)
        t_build = time.time() - t0
        interp = FWT.FwInterpolator(xc_grid, y_grid, w_grid, table)
        if args.go_residual is not None:
            t0 = time.time()
            FWT.subtract_go_reference(interp, w_ref=args.go_residual, n_workers=args.n_workers)
            t_go = time.time() - t0
        if args.table_cache:
            FWT.save_table(args.table_cache, interp, cfg)
    table = interp.table

    rows = []
    for M, (w, Fd) in truth.items():
        for k in range(n_pts):
            Ft = (FWT.F_hybrid(w, xc_pts[k], y_pts[k], interp, switch) if switch is not None
                  else interp(w, xc_pts[k], y_pts[k]))
            if not np.all(np.isfinite(Ft)):
                raise RuntimeError(f'non-finite F at x_c={xc_pts[k]}, y={y_pts[k]}, M={M}: '
                                   'a trimmed table row was queried')
            rows.append(dict(M=M, k=k, regime=points[k][2], xc=xc_pts[k], y=y_pts[k],
                             **err_metrics(Ft, Fd[:, k], w)))
    w_q = FWT.w_of(100.0, f_all[1:])
    t0 = time.perf_counter()
    for _ in range(200):
        interp(w_q, 0.3, 1.0)
    t_query = (time.perf_counter() - t0) / 200
    results['B'] = dict(n_grid=N, n_w=len(w_grid), dw=args.dw, npd=args.n_per_decade,
                        gl_schedule=args.gl_schedule, go_residual=args.go_residual, go_s=t_go,
                        table_gib=table.nbytes / 2**30, build_s=t_build,
                        query_s=t_query, rows=rows)
    print(f'\n=== Stage B: full table ({N}x{N}x{len(w_grid)}, GL {args.gl_schedule}, '
          f'GO residual {args.go_residual}, '
          f'{table.nbytes/2**30:.2f} GiB, built in {t_build:.0f}s, '
          f'query {t_query*1e3:.2f} ms/call) ===')
    summarize('bilinear (x_c,y) + linear w', rows)
    worst = sorted(rows, key=lambda r: -r['scaled'])[:5]
    print('  worst combos:')
    for r in worst:
        print(f"    M_Lz={r['M']:<8g} x_c={r['xc']:.4f} y={r['y']:.4f} {r['regime']:6s} "
              f"scaled={r['scaled']:.3%}  absmax={r['absmax']:.3g}  at w={r['w_worst']:.4g}")

json.dump(results, open(os.path.join(CHECKS, args.out + ('_quick' if args.quick else '') + '.json'), 'w'),
          indent=1, default=float)
print('\nSaved', os.path.join(CHECKS, args.out + ('_quick' if args.quick else '') + '.json'))
