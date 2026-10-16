#!/usr/bin/env python
"""For the ULA soliton or the point lens (PBH): (1) measure the FIONA -> bGO switch frequency
w_s(shape, y), as derive_bgo_switch_cis.py did for CIS, and (2) check that the v4 GL schedule,
derived for CIS, is also converged for this lens over the full w range of the PE prior.

Reference: FIONA with 2x the v4 node count at every w (capped at the largest cached rule,
1,024,000). Where v4 itself is already at the cap (w <~ 3e-4), the GL check instead compares
v4/2 with v4, which bounds v4's own error from above if convergence is monotone.

bGO: infer_with_fiona.fisher.generic_go.F_bGO, exactly as production (table.F_hybrid) uses it.
w_s = smallest sampled w >= 0.5 such that |F_bGO - F_ref| <= eps there and at every larger
sampled w up to w_max = 254; inf if never.

Output: bgo_switch_<lens>.npz with xc_grid (the shape-parameter grid; named as for CIS so
table.BgoSwitch reads it), y_grid, w_grid, err, w_s_<eps>, and the GL check
(gl_w, gl_err[w, shape, y], gl_mode). The point lens has one shape node (its fixed softening),
duplicated so the switch lookup has two rows.
"""
import argparse
import os
import time

import finufft  # noqa: F401 -- must precede any FIONA import
from infer_with_fiona import paths, quadrature  # noqa: E402 -- after finufft
import numpy as np

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--lens', choices=['ULA', 'PointLens'], required=True)
ap.add_argument('--n-workers', type=int, default=112)
ap.add_argument('--quick', action='store_true')
ap.add_argument('--gl-mult', type=int, default=1,
                help='Check the schedule mult x v4 (reference 2 mult x v4); 2 = the cis_x2 schedule, 4 = steep_core.')
args = ap.parse_args()

import infer_with_fiona.fiona.axisym as _axisym
import infer_with_fiona.fiona.utils as _fiona_utils
from infer_with_fiona.fiona.axisym import FresnelNUFHTLensBatched
from infer_with_fiona.fisher.generic_go import F_bGO
from infer_with_fiona.fisher.utils import G_over_c3_Msun
from infer_with_fiona.quadrature import schedule_cis as GL
from infer_with_fiona import table as FWT

ROOT = str(paths.REPO_ROOT)
CHECKS = str(paths.CHECKS_DIR); os.makedirs(CHECKS, exist_ok=True)   # outputs/checks
GL_DIR = str(paths.GL_RULES_DIR)
EPS_LIST = (1e-2, 3e-3, 1e-3)
W_MAX = 8.0 * np.pi * G_over_c3_Msun * 2000.0 * 1024.0          # ~253.6
N_CAP = 1024000
TAG = {'ULA': 'ula', 'PointLens': 'pbh'}[args.lens]

if args.lens == 'ULA':
    SH = (np.array([0.01, 0.1, 1.0]) if args.quick else
          np.unique(np.concatenate([np.geomspace(0.01, 0.1, 15), np.linspace(0.1, 1.0, 31)])))
else:
    SH = np.array([FWT.POINT_LENS_SOFTENING])
Y = np.array([0.1, 0.3, 1.0, 5.0]) if args.quick else np.linspace(0.1, 5.0, 99)
W_HI = np.geomspace(0.5, W_MAX, 8 if args.quick else 60)
W_LO = np.geomspace(2.0e-4, 0.5, 5 if args.quick else 25, endpoint=False)
W = np.concatenate([W_LO, W_HI])

GL.seed_gl_cache(GL_DIR, n_values=GL.REQUIRED_N_GL)
_fiona_utils._FIONA_GL2D_DIR = GL_DIR
lenses = [FWT.make_lens(args.lens, s) for s in SH]


def run(n_of_w, label):
    _axisym._adaptive_n_gl = n_of_w
    t0 = time.time()
    solver = FresnelNUFHTLensBatched(lenses, gl_dir=GL_DIR, n_workers=args.n_workers,
                                     print_timing=False)
    F = solver(W, Y)                                             # (n_w, n_shape, n_y)
    print(f'{label}: {len(W)} w x {len(SH)} shape x {len(Y)} y in {time.time()-t0:.0f}s',
          flush=True)
    return F


K = args.gl_mult
n_prod = lambda w: min(K * GL.adaptive_n_gl_v4(w), N_CAP)          # schedule under test
n_v4 = np.array([n_prod(w) for w in W])
F_v4 = run(n_prod, f'FIONA {K}x v4')
F_2x = run(lambda w: min(2 * n_prod(w), N_CAP), f'FIONA {2 * K}x v4')
capped = n_v4 >= N_CAP
F_half = run(lambda w: n_prod(w) // 2, f'FIONA {K}x v4 / 2') if capped.any() else None

# ─── GL check ───────────────────────────────────────────────────────────────────────────
gl_err = np.abs(F_v4 - F_2x)
gl_mode = np.where(capped, 'half_vs_v4', 'v4_vs_2x')
if capped.any():
    gl_err[capped] = np.abs(F_half[capped] - F_v4[capped])
mask = np.zeros((len(SH), len(Y)), dtype=bool)
if args.lens == 'ULA':
    mask = np.array([[bool(FWT.ula_prior_masked(s, y)) for y in Y] for s in SH])
print(f'\n=== GL {K}x v4 check ({args.lens}): max over unmasked (shape, y) of |dF| per w ===')
for iw, w in enumerate(W):
    e = gl_err[iw][~mask]
    print(f'  w={w:9.4g}  n_v4={n_v4[iw]:8d}  {gl_mode[iw]:11s}  max {e.max():.2e}  '
          f'median {np.median(e):.2e}' + ('   <-- > 1e-3' if e.max() > 1e-3 else ''))
e_all = gl_err[:, ~mask]
print(f'GL v4 overall: max {e_all.max():.2e}, 99th pct {np.percentile(e_all, 99):.2e}, '
      f'fraction > 1e-3: {(e_all > 1e-3).mean():.4f}', flush=True)

# ─── bGO switch ─────────────────────────────────────────────────────────────────────────
hi = slice(len(W_LO), len(W))
F_ref = F_2x[hi]
t0 = time.time()
F_b = np.empty_like(F_ref)
for i, s in enumerate(SH):
    for j, y in enumerate(Y):
        F_b[:, i, j] = F_bGO(lenses[i], float(y), W_HI)
print(f'bGO: {time.time()-t0:.0f}s', flush=True)
err = np.abs(F_b - F_ref)
out = dict(xc_grid=SH, y_grid=Y, w_grid=W_HI, err=err, gl_w=W, gl_err=gl_err,
           gl_mode=gl_mode, gl_n_v4=n_v4, masked=mask, lens=args.lens)
for eps in EPS_LIST:
    ok_suffix = np.minimum.accumulate((err <= eps)[::-1], axis=0)[::-1]
    w_s = np.full((len(SH), len(Y)), np.inf)
    for i in range(len(SH)):
        for j in range(len(Y)):
            idx = np.nonzero(ok_suffix[:, i, j])[0]
            if len(idx):
                w_s[i, j] = W_HI[idx[0]]
    out[f'w_s_{eps:g}'] = w_s
    print(f'\n=== w_s at eps={eps:g} (rows shape, cols y; "-" = never up to {W_MAX:.4g}) ===')
    step = max(1, len(Y) // 25)
    print('         ' + ''.join(f'{y:6.2f}' for y in Y[::step]))
    for i, s in enumerate(SH):
        print(f'{s:9.3g}' + ''.join(f'{v:6.3g}' if np.isfinite(v) else f'{"-":>6s}'
                                    for v in w_s[i, ::step]))
if len(SH) == 1:                                    # point lens: two identical rows
    out['xc_grid'] = np.array([SH[0], 2.0 * SH[0]])
    for k in list(out):
        if k.startswith('w_s_'):
            out[k] = np.repeat(out[k], 2, axis=0)
name = (f'bgo_switch_{TAG}' + (f'_glx{K}' if K > 1 else '') + ('_quick' if args.quick else '')
        + '.npz')
np.savez(os.path.join(CHECKS, name), **out)
print('Saved', os.path.join(CHECKS, name), '(production copies: data/bgo_switch/ula.npz, pbh.npz)')
print('\nSaved', name)
