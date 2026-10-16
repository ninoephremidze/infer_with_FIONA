#!/usr/bin/env python
"""Measure, on a fine (x_c, y) grid, the frequency w_s(x_c, y) above which production's bGO
formula stays within EPS of converged FIONA -- the basis for a lens-dependent FIONA -> bGO
switch replacing the fixed W_HIGH = 100.

Reference: FIONA with 2x the v4 node count at every w (v4 is itself accurate to 1e-3, so the
reference is well below that). bGO: fisherWithFIONA.generic_go.F_bGO, exactly as used by
PE_CIS_glworia_reproduction.F_of_theta.

w_s(x_c, y) = smallest sampled w such that |F_bGO - F_ref| <= EPS at it and at every larger
sampled w up to w_max = 254 (M_Lz = 2000 at 1024 Hz); inf if never.
Output: bgo_switch.npz (xc_grid, y_grid, w_grid, err, w_s for each eps).
"""
import argparse
import os
import time

import finufft  # noqa: F401 -- must precede any fisherWithFIONA/fiona import
import numpy as np

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--n-workers', type=int, default=112)
ap.add_argument('--out', default='bgo_switch')
ap.add_argument('--quick', action='store_true')
ap.add_argument('--fine', action='store_true',
                help='Finer (x_c, y) grid: 40 x_c x 99 y (dy = 0.05). The coarse default lets '
                     'masked caustic nodes veto bGO for unmasked neighbours through the '
                     'conservative 4-node lookup, forcing FIONA to w = 254 on most table rows.')
args = ap.parse_args()

import fiona.axisym as _axisym
import fiona.utils as _fiona_utils
from fiona.axisym import FresnelNUFHTLensBatched
from fiona.lenses import CIS
from fisherWithFIONA.generic_go import F_bGO
from fisherWithFIONA.utils import G_over_c3_Msun
import fiona_adaptive_gl_v4 as GL

ROOT = os.path.dirname(os.path.abspath(__file__))
GL_DIR = os.path.join(ROOT, 'fiona_gl_cache')
EPS_LIST = (1e-2, 3e-3, 1e-3)
W_MAX = 8.0 * np.pi * G_over_c3_Msun * 2000.0 * 1024.0          # ~253.6

if args.quick:
    XC = np.array([0.01, 0.5])
    Y = np.array([0.3, 5.0])
    W = np.geomspace(0.5, W_MAX, 8)
elif args.fine:
    XC = np.unique(np.concatenate([[1e-6], np.geomspace(1e-3, 0.1, 19), np.linspace(0.1, 1.0, 21)]))
    Y = np.linspace(0.1, 5.0, 99)
    W = np.geomspace(0.5, W_MAX, 60)
else:
    XC = np.array([1e-6, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2,
                   0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    Y = np.linspace(0.1, 5.0, 25)
    W = np.geomspace(0.5, W_MAX, 60)

GL.seed_gl_cache(GL_DIR, n_values=GL.REQUIRED_N_GL)
_axisym._adaptive_n_gl = lambda w: min(2 * GL.adaptive_n_gl_v4(w), 1024000)
_fiona_utils._FIONA_GL2D_DIR = GL_DIR

t0 = time.time()
solver = FresnelNUFHTLensBatched([CIS(psi0=1.0, xc=float(xc)) for xc in XC], gl_dir=GL_DIR,
                                 n_workers=args.n_workers, print_timing=False)
F_ref = solver(W, Y)                                             # (n_w, n_xc, n_y)
print(f'FIONA reference: {len(W)} w x {len(XC)} x_c x {len(Y)} y in {time.time()-t0:.0f}s',
      flush=True)

t0 = time.time()
F_b = np.empty_like(F_ref)
for i, xc in enumerate(XC):
    for j, y in enumerate(Y):
        F_b[:, i, j] = F_bGO(CIS(psi0=1.0, xc=max(float(xc), 1e-6)), float(y), W)
print(f'bGO: {time.time()-t0:.0f}s', flush=True)

err = np.abs(F_b - F_ref)
out = dict(xc_grid=XC, y_grid=Y, w_grid=W, err=err)
for eps in EPS_LIST:
    ok = err <= eps
    ok_suffix = np.minimum.accumulate(ok[::-1], axis=0)[::-1]
    w_s = np.full((len(XC), len(Y)), np.inf)
    for i in range(len(XC)):
        for j in range(len(Y)):
            idx = np.nonzero(ok_suffix[:, i, j])[0]
            if len(idx):
                w_s[i, j] = W[idx[0]]
    out[f'w_s_{eps:g}'] = w_s
    print(f'\n=== w_s at eps={eps:g} (rows x_c, cols y; "-" = never up to {W_MAX:.4g}) ===')
    print('        ' + ''.join(f'{y:6.2f}' for y in Y))
    for i, xc in enumerate(XC):
        print(f'{xc:8.3g}' + ''.join(f'{v:6.3g}' if np.isfinite(v) else f'{"-":>6s}'
                                     for v in w_s[i]))
np.savez(os.path.join(ROOT, args.out + ('_quick' if args.quick else '') + '.npz'), **out)
print('\nSaved', args.out + ('_quick' if args.quick else '') + '.npz')
