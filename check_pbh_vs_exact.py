#!/usr/bin/env python
"""FIONA's softened point lens (eps = 1e-5) with the v4 and v4x2 GL schedules against the
exact point-mass amplification factor
  F = exp(pi w/4 + i w/2 [ln(w/2) - 2 phi_m]) Gamma(1 - i w/2) 1F1(i w/2; 1; i w y^2/2),
phi_m = (x_m - y)^2/2 - ln x_m, x_m = (y + sqrt(y^2 + 4))/2 (time origin at the minimum
image; FIONA's F is moved to the same origin with exp(-i w phi_min))."""
import finufft  # noqa: F401
import numpy as np
import mpmath as mp
import fiona.axisym as _axisym
import fiona.utils as _fiona_utils
from fiona.axisym import FresnelNUFHTLensBatched
import fiona_adaptive_gl_v4 as GL
import fiona_w_table as FWT

GL_DIR = 'fiona_gl_cache'
_fiona_utils._FIONA_GL2D_DIR = GL_DIR
mp.mp.dps = 40
W = np.concatenate([np.geomspace(2e-4, 0.5, 8, endpoint=False), np.geomspace(0.5, 253.6, 16)])
Y = np.array([0.1, 0.3, 1.0, 3.0, 5.0])


def F_exact(w, y):
    xm = (y + mp.sqrt(y * y + 4)) / 2
    phim = (xm - y) ** 2 / 2 - mp.log(xm)
    hw = mp.mpf(w) / 2
    v = (mp.exp(mp.pi * w / 4 + 1j * hw * (mp.log(hw) - 2 * phim)) * mp.gamma(1 - 1j * hw)
         * mp.hyp1f1(1j * hw, 1, 1j * hw * y * y))
    return complex(v)


Fe = np.array([[F_exact(w, y) for y in Y] for w in W])
lens = [FWT.make_lens('PointLens', FWT.POINT_LENS_SOFTENING)]
phimin = np.array([FWT.image_set('PointLens', 1e-5, y)[0].min() for y in Y])
for K in (1, 2, 4):
    _axisym._adaptive_n_gl = lambda w, K=K: min(K * GL.adaptive_n_gl_v4(w), 1024000)
    F = FresnelNUFHTLensBatched(lens, gl_dir=GL_DIR, n_workers=56, print_timing=False)(W, Y)[:, 0, :]
    F = F * np.exp(-1j * W[:, None] * phimin[None, :])
    e = np.abs(F - Fe)
    print(f'\n=== {K}x v4 vs exact point lens: max over y of |dF| ===')
    for iw, w in enumerate(W):
        print(f'  w={w:9.4g}  n={min(K * GL.adaptive_n_gl_v4(w), 1024000):8d}  max {e[iw].max():.2e}  '
              f'(worst y={Y[e[iw].argmax()]:g})')
    print(f'  overall max {e.max():.2e}')
