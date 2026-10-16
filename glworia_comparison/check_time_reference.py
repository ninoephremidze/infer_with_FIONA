#!/usr/bin/env python
"""Does FIONA's F in the 'min_image' time reference agree with glworia's F as a complex number
(phase included), where the 'absolute' one differs by the lensing time shift? Compared at the
four benchmark injections on bilby's frequency grid (f >= 20 Hz).

Needs a patched glworia checkout with its CIS interpolation table built (see glworia_comparison/README.md):
  GLWORIA_PREFIX=/path/to/glworia python glworia_comparison/check_time_reference.py
"""
import json, os, sys
import finufft  # noqa: F401
import numpy as np
from scipy.constants import c, G
from infer_with_fiona import amplification as P, paths
ROOT = str(paths.REPO_ROOT)
GLW = os.environ.get('GLWORIA_PREFIX')
if not GLW:
    sys.exit('set GLWORIA_PREFIX to the root of a patched glworia checkout')
sys.path.append(GLW)
from glworia.amp.load_interp import load_interpolators, F_interp
MTOW = 8 * np.pi * G / c ** 3 * 1.9884099e30
cfg = json.load(open(os.path.join(GLW, 'PE_submission_settings/CIS/MLz_100.0_dL_500.0_y_0.3000_xc_0.0200.json')))
S = cfg['interpolator_settings']; I = load_interpolators(os.path.join(GLW, 'interpolation'), **S)
f = np.arange(4097) * 0.25; f = f[f >= 20.0]
Fs = {tr: P.HybridF(0.0, 1.0, 0.1, 5.0, 201, 0.1, 2000.0, 1024.0, str(paths.GL_RULES_DIR), 1,
                    root=ROOT, time_reference=tr, fourier_convention='fiona') for tr in ('absolute', 'min_image')}
print(f"{'case':18s} {'M_Lz':>6s} {'|F| only':>9s} {'complex, absolute':>18s} {'complex, min_image':>19s}")
for xc, y in [(0.02, 0.3), (0.2, 1.0), (0.2, 2.0), (0.2, 3.0)]:
    for M in (100.0, 1000.0):
        Fg = F_interp(f * MTOW * M, y, xc, I, S); w = MTOW * M * f
        scale = np.median(np.abs(Fg))
        out = []
        for tr in ('absolute', 'min_image'):
            Ff = Fs[tr](w, xc, y); out.append(np.max(np.abs(Ff - Fg)) / scale)
        mag = np.max(np.abs(np.abs(Fs['min_image'](w, xc, y)) - np.abs(Fg))) / scale
        print(f"x_c={xc:<5} y={y:<4}   {M:6g} {mag:9.2%} {out[0]:18.2%} {out[1]:19.2%}")
