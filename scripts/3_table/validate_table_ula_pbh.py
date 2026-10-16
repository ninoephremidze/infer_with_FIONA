#!/usr/bin/env python
"""Build (or load) the production F table for the ULA soliton or the point lens and validate
the full production chain (amplification.HybridF: table + GO residual + lens-dependent bGO switch)
against direct FIONA, as validate_table_cis.py did for CIS (job 48205398).

Test points: N random (shape, y) in the prior, outside the prior mask (ULA caustic band), each
with a random M_Lz from the log-uniform prior [0.1, 2000], evaluated on the production
frequencies 20-1024 Hz (every --f-stride-th 0.25 Hz bin), w = Mtow M_Lz f. The injection
(shape_fid, y = 0.3, M_Lz = 100 and 2000) is always included.
Reference: direct FIONA with the same GL schedule (so this isolates the table's own error),
in the same time reference and Fourier convention as HybridF.
Metric per test point: max |F_hybrid - F_direct| / median |F_direct| (as for CIS), and the
absolute max |dF|.

Run with the production settings, so the table it caches is the one the PE runs load.
"""
import argparse
import json
import os
import time

import finufft  # noqa: F401 -- must precede any FIONA import
from infer_with_fiona import paths, quadrature  # noqa: E402 -- after finufft
import numpy as np
from scipy.constants import c as C_SI, G as G_SI

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--lens', choices=['ULA', 'PBH'], required=True)
ap.add_argument('--gl-schedule', default='steep_core', choices=quadrature.CHOICES)
ap.add_argument('--n-grid', type=int, default=201)
ap.add_argument('--n-test', type=int, default=120)
ap.add_argument('--f-stride', type=int, default=4)
ap.add_argument('--n-workers', type=int, default=112)
ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--fail-above', type=float, default=0.02,
                help='Exit non-zero if the worst max|dF|/median|F| exceeds this, so PE jobs '
                     'submitted with --dependency=afterok do not start on a bad table. (default: 0.02)')
args = ap.parse_args()

from infer_with_fiona import amplification as P
FWT = P.FWT
from infer_with_fiona.fiona.axisym import FresnelNUFHTLensBatched

ROOT = str(paths.REPO_ROOT)
GL_DIR = str(paths.GL_RULES_DIR)
LM = {'ULA': 'ULA', 'PBH': 'PointLens'}[args.lens]
SW = str(paths.BGO_SWITCH[args.lens])
SH_MIN, SH_MAX, SH_FID = ((0.01, 1.0, 0.1) if LM == 'ULA' else (FWT.POINT_LENS_SOFTENING,) * 3)
MTOW = 8.0 * np.pi * G_SI / C_SI ** 3 * 1.9884099e+30        # glworia's, as the PE scripts

t0 = time.time()
F = P.HybridF(SH_MIN, SH_MAX, 0.1, 5.0, args.n_grid, 0.1, 2000.0, 1024.0, GL_DIR,
              args.n_workers, root=ROOT, gl_schedule=args.gl_schedule, switch_path=SW,
              lens_model=LM)
print(f'[validate] table ready in {time.time()-t0:.0f}s', flush=True)

rng = np.random.default_rng(args.seed)
tests = [(SH_FID, 0.3, 100.0), (SH_FID, 0.3, 2000.0)]
masked = FWT.prior_masked_fn(LM)
while len(tests) < args.n_test:
    s = rng.uniform(SH_MIN, SH_MAX) if LM == 'ULA' else SH_FID
    y = rng.uniform(0.1, 5.0)
    if masked(s, y):
        continue
    tests.append((s, y, float(np.exp(rng.uniform(np.log(0.1), np.log(2000.0))))))
freqs = np.arange(20.0, 1024.0 + 1e-9, 0.25)[::args.f_stride]

rows = []
for k, (s, y, M) in enumerate(tests):
    w = MTOW * M * freqs
    w = w[w <= F.w_max]
    t1 = time.time()
    solver = FresnelNUFHTLensBatched([FWT.make_lens(LM, s)], gl_dir=GL_DIR,
                                     n_workers=args.n_workers, print_timing=False)
    Fd = solver(w, np.array([y]))[:, 0, 0]
    Fd = P.to_fourier_convention(Fd * P.time_reference_factor(w, s, y, 'min_image', LM), 'lal')
    Fh = F(w, s, y)
    d = np.abs(Fh - Fd)
    rel = float(d.max() / np.median(np.abs(Fd)))
    w_s = F.switch.w_switch(s, y)
    rows.append(dict(shape=s, y=y, M_Lz=M, w_max=float(w.max()), w_s=w_s, rel=rel,
                     abs_max=float(d.max()), w_at_max=float(w[d.argmax()])))
    print(f'  [{k+1:3d}/{len(tests)}] shape={s:.4g} y={y:.3f} M_Lz={M:8.2f} w<={w.max():7.2f} '
          f'w_s={w_s:7.3g}  max|dF|/med|F|={rel:.3%}  max|dF|={d.max():.2e} at w={w[d.argmax()]:.3g}  '
          f'({time.time()-t1:.0f}s)', flush=True)

rel = np.array([r['rel'] for r in rows])
print(f'\n[validate] {args.lens} ({args.gl_schedule}), {len(rows)} test points: worst {rel.max():.2%}, '
      f'95th {np.percentile(rel, 95):.2%}, median {np.median(rel):.3%}')
os.makedirs(paths.CHECKS_DIR, exist_ok=True)
out = os.path.join(paths.CHECKS_DIR, f'table_validation_{args.lens.lower()}.json')
json.dump(dict(lens=args.lens, gl_schedule=args.gl_schedule, n_grid=args.n_grid,
               rows=rows), open(out, 'w'), indent=1)
print('Saved', out)
if rel.max() > args.fail_above:
    print(f'[validate] FAIL: worst {rel.max():.2%} > {args.fail_above:.0%}', flush=True)
    raise SystemExit(1)
print('[validate] PASS', flush=True)
