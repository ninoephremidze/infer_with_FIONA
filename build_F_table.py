#!/usr/bin/env python
"""Build (or confirm) the cached F(w, x_c, y) table a CIS PE run with the given M_Lz prior and
grid resolution will load, WITHOUT running PE. Run it once before submitting several PE jobs
that need a table not yet in fiona_w_table_cache/, so they don't all build it at once.

Uses exactly the configuration PE_CIS_glworia_reproduction.py passes to fiona_pe_F.HybridF
(x_c in [0, 1], y in [0.1, 5], f_max = 1024 Hz, v4 schedule, GO residual, bgo_switch_fine).

Usage: python build_F_table.py --m-lz-prior-min 50 --m-lz-prior-max 150 [--grid-res high]
"""
import argparse
import os

import finufft  # noqa: F401 -- must precede any fisherWithFIONA/fiona import

import fiona_pe_F

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--m-lz-prior-min', type=float, default=0.1)
ap.add_argument('--m-lz-prior-max', type=float, default=2000.0)
ap.add_argument('--grid-res', choices=['tiny', 'low', 'medium', 'high'], default='high')
ap.add_argument('--n-workers', type=int, default=112)
args = ap.parse_args()

ROOT = os.path.dirname(os.path.abspath(__file__))
N_GRID = {'tiny': 21, 'low': 61, 'medium': 101, 'high': 201}[args.grid_res]
F = fiona_pe_F.HybridF(0.0, 1.0, 0.1, 5.0, N_GRID, args.m_lz_prior_min, args.m_lz_prior_max,
                       1024.0, os.path.join(ROOT, 'fiona_gl_cache'), args.n_workers, root=ROOT)
print(f'[F] table ready: {F.interp.table.shape}, w in [{F.w_grid[0]:.3g}, {F.w_max:g}]')
