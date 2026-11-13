"""Run fiona_cdm_rate.py with an 8x-v4 Gauss-Legendre schedule (twice steep_core = v4x4), capped at 1,024,000 nodes
(the largest shipped rule). Scratch only: the schedule is patched in this process, the FIONA repo is not touched.
  python fiona_cdm_rate_x8.py SMBHB_1e5 --midx 29,30 --out ...      (other args as fiona_cdm_rate.py; --sched ignored)
"""
import finufft  # noqa: F401
import os, sys, runpy, types
from infer_with_fiona import quadrature
from infer_with_fiona.quadrature import schedule_cis as v4
from infer_with_fiona.fiona import axisym

N_CAP = 1024000
def n_x8(w_abs):
    return min(8 * v4.adaptive_n_gl_v4(w_abs), N_CAP)
X8 = types.SimpleNamespace(apply=lambda: setattr(axisym, '_adaptive_n_gl', n_x8))
quadrature.get_schedule = lambda name: X8
sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fiona_cdm_rate.py')] + sys.argv[1:] + ['--sched', 'cis_x8']
runpy.run_path(sys.argv[0], run_name='__main__')
