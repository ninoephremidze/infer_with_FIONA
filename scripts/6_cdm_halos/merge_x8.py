"""Merge the 8x-schedule reruns of the high-mass halos into the convergence pair read by fiona_cdm_rate_post.py.
For masses rerun with x8: result = x8, check = x4 (steep_core). Elsewhere: result = x4, check = x2 (cis_x2), as before.
Writes fiona_cdm_rate_<name>_{steep_core,cis_x2}_imrhi.npz (post step: IN_TAG=_imrhi).
  python merge_x8.py SMBHB_1e4 [...]"""
import os, sys
import numpy as np
from infer_with_fiona import paths
os.chdir(os.environ.get('CDM_RAW', str(paths.OUTPUTS_DIR / 'cdm')))
PER_M = ('dchi2', 'tbest', 'F0', 'dF', 'fish', 'done', 'c200', 'kappa', 'rF_pc')
for name in sys.argv[1:]:
    x4 = dict(np.load(f'fiona_cdm_rate_{name}_steep_core_imr.npz', allow_pickle=True))
    x2 = dict(np.load(f'fiona_cdm_rate_{name}_cis_x2_imr.npz', allow_pickle=True))
    A, B = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in x4.items()}, \
           {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in x2.items()}
    mi_all = []
    for suf in ('x8_imr', 'x8lo_imr'):                  # high-mass and onset-mass reruns
        if not os.path.exists(f'fiona_cdm_rate_{name}_{suf}.npz'):
            continue
        x8 = dict(np.load(f'fiona_cdm_rate_{name}_{suf}.npz', allow_pickle=True))
        mi = np.where(x8['done'].all(axis=0))[0]
        for k in PER_M:
            A[k][:, mi] = x8[k][:, mi]
            B[k][:, mi] = x4[k][:, mi]
        mi_all += list(mi)
    mi = np.array(sorted(mi_all)); x8 = A
    np.savez(f'fiona_cdm_rate_{name}_steep_core_imrhi.npz', **A)
    np.savez(f'fiona_cdm_rate_{name}_cis_x2_imrhi.npz', **B)
    k1 = list(np.round(x4['b_rF'], 3)).index(1.0)
    print(name, 'rerun masses', np.round(np.log10(x4['M200'][mi]), 2))
    for i in mi:
        r = lambda a: a['dchi2'][2, i, k1]
        print(f'  M {x4["M200"][i]:.2e}  b=r_F z~0.94: x2 {r(x2):.4g}  x4 {r(x4):.4g}  x8 {r(x8):.4g}  |x8-x4|/x8 {abs(r(x8)-r(x4))/abs(r(x8)):.3f}')
