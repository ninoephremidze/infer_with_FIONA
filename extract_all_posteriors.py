#!/usr/bin/env python
"""Cache the posterior samples of every FIONA lal run (CIS, ULA, PBH) for the 0-all_*_results notebooks.

A run is read from its final result.json if it exists, otherwise from the latest dynesty checkpoint (dead points,
importance-resampled to equal weights). Loading a checkpoint takes minutes and up to ~10 GB, so this runs as a
Slurm job (submit_extract_all_posteriors.sh) with one process per run. Output:
  all_results_figs/cache/<label>.npz   samples of the sampled parameters + status info (json string)

Run: sbatch submit_extract_all_posteriors.sh
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import dill
import numpy as np
from dynesty.utils import resample_equal

BASE = Path('full_parameter_inference')
CACHE = Path('all_results_figs/cache')
GW = ['mass_ratio', 'chirp_mass', 'luminosity_distance', 'theta_jn', 'psi', 'phase', 'a_1', 'a_2']
LENS_ORDER = dict(CIS=['MLz', 'y', 'lp'], ULA=['MLz', 'y', 'lp'], PBH=['MLz', 'y'])


def extract(d):
    lens, label = d.parent.name.split('_')[0], d.name
    order = GW + LENS_ORDER[lens] + ['H1_time', 'zenith', 'azimuth']
    t0 = time.time()
    res_json = d / f'{label}_result.json'
    if res_json.exists():
        r = json.load(open(res_json))
        post = {k: np.asarray(r['posterior']['content'][k], float) for k in order}
        info = dict(status='final', iterations=r.get('num_likelihood_evaluations'), dlogz=float('nan'),
                    ESS=len(post['y']), log_evidence=r['log_evidence'], file_mtime=res_json.stat().st_mtime)
    else:
        pk = d / f'{label}_resume.pickle'
        for attempt in range(5):  # the checkpoint can be mid-write
            try:
                with open(pk, 'rb') as f:
                    obj = dill.load(f)
                break
            except Exception as e:
                print(f'{label}: retry ({e!r})', flush=True); time.sleep(60)
        else:
            return f'{label}: FAILED to read checkpoint'
        sampler = next(o for o in (obj if isinstance(obj, tuple) else (obj,)) if hasattr(o, 'results'))
        res = sampler.results
        w = np.exp(res.logwt - res.logz[-1]); w /= w.sum()
        eq = resample_equal(res.samples, w, rstate=np.random.default_rng(0))
        logl_max = np.max(sampler.live_logl) if hasattr(sampler, 'live_logl') else np.max(res.logl)
        post = {k: eq[:, i] for i, k in enumerate(order)}
        info = dict(status='in progress', iterations=int(len(res.logl)),
                    dlogz=float(np.logaddexp(0, logl_max + res.logvol[-1] - res.logz[-1])),
                    ESS=float(1.0 / np.sum(w ** 2)), log_evidence=float(res.logz[-1]),
                    file_mtime=pk.stat().st_mtime)
    info['extracted'] = time.time()
    np.savez(CACHE / f'{label}.npz', info=json.dumps(info), **post)
    return f'{label}: {info["status"]}, dlogz {info["dlogz"]:.3g}, {time.time() - t0:.0f} s'


if __name__ == '__main__':
    CACHE.mkdir(parents=True, exist_ok=True)
    pattern = sys.argv[1] if len(sys.argv) > 1 else '*'
    dirs = sorted(d for lens in LENS_ORDER for d in (BASE / f'{lens}_glworia_repro').glob(f'{lens}_*lal_0')
                  if d.is_dir() and Path(d.name).match(pattern))
    print(f'{len(dirs)} runs', flush=True)
    with Pool(len(dirs)) as p:
        for msg in p.imap_unordered(extract, dirs):
            print(msg, flush=True)
