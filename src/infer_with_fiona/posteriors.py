"""Posterior samples of the paper's inference runs.

Two sources are supported:

  'shipped'  the equal-weight samples in results/posteriors/{fiona,glworia}/<run>.npz (default)
  'bilby'    the full output of a fresh run: <run>_result.json if the run has finished, otherwise
             its dynesty checkpoint (<run>_resume.pickle), importance-resampled to equal weights

FIONA runs are named by their output label, e.g.
  CIS_glworia_repro_mlz100_xc0.02_y0.3_high_mlzprior0.1-2000.0_log-uniform_wtabv4lal_0,
and live in <runs_dir>/<lens>_glworia_repro/<label>/ (runs_dir defaults to outputs/runs).
glworia runs are named by their configuration, e.g.
  MLz_100.0_dL_500.0_y_0.3000_xc_0.0200_lal_loguniform,
and live where glworia's scripts/PE_full.py writes them:
  <runs_dir>/CIS_PE_<strong|weak>_lal_loguniform/CIS_PE_<strong|weak>_lal_loguniform/<name>/.
"""
import json
from pathlib import Path

import numpy as np

from .paths import POSTERIORS_DIR, RUNS_DIR

GW_PARAMS = ['mass_ratio', 'chirp_mass', 'luminosity_distance', 'theta_jn', 'psi', 'phase', 'a_1', 'a_2']
LENS_PARAMS = {'CIS': ['MLz', 'y', 'lp'], 'ULA': ['MLz', 'y', 'lp'], 'PBH': ['MLz', 'y']}
TAIL_PARAMS = ['H1_time', 'zenith', 'azimuth']

# The injection shared by every run (a GW150914-like 30 + 30 Msun binary at 500 Mpc, as in C24).
# zenith and azimuth are the detector-frame sky coordinates sampled instead of (ra, dec).
INJECTION = dict(mass_1=30.0, mass_2=30.0, mass_ratio=1.0, chirp_mass=26.11651689888372,
                 luminosity_distance=500.0, theta_jn=np.pi / 3, psi=np.pi / 2, phase=np.pi / 2,
                 a_1=0.0, a_2=0.0, ra=1.375, dec=-1.2108, geocent_time=1126259642.413,
                 zenith=2.3666, azimuth=2.9995, MLz=100.0, y=0.3)


def search_order(lens):
    """The order of the sampled parameters in the dynesty samples (needed to read a checkpoint)."""
    return GW_PARAMS + LENS_PARAMS[lens] + TAIL_PARAMS


def fiona_label(lens, y, shape=None, mlz=100.0):
    """Output label of a FIONA run in the paper (shape: x_c for CIS, r_s for ULA, unused for PBH)."""
    if lens == 'CIS':
        return f'CIS_glworia_repro_mlz{mlz:g}_xc{shape:g}_y{y:.1f}_high_mlzprior0.1-2000.0_log-uniform_wtabv4lal_0'
    if lens == 'ULA':
        return (f'ULA_glworia_repro_mlz{mlz:g}_rs{shape:g}_y{y:g}_high_mlzprior0.1-2000.0'
                '_log-uniform_wtabv4x4lal_0')
    if lens == 'PBH':
        return f'PBH_glworia_repro_mlz{mlz:g}_y{y:g}_high_mlzprior0.1-2000.0_log-uniform_wtabv4x4lal_0'
    raise ValueError(f'unknown lens {lens!r}')


def glworia_name(y, xc, mlz=100.0):
    """Configuration name of a glworia CIS run in the paper."""
    return f'MLz_{mlz:.1f}_dL_500.0_y_{y:.4f}_xc_{xc:.4f}_lal_loguniform'


def _from_checkpoint(pickle_path, order):
    import dill
    from dynesty.utils import resample_equal
    with open(pickle_path, 'rb') as f:
        obj = dill.load(f)
    sampler = next(o for o in (obj if isinstance(obj, tuple) else (obj,)) if hasattr(o, 'results'))
    res = sampler.results
    w = np.exp(res.logwt - res.logz[-1])
    w /= w.sum()
    eq = resample_equal(res.samples, w, rstate=np.random.default_rng(0))
    logl_max = np.max(sampler.live_logl) if hasattr(sampler, 'live_logl') else np.max(res.logl)
    dlogz = float(np.logaddexp(0, logl_max + res.logvol[-1] - res.logz[-1]))
    info = dict(status=f'checkpoint, {len(res.logl)} iterations, dlogz {dlogz:.3g}',
                log_evidence=float(res.logz[-1]))
    return {k: eq[:, i] for i, k in enumerate(order)}, info


def _from_bilby_dir(d, name, order):
    d = Path(d)
    res = d / f'{name}_result.json'
    if res.exists():
        r = json.load(open(res))
        post = {k: np.asarray(v, float) for k, v in r['posterior']['content'].items() if k in order}
        info = dict(status='final', log_evidence=r.get('log_evidence'),
                    log_bayes_factor=r.get('log_bayes_factor'), sampling_time=r.get('sampling_time'),
                    num_likelihood_evaluations=r.get('num_likelihood_evaluations'))
        return post, info
    pk = d / f'{name}_resume.pickle'
    if pk.exists():
        return _from_checkpoint(pk, order)
    raise FileNotFoundError(f'no {name}_result.json or {name}_resume.pickle in {d}')


def _from_npz(path):
    d = np.load(path)
    info = json.loads(str(d['info']))
    return {k: np.asarray(d[k], float) for k in d.files if k != 'info'}, info


def load_fiona(label, source='shipped', runs_dir=None):
    """(samples dict, info dict) of a FIONA run, by output label."""
    lens = label.split('_')[0]
    if source == 'shipped':
        return _from_npz(POSTERIORS_DIR / 'fiona' / f'{label}.npz')
    base = Path(runs_dir) if runs_dir else RUNS_DIR
    return _from_bilby_dir(base / f'{lens}_glworia_repro' / label, label, search_order(lens))


def load_glworia(name, source='shipped', runs_dir=None):
    """(samples dict, info dict) of a glworia CIS run, by configuration name."""
    if source == 'shipped':
        return _from_npz(POSTERIORS_DIR / 'glworia' / f'{name}.npz')
    base = Path(runs_dir) if runs_dir else RUNS_DIR / 'glworia'
    regime = 'strong' if '_y_0.3000_' in name else 'weak'
    ext = f'CIS_PE_{regime}_lal_loguniform'
    return _from_bilby_dir(base / ext / ext / name, name, search_order('CIS'))


def add_source_arguments(ap):
    """Add the --from-runs / --runs-dir / --glworia-dir / --out-dir options shared by the plot scripts."""
    from .paths import FIGURES_DIR
    ap.add_argument('--from-runs', action='store_true',
                    help='Read full bilby output of fresh runs instead of the shipped samples.')
    ap.add_argument('--runs-dir', default=None,
                    help='With --from-runs: directory holding <lens>_glworia_repro/ (default: outputs/runs).')
    ap.add_argument('--glworia-dir', default=None,
                    help="With --from-runs: glworia's outdir, holding CIS_PE_*_lal_loguniform/ "
                         '(default: outputs/runs/glworia).')
    ap.add_argument('--out-dir', default=str(FIGURES_DIR), help='(default: outputs/figures)')
    return ap
