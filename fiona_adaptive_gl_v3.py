"""Overlay patch for FIONA's adaptive Gauss-Legendre node-count schedule (CIS lens), v3.

Does NOT modify any file under 0-fiona/. `fiona.axisym._adaptive_n_gl` is a plain
module-level function; reassigning the module attribute at runtime only changes what this
process sees, never the .py file on disk.

WHY v3 SUPERSEDES v2
--------------------
v2 was derived over w in [0.5, 12] only (anchored to M_Lz=100) and then extended by hand:
a flat 500 floor below w=1.5 and a flat clamp at 8000 above w=13. Both hand-extensions were
wrong, and the clamp rested on a false premise:

  - The "performance cliff above n_gl~10000" that motivated v2's clamp does not exist. GL
    nodes are cached to disk per distinct n (fiona/utils.py). The apparent cliff was a
    cache HIT (n=10000, pre-cached) vs a cache MISS (n=12000, not cached) -- generation,
    not evaluation. Measured recurring per-call cost is nearly flat in n_gl:
    500 -> 0.0090s, 8000 -> 0.0091s, 10000 -> 0.0090s, 16000 -> 0.0113s.
  - Generation is O(n^3) with numpy.leggauss (n=32000 would be ~50 min) but O(n) with
    scipy.special.roots_legendre (n=32000 in 26s, agreeing to 2e-16 on nodes and 6e-13 on
    weights). So any n is cheap to adopt: seed the cache once with scipy.

Measured consequences of the v2 hand-extensions, against glworia:
  - low w: at w~0.01 the 500 floor gives |F| off by ~20%; n_gl=2000 -> 0.9%, 8000 -> 0.14%.
  - high w: in w in [20,100], the 8000 clamp gives up to 10.2% error (M_Lz=2000, strong);
    raising to 32000 brings the same cases to 4.1%, and every weak-lensing case to <=1.1%.

DERIVATION
----------
derive_adaptive_gl_schedule_v3.py, cap 32000, over w in [2.5e-4, 100] -- the range the
production prior actually reaches (M_Lz in [0.1,2000] over f in [20,1024] Hz spans
w in [2.5e-4, 253.6]; above w=100 production uses the exact closed-form bGO sum instead,
so no schedule entry is needed there). Same 10 (x_c,y) points as v1/v2, 1% self-convergence
tolerance, doubling from n_gl=250. Tiers below are the max requirement over all 10 points.

KNOWN REMAINING GAPS (both outside or at the edge of the production prior)
--------------------------------------------------------------------------
  - w < 0.003: 6-8 of 10 points do not reach 1% self-convergence even at 32000. This
    corresponds to M_Lz < ~1 Msun at the bottom of the band -- BELOW the production
    log-uniform [1,2000] prior, whose w_min is 2.5e-3. Flagged, not fixed.
  - w >= 80: 1/10 points at w=80 and 5/10 at w=100 do not reach 1% self-convergence at
    32000. Direct comparison against glworia is nonetheless ~4% worst-case there
    (M_Lz=2000, strong), so the self-convergence test is stricter than the realized error.
    Note also that pairwise self-convergence is a weak criterion in this oscillatory
    regime and can both false-pass and false-fail.

Usage: call `apply()` once, early, before any fisherWithFIONA/fiona CIS computation in a
given process. `restore()` undoes it. Seed the node cache first (see seed_gl_cache) or the
first call at each new n pays numpy's O(n^3) generation.
"""
import fiona.axisym as _axisym

_ORIGINAL_ADAPTIVE_N_GL = _axisym._adaptive_n_gl

# Node counts this schedule can request -- seed these into FIONA's GL cache with scipy
# (see seed_gl_cache) to avoid paying numpy's O(n^3) generation on first use.
REQUIRED_N_GL = (500, 1000, 2000, 4000, 8000, 16000, 32000)


def adaptive_n_gl_v3(w_abs):
    if w_abs < 0.003:
        return 32000   # below the production prior; does NOT fully converge even here
    elif w_abs < 0.03:
        return 16000
    elif w_abs < 0.1:
        return 4000
    elif w_abs < 2.0:
        return 500
    elif w_abs < 3.0:
        return 1000
    elif w_abs < 5.0:
        return 2000
    elif w_abs < 9.0:
        return 4000
    elif w_abs < 20.0:
        return 8000
    elif w_abs < 40.0:
        return 16000
    else:
        return 32000   # w>=100 is handled by the exact bGO sum in production anyway


def seed_gl_cache(gl_dir, u_max=1.0, n_values=REQUIRED_N_GL, verbose=True):
    """Pre-generate any missing GL rules with scipy (O(n)) instead of letting FIONA fall
    back to numpy.leggauss (O(n^3)) on first use. Writes into FIONA's own cache layout via
    its own path helpers, data files first and meta last, so a partially written entry is
    never seen as present."""
    import os, json, time, hashlib, tempfile
    import numpy as np
    from scipy.special import roots_legendre
    import fiona.utils as U

    # fiona.utils captures _FIONA_GL2D_DIR at IMPORT time (utils.py line 31), so setting
    # the env var alone is too late if anything already imported it. Patch both, and
    # restore afterwards so we don't disturb the caller's environment.
    os.environ['FIONA_GL2D_DIR'] = str(gl_dir)
    _saved_dir = U._FIONA_GL2D_DIR
    U._FIONA_GL2D_DIR = str(gl_dir)
    try:
        _seed_inner(U, gl_dir, u_max, n_values, verbose,
                    np, roots_legendre, os, json, time, hashlib, tempfile)
    finally:
        U._FIONA_GL2D_DIR = _saved_dir


def _seed_inner(U, gl_dir, u_max, n_values, verbose,
                np, roots_legendre, os, json, time, hashlib, tempfile):
    for n in n_values:
        if U._gl1d_exists(n, u_max):
            continue
        paths = U._gl1d_paths(n, u_max)
        if paths is None:
            raise RuntimeError(
                f'FIONA could not resolve its GL cache dir for {gl_dir!r} '
                '(fiona.utils._FIONA_GL2D_DIR unset or directory not creatable).')
        t0 = time.perf_counter()
        xi, wi = roots_legendre(n)
        x, w = u_max * xi, u_max * wi
        build = time.perf_counter() - t0

        def _atomic(path, arr):
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp'); os.close(fd)
            np.save(tmp, arr, allow_pickle=False)
            os.replace(tmp + '.npy' if os.path.exists(tmp + '.npy') else tmp, path)

        _atomic(paths['x'], x)
        _atomic(paths['w'], w)

        def _sha(p):
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                for b in iter(lambda: f.read(1 << 22), b''):
                    h.update(b)
            return h.hexdigest()

        meta = {'version': 1, 'n': int(n), 'u_max': float(u_max), 'dtype': 'float64',
                'dim': 1, 'build_time_sec': build, 'cores_used': 1,
                'files': {k: paths[k].name for k in ('x', 'w')},
                'sha256': {k: _sha(paths[k]) for k in ('x', 'w')},
                'hostname': os.uname().nodename, 'created': time.time(),
                'note': 'scipy.special.roots_legendre (O(n)); numpy leggauss is O(n^3)'}
        tmpm = str(paths['meta']) + '.tmp'
        with open(tmpm, 'w') as f:
            json.dump(meta, f, indent=2)
        os.replace(tmpm, paths['meta'])
        if verbose:
            print(f'[seed_gl_cache] generated n={n} in {build:.1f}s')


def apply():
    """Activate the v3 schedule for this process. Idempotent."""
    _axisym._adaptive_n_gl = adaptive_n_gl_v3


def restore():
    """Restore FIONA's own original schedule for this process."""
    _axisym._adaptive_n_gl = _ORIGINAL_ADAPTIVE_N_GL


def is_active():
    return _axisym._adaptive_n_gl is adaptive_n_gl_v3
