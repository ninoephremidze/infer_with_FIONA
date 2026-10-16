"""Pre-generate Gauss-Legendre rules in FIONA's on-disk node cache.

FIONA falls back to numpy.leggauss, which is O(n^3), the first time it needs a rule with
n nodes. seed_gl_cache generates any missing rules with scipy.special.roots_legendre
instead and writes them in FIONA's own cache layout. The rules used in the paper, n =
250 * 2^k up to 1,024,000, ship in data/gl_rules/, so this is normally a no-op.
"""


def seed_gl_cache(gl_dir, u_max=1.0, n_values=(), verbose=True):
    """Pre-generate any missing GL rules with scipy (O(n)) instead of letting FIONA fall
    back to numpy.leggauss (O(n^3)) on first use. Writes into FIONA's own cache layout via
    its own path helpers, data files first and meta last, so a partially written entry is
    never seen as present."""
    import os, json, time, hashlib, tempfile
    import numpy as np
    from scipy.special import roots_legendre
    from ..fiona import utils as U

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
