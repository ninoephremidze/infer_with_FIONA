"""GL schedule v4x4: four times the v4 node count at every w (capped at 1,024,000, the largest
cached rule), for lenses with a much steeper core than any CIS in the v4 derivation. FIONA's
error then falls about 2x per doubling of n: for the softened point lens (eps = 1e-5) against
the exact point-mass F, max |dF| is 3.2e-3 (v4), 1.6e-3 (v4x2), 8.0e-4 (v4x4) over
w in [2e-4, 254] and y in [0.1, 5] (check_pbh_vs_exact.py); the ULA soliton at r_s = 0.01
behaves the same way (derive_bgo_switch_lens.py). Same API as fiona_adaptive_gl_v4."""
import fiona.axisym as _axisym

import fiona_adaptive_gl_v4 as _v4
from fiona_adaptive_gl_v3 import seed_gl_cache  # noqa: F401 -- same cache writer

N_CAP = 1024000
REQUIRED_N_GL = tuple(sorted({min(4 * n, N_CAP) for n in _v4.REQUIRED_N_GL}))


def adaptive_n_gl_v4x4(w_abs):
    return min(4 * _v4.adaptive_n_gl_v4(w_abs), N_CAP)


def apply():
    """Activate the v4x4 schedule for this process. Idempotent."""
    _axisym._adaptive_n_gl = adaptive_n_gl_v4x4


def restore():
    _axisym._adaptive_n_gl = _v4._ORIGINAL_ADAPTIVE_N_GL


def is_active():
    return _axisym._adaptive_n_gl is adaptive_n_gl_v4x4
