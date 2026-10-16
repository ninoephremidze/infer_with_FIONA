"""GL schedule v4x2: twice the v4 node count at every w (capped at 1,024,000, the largest
cached rule). For lenses whose core is much steeper than any CIS in the v4 derivation, such
as the ULA soliton at r_s ~ 0.01 (central convergence ~6e4), where v4 is off by up to ~4e-3
(derive_bgo_switch_ula_pbh.py). Same API as quadrature.schedule_cis."""
from ..fiona import axisym as _axisym

from . import schedule_cis as _v4
from .gl_cache import seed_gl_cache  # noqa: F401 -- re-exported for callers

N_CAP = 1024000
REQUIRED_N_GL = tuple(sorted({min(2 * n, N_CAP) for n in _v4.REQUIRED_N_GL}))


def adaptive_n_gl_v4x2(w_abs):
    return min(2 * _v4.adaptive_n_gl_v4(w_abs), N_CAP)


def apply():
    """Activate the v4x2 schedule for this process. Idempotent."""
    _axisym._adaptive_n_gl = adaptive_n_gl_v4x2


def restore():
    _axisym._adaptive_n_gl = _v4._ORIGINAL_ADAPTIVE_N_GL


def is_active():
    return _axisym._adaptive_n_gl is adaptive_n_gl_v4x2
