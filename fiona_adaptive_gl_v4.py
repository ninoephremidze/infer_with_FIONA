"""Overlay patch for FIONA's adaptive Gauss-Legendre node-count schedule (CIS lens), v4.

Does NOT modify any file under 0-fiona/. Same mechanism and API as fiona_adaptive_gl_v3:
reassigns the module attribute fiona.axisym._adaptive_n_gl at runtime.

WHY v4 SUPERSEDES v3
--------------------
v3 was derived by pairwise self-convergence at 10 (x_c, y) points. Against a certified
reference (derive_gl_schedule_v4.py, job 48174280) it misses |F - F_ref| <= 1e-3 at 49 of 69
prior w values, worst 0.60 at w = 100; it is also O(1) wrong above w = 100, where FIONA is
now needed for small-x_c / small-y lenses (bGO is not accurate there).

WHAT n_gl CONTROLS
------------------
FIONA's radial domain is X = max(1, sqrt(n_gl / 2w)); the sampling density per wavelength is
fixed, so n_gl sets the domain RADIUS. The measured requirement is X ~ 20-30 at w >~ 1
(n ~ 1250 w) and n ~ 1/w below the ~500 floor (see derive_gl_schedule_v4.py's docstring for
the first-principles model, which gets these shapes right but is ~2x low).

DERIVATION
----------
derive_gl_schedule_v4.py --n-ref 1024000 --w-high-ext (job 48174280):
  66 lenses (x_c in {0.001,...,1} x y in {0.1,...,5}), 122 w in [2e-4, 254],
  n_gl = 250 * 2^k up to 1,024,000; reference certified vs 512,000 to 1e-3 at all but 6 of
  8,052 points (all at w = 2e-4, below the prior's w_min = 2.48e-4).
  n_req = smallest n with |F_n - F_ref| <= 1e-3 at n and every larger n, max over lenses.
  Tier value = max of n_req over each w sample and its two neighbours (the error oscillates
  with w between samples); tier edges at geometric midpoints between samples. Node counts
  are powers of two times 250 only, because GL generation is O(n^2) (1,024,000 took 7 h 40
  min) and all of these are already cached.

MARGIN (added after the independent check, job 48182125)
--------------------------------------------------------
On an independent w grid (log grid shifted by half a step, new jitter), the measured tiers
exceeded 1e-3 at 15 of 8,052 points (worst 1.3e-3), all just below a tier's upper edge on the
rising branch (w/edge = 0.91-0.99) or at the lower edge of a falling-branch tier. So tier
edges are moved by a factor in w: rising branch (n increasing with w) ends each tier 15%
earlier; falling branch (low w) extends each tier 25% further. Because this margin was set
from the check grid, that check is no longer fully independent; re-check with a third grid.
"""
import fiona.axisym as _axisym

from fiona_adaptive_gl_v3 import seed_gl_cache  # noqa: F401 -- same cache writer

_ORIGINAL_ADAPTIVE_N_GL = _axisym._adaptive_n_gl

# Measured (lower w edge, n_gl) before margin; each tier holds until the next edge.
TIERS_MEASURED = (
    (0.0,       1024000),   # below the prior's w_min = 2.48e-4
    (2.576e-4,   512000),
    (5.566e-4,   256000),
    (7.456e-4,   128000),
    (2.267e-3,    64000),
    (4.310e-3,    32000),
    (1.036e-2,    16000),
    (1.860e-2,     8000),
    (3.338e-2,     4000),
    (8.026e-2,     2000),
    (0.1440,       1000),
    (0.3463,        500),
    (0.6216,       1000),
    (1.495,        2000),
    (2.682,        4000),
    (4.814,        8000),
    (8.725,       16000),
    (19.21,       32000),
    (39.63,       64000),
    (72.04,      128000),
    (137.3,      256000),
    (250.5,      512000),
)

MARGIN_RISING = 1.15    # edges on the rising branch are divided by this
MARGIN_FALLING = 1.25   # edges on the falling branch are multiplied by this


def _with_margin(tiers):
    out = [tiers[0]]
    for (lo_prev, n_prev), (lo, n) in zip(tiers[:-1], tiers[1:]):
        out.append((lo / MARGIN_RISING if n > n_prev else lo * MARGIN_FALLING, n))
    return tuple(out)


TIERS = _with_margin(TIERS_MEASURED)
_EDGES = tuple(t[0] for t in TIERS)

REQUIRED_N_GL = tuple(sorted({n for _, n in TIERS}))


def adaptive_n_gl_v4(w_abs):
    import bisect
    return TIERS[bisect.bisect_right(_EDGES, w_abs) - 1][1]


def apply():
    """Activate the v4 schedule for this process. Idempotent."""
    _axisym._adaptive_n_gl = adaptive_n_gl_v4


def restore():
    """Restore FIONA's own original schedule for this process."""
    _axisym._adaptive_n_gl = _ORIGINAL_ADAPTIVE_N_GL


def is_active():
    return _axisym._adaptive_n_gl is adaptive_n_gl_v4
