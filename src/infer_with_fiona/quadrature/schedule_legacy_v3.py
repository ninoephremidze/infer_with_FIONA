"""Legacy quadrature schedule (called v3 during development), kept only so that
scripts/2_convergence/derive_quadrature_schedule.py can compare the converged schedule
with it. It was derived by pairwise self-convergence at 10 lenses and misses the 1e-3
target at most prior frequencies. Do not use it for production runs.

Original notes:

Does NOT modify any file in the vendored FIONA package. `fiona.axisym._adaptive_n_gl` is a plain
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

Usage: call `apply()` once, early, before any infer_with_fiona.fisher/fiona CIS computation in a
given process. `restore()` undoes it. Seed the node cache first (see seed_gl_cache) or the
first call at each new n pays numpy's O(n^3) generation.
"""
from ..fiona import axisym as _axisym
from .gl_cache import seed_gl_cache  # noqa: F401 -- re-exported for callers

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


def apply():
    """Activate the v3 schedule for this process. Idempotent."""
    _axisym._adaptive_n_gl = adaptive_n_gl_v3


def restore():
    """Restore FIONA's own original schedule for this process."""
    _axisym._adaptive_n_gl = _ORIGINAL_ADAPTIVE_N_GL


def is_active():
    return _axisym._adaptive_n_gl is adaptive_n_gl_v3
