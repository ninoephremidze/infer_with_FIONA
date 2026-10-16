"""Gauss-Legendre node-count schedules n(w) for FIONA's axisymmetric solver.

Each schedule module exposes REQUIRED_N_GL (the node counts it can request), apply(), restore(),
is_active() and seed_gl_cache(). apply() replaces fiona.axisym._adaptive_n_gl for the current
process.

  cis         converged schedule for the CIS lens: n = 250 * 2^k, the smallest count for which F
              changes by less than 1e-3 at that and every larger count, maximized over 66 lenses
              spanning the prior (scripts/2_convergence/derive_quadrature_schedule.py)
  cis_x2      twice the cis node count at every w
  steep_core  four times the cis node count (capped at 1,024,000), for the ULA soliton and the
              point lens, whose cores are much steeper than any CIS lens in the derivation
  legacy_v3   an earlier, unconverged schedule, kept only for comparison in the derivation

Names used during development (v4, v4x2, v4x4, v3) are accepted as aliases.
"""
import importlib

SCHEDULES = {'cis': 'schedule_cis', 'cis_x2': 'schedule_cis_x2',
             'steep_core': 'schedule_steep_core', 'legacy_v3': 'schedule_legacy_v3'}
ALIASES = {'v4': 'cis', 'v4x2': 'cis_x2', 'v4x4': 'steep_core', 'v3': 'legacy_v3'}
CHOICES = sorted(SCHEDULES) + sorted(ALIASES)


def canonical_name(name):
    """Map a schedule name or alias to its canonical name."""
    name = ALIASES.get(name, name)
    if name not in SCHEDULES:
        raise ValueError(f'unknown quadrature schedule {name!r}; choose from {CHOICES}')
    return name


def get_schedule(name):
    """Return the schedule module for a name or alias (see the module docstring)."""
    return importlib.import_module(f'.{SCHEDULES[canonical_name(name)]}', __name__)
