#!/usr/bin/env python
"""Re-derive FIONA's GL node schedule n_gl(w) for the CIS lens from scratch, by convergence
against a high-n reference over a lens grid spanning the whole prior. Also measures where
bGO is accurate, using the same converged references.

WHAT n_gl CONTROLS (fiona/axisym.py, _lens_batch_core_compute)
---------------------------------------------------------------
The radial integral runs over [0, X] with X = max(1, sqrt(n_gl / 2w)), so the number of
nodes per local chirp wavelength is fixed (~4 pi at the edge) and n_gl really sets the
DOMAIN RADIUS: doubling n_gl enlarges X by sqrt(2) at constant sampling density.
The potential is tapered by 1/2[1 - tanh(x - 0.75X)] and the integrand by
1/2[1 - tanh((x - X)/(0.02X))].

FIRST-PRINCIPLES MODEL (tested here, not assumed)
------------------------------------------------
  edge window:      n >= 64 ln(1/eps)                      (~440 at eps=1e-3), any w
  potential taper:  X >= (4/3)(1 + y + c/w),  c = (2/pi) ln(1/eps)
                    => n >= (32/9) w (1 + y + c/w)^2        (~70/w at low w, ~3.6 w (1+y)^2 high)
  plus a candidate high-w effect: the taper region is resolved at only ~3.5-5.4 points per
  wavelength, independent of n, which could make high-w / large-y convergence slow.

  REVISION after the smoke test (job 48119292): the tanh potential taper is not compact --
  its exponential tail perturbs the potential AT the image, shifting the image phase by
  ~ w psi(x_img) exp(-2(R - x_img)). This is a first-order error, not oscillation-suppressed:
  taper tail:       R = 0.75 X >= x_img + (1/2) ln(w x_img / eps),   x_img <= 1 + y
                    => n >= 2 w X^2  (linear in w, log-corrected; ~57,000 at w=100, y=5, 1e-3)
  n_model() below takes the max of all three conditions.

CONVERGENCE TEST
----------------
  lenses : x_c x y grid spanning the prior (one batched FIONA call per n)
  w      : ~70 values -- log-spaced on [2e-4, 10), one jittered point per 3-wide bin on
           [10, 100), the production prior edges (M_Lz = 0.1 and 1 at 20 Hz) and w = 100
  n_gl   : 250 * 2^k up to N_REF (default 1024000), FIXED per call (adaptive_n_gl=False)
  ref    : F at N_REF, certified against N_REF/2
  error  : |F_n - F_ref| (absolute; F is O(1), and absolute error is what enters h*F)
  n_req  : smallest n such that the error is <= eps at n AND at every larger tested n
           (avoids the false passes of pairwise self-convergence)

bGO: |F_bGO - F_ref| on the same grid -> per lens, the lowest w above which bGO stays <= eps.

Outputs: gl_schedule_v4_raw.npz (all F values) and gl_schedule_v4_summary.json.
Re-run the analysis without recomputing: --analyze-only.
"""
import argparse
import json
import os
import time

import finufft  # noqa: F401 -- must precede any FIONA import
from infer_with_fiona import paths, quadrature  # noqa: E402 -- after finufft
import numpy as np

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--n-ref', type=int, default=1024000)
ap.add_argument('--n-workers', type=int, default=112)
ap.add_argument('--quick', action='store_true',
                help='Smoke test: 8 w values, 2 x_c x 3 y, n up to 32000.')
ap.add_argument('--analyze-only', action='store_true')
ap.add_argument('--out', default='quadrature_schedule',
                help='Output name stem, written to outputs/checks/. (default: quadrature_schedule)')
ap.add_argument('--w-range', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                help='Keep only w values in [LO, HI] (e.g. a low-w rerun with a larger --n-ref).')
ap.add_argument('--w-high-ext', action='store_true',
                help='Add w in (100, 254] (one jittered point per 3-wide bin, plus w at '
                     'M_Lz=2000, 1024 Hz): where production currently switches to bGO.')
ap.add_argument('--check-schedule', default=None, metavar='NAME',
                help='Instead of the convergence scan: evaluate F with the adaptive schedule '
                     'NAME (see infer_with_fiona.quadrature) on the w/lens grid of --ref-raw and report its '
                     'error against that file\'s certified reference.')
ap.add_argument('--ref-raw', default=None, help='Raw .npz with the reference (for --check-schedule).')
ap.add_argument('--w-seed', type=int, default=20260924,
                help='Seed for the jittered high-w points. Any other value also shifts the log '
                     'grid by half a step, giving an independent w set for checking a schedule '
                     'built from the default one.')
args = ap.parse_args()

ROOT = str(paths.REPO_ROOT)
CHECKS = str(paths.CHECKS_DIR); os.makedirs(CHECKS, exist_ok=True)   # outputs/checks
GL_DIR = str(paths.GL_RULES_DIR)
W_HIGH = 100.0
EPS_LIST = (1e-2, 1e-3, 1e-4)
SUFFIX = '_quick' if args.quick else ''
RAW = os.path.join(CHECKS, args.out + '_raw' + SUFFIX + '.npz')
SUMMARY = os.path.join(CHECKS, args.out + '_summary' + SUFFIX + '.json')

# ─── grids ─────────────────────────────────────────────────────────────────────────────


def make_w_values(quick):
    from infer_with_fiona.fisher.utils import G_over_c3_Msun
    w_of = lambda M, f: 8.0 * np.pi * G_over_c3_Msun * M * f
    if quick:
        return np.array([w_of(0.1, 20.0), 3e-3, 0.05, 0.5, 3.0, 20.0, 60.0, 100.0])
    w_log = np.logspace(np.log10(2e-4), np.log10(10.0), 37, endpoint=False)
    if args.w_seed != 20260924:
        w_log = w_log * (10.0 / 2e-4) ** (0.5 / 37)          # half a log step
    rng = np.random.default_rng(args.w_seed)
    edges = np.arange(10.0, 100.0 + 1e-9, 3.0)              # 30 bins of width 3
    w_lin = edges[:-1] + rng.uniform(0.0, 3.0, len(edges) - 1)
    w_edges = [w_of(0.1, 20.0), w_of(1.0, 20.0), 100.0]     # prior edges + W_HIGH
    parts = [w_log, w_lin, w_edges]
    if args.w_high_ext:
        e2 = np.arange(100.0, 253.0 + 1e-9, 3.0)
        parts += [e2[:-1] + rng.uniform(0.0, 3.0, len(e2) - 1), [w_of(2000.0, 1024.0)]]
    W = np.unique(np.concatenate(parts))
    if args.w_range is not None:
        W = W[(W >= args.w_range[0]) & (W <= args.w_range[1])]
    return W


if args.quick:
    XC = np.array([0.01, 0.5])
    Y = np.array([0.3, 1.0, 5.0])
    N_LIST = [250 * 2 ** k for k in range(8)]               # 250 .. 32000
else:
    XC = np.array([0.001, 0.01, 0.05, 0.2, 0.5, 1.0])
    Y = np.array([0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    N_LIST = [250 * 2 ** k for k in range(20) if 250 * 2 ** k <= args.n_ref]
N_REF = N_LIST[-1]


def n_model(w, y, eps):
    """First-principles prediction (see module docstring)."""
    c = (2.0 / np.pi) * np.log(1.0 / eps)
    x_img = 1.0 + y
    n_window = 64.0 * np.log(1.0 / eps)
    n_smooth = (32.0 / 9.0) * w * (x_img + c / w) ** 2
    X_tail = (4.0 / 3.0) * (x_img + 0.5 * np.log(np.maximum(w * x_img / eps, 1.0)))
    n_tail = 2.0 * w * X_tail ** 2
    return np.maximum(n_window, np.maximum(n_smooth, n_tail))


# ─── check an adaptive schedule against a saved, certified reference ───────────────────
if args.check_schedule is not None:
    import importlib
    import infer_with_fiona.fiona.utils as _fiona_utils
    from infer_with_fiona.fiona.axisym import FresnelNUFHTLensBatched
    from infer_with_fiona.fiona.lenses import CIS
    import infer_with_fiona.fiona.axisym as _axisym
    ref = np.load(args.ref_raw)
    W, XC, Y, N_LIST = ref['W'], ref['XC'], ref['Y'], list(ref['N_LIST'])
    F_ref, cert = ref['F_all'][-1], np.abs(ref['F_all'][-2] - ref['F_all'][-1])
    mod = quadrature.get_schedule(args.check_schedule)
    mod.seed_gl_cache(GL_DIR, n_values=tuple(mod.REQUIRED_N_GL))
    mod.apply()
    _fiona_utils._FIONA_GL2D_DIR = GL_DIR
    solver = FresnelNUFHTLensBatched([CIS(psi0=1.0, xc=float(xc)) for xc in XC],
                                     gl_dir=GL_DIR, n_workers=args.n_workers, print_timing=False)
    t0 = time.time()
    F_s = solver(W, Y)
    e = np.abs(F_s - F_ref)
    n_used = np.array([_axisym._adaptive_n_gl(w) for w in W])
    print(f'schedule {args.check_schedule}: evaluated {len(W)} w x {len(XC)*len(Y)} lenses '
          f'in {time.time()-t0:.0f}s')
    print(f'{"w":>9s} {"n_used":>8s} {"max err":>10s} {"ref cert":>10s}')
    for iw, w in enumerate(W):
        print(f'{w:9.4g} {n_used[iw]:8d} {e[iw].max():10.2e} {cert[iw].max():10.2e}')
    ok_ref = cert <= 1e-3
    print(f'\nmax error where reference certified to 1e-3: {e[ok_ref].max():.2e}; '
          f'points above 1e-3: {int((e[ok_ref] > 1e-3).sum())} / {int(ok_ref.sum())}')
    json.dump(dict(W=W.tolist(), n_used=n_used.tolist(), max_err=e.max(axis=(1, 2)).tolist(),
                   cert_max=cert.max(axis=(1, 2)).tolist()),
              open(os.path.join(CHECKS, f'{args.out}_check_{quadrature.canonical_name(args.check_schedule)}.json'), 'w'),
              indent=1)
    raise SystemExit(0)

# ─── compute ───────────────────────────────────────────────────────────────────────────
if not args.analyze_only:
    import infer_with_fiona.fiona.utils as _fiona_utils
    from infer_with_fiona.fiona.axisym import FresnelNUFHTLensBatched
    from infer_with_fiona.fiona.lenses import CIS
    from infer_with_fiona.fisher.generic_go import F_bGO
    from infer_with_fiona.quadrature import gl_cache, schedule_legacy_v3
    W = make_w_values(args.quick)
    print(f'{len(W)} w values in [{W.min():.3g}, {W.max():.3g}]; {len(XC)} x_c x {len(Y)} y '
          f'= {len(XC)*len(Y)} lenses; n_gl = {N_LIST}', flush=True)

    t0 = time.time()
    gl_cache.seed_gl_cache(GL_DIR, n_values=tuple(N_LIST))
    print(f'GL cache seeded ({time.time()-t0:.0f}s)', flush=True)
    _fiona_utils._FIONA_GL2D_DIR = GL_DIR

    lenses = [CIS(psi0=1.0, xc=float(xc)) for xc in XC]
    F_all = np.empty((len(N_LIST), len(W), len(XC), len(Y)), dtype=np.complex128)
    t_n = []
    for k, n in enumerate(N_LIST):
        t0 = time.time()
        solver = FresnelNUFHTLensBatched(lenses, gl_nodes_per_dim=n, adaptive_n_gl=False,
                                         gl_dir=GL_DIR, n_workers=args.n_workers,
                                         print_timing=False)
        F_all[k] = solver(W, Y)                                # (n_w, n_xc, n_y)
        t_n.append(time.time() - t0)
        print(f'  n_gl={n:7d}  {t_n[-1]:7.1f}s', flush=True)

    F_bgo = np.full((len(W), len(XC), len(Y)), np.nan + 0j)
    for i, xc in enumerate(XC):
        for j, y in enumerate(Y):
            try:
                F_bgo[:, i, j] = F_bGO(CIS(psi0=1.0, xc=max(float(xc), 1e-6)), float(y), W)
            except Exception as e:                            # noqa: BLE001
                print(f'  bGO failed at x_c={xc}, y={y}: {e!r}', flush=True)
    from infer_with_fiona import table as FWT
    F_go_all = np.empty((len(W), len(XC), len(Y)), dtype=complex)   # all-image plain GO
    for i, xc in enumerate(XC):
        for j, y in enumerate(Y):
            F_go_all[:, i, j] = FWT.F_go(W, *FWT.cis_image_set(xc, y))
    n_v3 = np.array([schedule_legacy_v3.adaptive_n_gl_v3(w) for w in W])
    np.savez(RAW, W=W, XC=XC, Y=Y, N_LIST=np.array(N_LIST), F_all=F_all, F_bgo=F_bgo,
             F_go=F_go_all, n_v3=n_v3, t_n=np.array(t_n))
    print('Saved', RAW, flush=True)

# ─── analysis ──────────────────────────────────────────────────────────────────────────
d = np.load(RAW)
W, XC, Y, N_LIST = d['W'], d['XC'], d['Y'], list(d['N_LIST'])
F_all, F_bgo, n_v3 = d['F_all'], d['F_bgo'], d['n_v3']
N_REF = N_LIST[-1]
F_ref = F_all[-1]
err = np.abs(F_all[:-1] - F_ref[None])                          # (n_n-1, n_w, n_xc, n_y)
cert = err[-1]                                                   # N_REF/2 vs N_REF
n_arr = np.array(N_LIST[:-1], dtype=float)

# "stays converged": suffix-max of the error over larger n
err_suffix = np.maximum.accumulate(err[::-1], axis=0)[::-1]

summary = dict(W=W.tolist(), XC=XC.tolist(), Y=Y.tolist(), N_LIST=[int(n) for n in N_LIST],
               t_n=d['t_n'].tolist(), eps={})
YY = np.broadcast_to(Y[None, None, :], cert.shape)

print(f'\nReference n={N_REF}; certification |F({N_REF//2}) - F({N_REF})|: '
      f'max={cert.max():.2e}, median={np.median(cert):.2e}')
for eps in EPS_LIST:
    bad = cert > eps
    print(f'  eps={eps:g}: reference NOT certified at {bad.sum()} / {bad.size} (w, lens) points'
          + (f'; worst w={W[np.nonzero(bad.any(axis=(1,2)))[0]].round(3).tolist()}' if bad.any() else ''))

for eps in EPS_LIST:
    ok = err_suffix <= eps
    first = np.argmax(ok, axis=0)                                # first index where ok
    n_req = np.where(ok.any(axis=0), n_arr[first], np.inf)       # (n_w, n_xc, n_y)
    n_req[cert > eps] = np.inf                                    # can't certify -> unknown
    worst = n_req.reshape(len(W), -1).max(axis=1)
    arg = n_req.reshape(len(W), -1).argmax(axis=1)
    ix, iy = np.unravel_index(arg, (len(XC), len(Y)))
    model_max = n_model(W, Y.max(), eps)
    # v3's actual error
    k_v3 = np.array([N_LIST.index(int(n)) if int(n) in N_LIST[:-1] else -1 for n in n_v3])
    err_v3 = np.array([err[k, iw].max() if k >= 0 else np.nan for iw, k in enumerate(k_v3)])

    print(f'\n=== eps = {eps:g}: required n_gl per w (max over {len(XC)*len(Y)} lenses) ===')
    print(f'{"w":>9s} {"n_req":>8s} {"worst (x_c, y)":>16s} {"model(y=5)":>10s} {"v3 n":>7s} {"v3 err":>9s}')
    for iw, w in enumerate(W):
        nr = 'unconv' if not np.isfinite(worst[iw]) else f'{int(worst[iw]):d}'
        print(f'{w:9.4g} {nr:>8s}   ({XC[ix[iw]]:.3g}, {Y[iy[iw]]:.3g}) {model_max[iw]:10.0f} '
              f'{int(n_v3[iw]):7d} {err_v3[iw]:9.2e}')
    summary['eps'][f'{eps:g}'] = dict(
        n_req=np.where(np.isfinite(n_req), n_req, -1).tolist(),
        n_req_worst=np.where(np.isfinite(worst), worst, -1).tolist(),
        model_ymax=model_max.tolist(), v3_err=err_v3.tolist())

# ─── closed forms (bGO as in production; all-image plain GO) vs converged FIONA ────────
closed = {'bgo': F_bgo}
if 'F_go' in d.files:
    closed['go_all'] = d['F_go']
for name, F_cf in closed.items():
    err_cf = np.abs(F_cf - F_ref)                                # (n_w, n_xc, n_y)
    print(f'\n=== {name} vs converged FIONA: lowest w above which the error stays <= eps ===')
    cf_summary = {}
    for eps in EPS_LIST:
        ok = (err_cf <= eps) | (cert > eps)   # uncertified reference: don't count against it
        ok_suffix = np.minimum.accumulate(ok[::-1], axis=0)[::-1]   # ok at this w and above
        w_min = np.full((len(XC), len(Y)), np.nan)
        for i in range(len(XC)):
            for j in range(len(Y)):
                idx = np.nonzero(ok_suffix[:, i, j])[0]
                if len(idx):
                    w_min[i, j] = W[idx[0]]
        cf_summary[f'{eps:g}'] = w_min.tolist()
        print(f'\n  eps={eps:g}  (rows x_c, columns y; "-" = not reached by w={W.max():.4g})')
        print('  ' + ' ' * 7 + ''.join(f'{y:>7g}' for y in Y))
        for i, xc in enumerate(XC):
            print(f'  {xc:7g}' + ''.join(f'{v:7.3g}' if np.isfinite(v) else f'{"-":>7s}'
                                         for v in w_min[i]))
    summary[f'{name}_w_min'] = cf_summary
    iw_hi = np.argmin(np.abs(W - W_HIGH))
    summary[f'{name}_err_at_W_HIGH'] = err_cf[iw_hi].tolist()
    print(f'  {name} error at w={W[iw_hi]:.4g}: max {err_cf[iw_hi].max():.2e}, '
          f'median {np.median(err_cf[iw_hi]):.2e}')
json.dump(summary, open(SUMMARY, 'w'), indent=1)
print('\nSaved', SUMMARY)
