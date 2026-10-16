#!/usr/bin/env python3
"""
Faithful reproduction of Cheung, Ng, Zumalacarregui & Berti (2024)'s ("glworia") CIS
parameter-estimation pipeline (https://github.com/mhycheung/glworia), using FIONA (this
project's own wave-optics solver) instead of their own time-domain-interpolation method
for the lensing amplification factor F(w,y,x_c), but otherwise matching their actual
released code as closely as possible -- NOT just their paper's prose description.

Built by directly reading scripts/PE_full.py, glworia/pe/waveform.py, glworia/pe/prior.py,
and their exact benchmark config PE_submission_settings/CIS/MLz_100.0_dL_500.0_y_*.json
from the cloned repo. Confirmed matches to our OLD pipeline (dynesty_LIGO_CIS_full_params.py):
GW fiducial values (masses, spins, angles, distance, sky location, time), the w=8pi(G/c^3)
M_Lz f convention, and applying F(w,y,x_c) identically to both h_+ and h_x.

Confirmed DIFFERENCES from our old pipeline, now fixed here:
  1. reference_frequency = 50.0 Hz (we assumed 20.0 -- WRONG, now fixed).
  2. Noise PSD: bilby's own default "aLIGO_ZERO_DET_high_P_psd.txt" for BOTH H1 and L1
     (not the O4-specific curve we used) -- note this contradicts the PAPER's own prose
     ("O4 design sensitivity"), but we follow the actual released code per instructions.
  3. Noise generation MECHANISM: actual bilby Interferometer objects + bilby's own RNG
     (bilby.core.utils.random.seed), not a hand-rolled closed-form draw -- this is the
     only way to get bit-identical noise realizations (matching their seed doesn't help
     otherwise, since a from-scratch reimplementation would call a RNG in a different
     sequence regardless of seed).
  4. Luminosity-distance prior: plain LINEAR-uniform over [1, 5000] Mpc (we used
     uniform-in-comoving-volume over a much narrower range).
  5. M_Lz prior: plain LINEAR-uniform over the FULL [0.1, 2000] Msun range (we had
     deliberately restricted this to a narrow +/-0.5-dex log-uniform window after finding
     the full range numerically catastrophic for the old FIONA table; the (x_c, y, w) table
     that replaced it has no M_Lz interpolation -- see "F(w, y, x_c)" below).
  6. (y, x_c) prior: a joint prior that uniformly samples the rectangle EXCEPT for a band
     around the CIS critical curve y_crit(x_c) (avoiding the strong/weak-lensing caustic
     divergence), rather than two independent uniforms. The mask uses glworia's OWN pickled
     y_crit interpolator (interpolation/crit_funcs/CIS_crit_funcs_0.000_1.000.pkl, exactly
     what their PE_full.py loads), and no mask box, since PE_full.py never passes the
     config's mask_boxes to the prior (2026-09-27). The analytic y_crit_cis below agrees
     with their interpolator for 0 < x_c < 0.49 but, unlike it, would also mask y < 0.15
     for x_c >= ~0.49, where CIS has no caustic and glworia's mask is off.
  7. Sky location + time: they do NOT sample (ra, dec, t_gps) directly. They sample
     (zenith, azimuth) -- a detector-frame sky parametrization -- plus "H1_time" (arrival
     time AT H1), and use time_marginalization=True in the likelihood (arrival time is
     analytically marginalized, not explicitly tracked as an ordinary sampled dimension).

F(w, y, x_c) (rewritten 2026-09-27): F depends on M_Lz and f only through
w = 8 pi G M_Lz f / c^3, so it is tabulated on an (x_c, y, w) grid (amplification /
table): 201 x 201 x ~8,450 w-nodes up to w = 254 at --grid-res high (~5 GB, cached on
disk and loaded into shared memory), computed with the v4 Gauss-Legendre schedule. The table
stores F minus the geometric-optics image sum (added back exactly at each query), which
makes bilinear (x_c, y) interpolation accurate at high w, and F blends smoothly into
beyond-geometric optics (bGO) above a lens-dependent w_s(x_c, y) where bGO is accurate to
1e-3. Validated against direct converged FIONA (validate_table_cis.py, job 48205398;
analyze_w_table_validation.ipynb): worst 1.02%, median 0.032% over 130 test combinations
outside the prior's caustic mask, masses 0.1-2000. Inside the masked band the table is not
valid, which is one more reason the prior excludes it. This replaces the earlier
(logM, x_c, y, f) table (298 GiB at grid-res high, out-of-memory in production), whose
log M_Lz interpolation was off by up to ~190%, and the fixed FIONA -> bGO switch at w = 100.

IMPORTANT: `finufft` must be imported before any infer_with_fiona.fisher/fiona import (else FIONA's
NUFHT extension fails to find its own shared library at import time).
"""
import finufft  # noqa: F401 -- see module docstring

import os
import sys
import time
import argparse
from multiprocessing import shared_memory

import numpy as np
from scipy.constants import c as C_SI, G as G_SI_SCIPY

import bilby
from bilby.core.prior.joint import BaseJointPriorDist, JointPrior, JointPriorDistError
from bilby.core.utils import random as bilby_random
from bilby.gw.source import _base_lal_cbc_fd_waveform


from infer_with_fiona import amplification  # cached F(w, x_c, y) table + bGO switch
from infer_with_fiona import paths, quadrature

ROOT = str(paths.REPO_ROOT)
GL_DIR = str(paths.GL_RULES_DIR)

ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--m-lz-fid', type=float, default=100.0,
                 help='Fiducial (injected) M_Lz [Msun]. (default: 100.0)')
ap.add_argument('--xc-fid', type=float, default=0.02,
                 help='Fiducial (injected) CIS core radius x_c. Cheung+2024 use 0.02 for '
                      'their strong-lensing benchmark, 0.2 for weak-lensing. (default: 0.02)')
ap.add_argument('--y-fid', type=float, default=0.3,
                 help='Fiducial (injected) source position y. Cheung+2024 use 0.3 for '
                      'strong-lensing, 1/2/3 for weak-lensing. (default: 0.3)')
ap.add_argument('--grid-res', choices=['tiny', 'low', 'medium', 'high'], default='high',
                 help="(x_c, y) density of the F table: a square N x N grid. 'high'=201 (the "
                      "validated and cached configuration; matches glworia's N_grid_strong), "
                      "'medium'=101, 'low'=61, 'tiny'=21 (smoke tests only; not validated, and "
                      "each builds its own cached table on first use). (default: high)")
ap.add_argument('--xc-prior-min', type=float, default=0.0,
                 help='x_c prior lower bound -- Cheung+2024 use 0.0. Narrow this (with '
                      '--xc-prior-max) for a fast tight-prior test around a known fiducial '
                      'x_c, at the cost of no longer matching their prior exactly. (default: 0.0)')
ap.add_argument('--xc-prior-max', type=float, default=1.0,
                 help='x_c prior upper bound -- Cheung+2024 use 1.0. (default: 1.0)')
ap.add_argument('--y-prior-min', type=float, default=0.1,
                 help='y prior lower bound -- Cheung+2024 use 0.1. Narrow this (with '
                      '--y-prior-max) for a fast tight-prior test around a known fiducial y. '
                      '(default: 0.1)')
ap.add_argument('--y-prior-max', type=float, default=5.0,
                 help='y prior upper bound -- Cheung+2024 use 5.0. (default: 5.0)')
ap.add_argument('--dl-prior-min', type=float, default=1.0,
                 help='Luminosity-distance prior lower bound [Mpc] -- Cheung+2024 use 1.0. '
                      '(default: 1.0)')
ap.add_argument('--dl-prior-max', type=float, default=5000.0,
                 help='Luminosity-distance prior upper bound [Mpc] -- Cheung+2024 use 5000.0. '
                      '(default: 5000.0)')
ap.add_argument('--m-lz-prior-min', type=float, default=0.1,
                 help='M_Lz prior lower bound [Msun] -- Cheung+2024 use 0.1 (their full '
                      'linear-uniform range). Narrow this (together with --m-lz-prior-max) '
                      'for a fast test around a known fiducial mass, at the cost of no longer '
                      'matching their prior exactly. The F table covers w from 2e-4 (or lower '
                      'if this bound requires it) to w at --m-lz-prior-max and 1024 Hz; '
                      'changing --m-lz-prior-max therefore builds a new cached table. '
                      '(default: 0.1)')
ap.add_argument('--m-lz-prior-max', type=float, default=2000.0,
                 help='M_Lz prior upper bound [Msun] -- Cheung+2024 use 2000.0. '
                      '(default: 2000.0)')
ap.add_argument('--m-lz-prior-type', choices=['uniform', 'log-uniform'], default='uniform',
                 help="'uniform' (default) matches Cheung+2024's own CIS convention "
                      '(linear-uniform over their full [0.1,2000] range). \'log-uniform\' '
                      'samples the same [--m-lz-prior-min, --m-lz-prior-max] range but flat '
                      'per decade instead -- a deliberate departure from C24 fidelity, '
                      'matching this project\'s own ULA/PBH convention (see '
                      'PE_ULA_glworia_reproduction.py). The F table is the same either way.')
ap.add_argument('--narrow-priors', action='store_true',
                 help='Fast variant: narrow priors on every sampled parameter except y and x_c '
                      '(and psi, phase, which the data leave broad). Boxes cover the 0.5-99.5%% '
                      'range of earlier posteriors of this same injection and noise realisation '
                      '(glworia prod_7 weak y=2,3; FIONA [50,150] runs), so they are informed by '
                      'the same data: use for speed and lensing checks, not as an independent '
                      'inference. M_Lz becomes [50, 150], which is prior-dominated for the weak '
                      'cases (their M_Lz posterior fills even the full [0.1, 2000]). Overrides '
                      '--m-lz-prior-min/max and --dl-prior-min/max. See NARROW_PRIORS.')
ap.add_argument('--n-logm', type=int, default=None,
                 help='DEPRECATED, ignored: the F table no longer has an M_Lz axis (F depends '
                      'on M_Lz and f only through w, so it is tabulated in (x_c, y, w)).')
ap.add_argument('--gl-schedule', default='cis', choices=quadrature.CHOICES,
                 help='Gauss-Legendre node schedule (infer_with_fiona.quadrature). cis is converged '
                      'to |dF| <= 1e-3 against a certified reference. (default: cis)')
ap.add_argument('--table-cache', default=str(paths.TABLE_CACHE_DIR),
                 help='Directory of cached F tables (keyed by a hash of the full table '
                      'configuration). A matching table loads in seconds; otherwise it is built '
                      '(~46 min on 112 cores at grid-res high) and saved here. '
                      '(default: outputs/table_cache)')
ap.add_argument('--no-table-cache', action='store_true',
                 help='Always build the F table and do not save it.')
ap.add_argument('--bgo-switch', default=str(paths.BGO_SWITCH['CIS']),
                 help='derive_bgo_switch_cis.py output giving w_s(x_c, y), above which F blends '
                      'into bGO. (default: data/bgo_switch/cis.npz)')
ap.add_argument('--fourier-convention', choices=['lal', 'fiona'], default='lal',
                 help="Fourier convention of F. 'lal' (default): conj(F), the correct factor for "
                      "LAL/bilby waveforms (h(f) = Int h e^{-2 pi i f t}); later images arrive later. "
                      "'fiona': F as computed (e^{+i w T}), which places later images EARLIER when "
                      "multiplied into a LAL waveform -- glworia's convention, kept only for "
                      "comparisons with runs made before 2026-10-07 (check_fourier_convention.py).")
ap.add_argument('--time-reference', choices=['min_image', 'absolute'], default='min_image',
                 help="Time reference of F. 'min_image' (default, glworia's convention): arrival "
                      "times measured from the first-arriving image, so F carries no overall, "
                      "lens-dependent time shift. 'absolute': FIONA's raw F, whose Fermat phase shifts "
                      "the signal by 4 G M_Lz phi_min / c^3 -- this made H1_time degenerate with the "
                      "lens parameters (slow sampling, and the +-0.1 s H1_time prior cut off "
                      "|shift| > 100 ms; analyze_sampler_efficiency.ipynb).")
ap.add_argument('--bgo-switch-eps', type=float, default=1e-3,
                 help='Accuracy level of the switch table to use (1e-2, 3e-3 or 1e-3). '
                      '(default: 1e-3)')
ap.add_argument('--n-workers', type=int, default=112,
                 help='Worker count for the FIONA table-build pool AND the bilby/dynesty '
                      'sampler pool. (default: 112)')
ap.add_argument('--nlive', type=int, default=1000,
                 help='dynesty nlive, via bilby.run_sampler -- Cheung+2024 use 1000, '
                      'everything else (walks/bound/sample/dlogz) left at bilby\'s own '
                      'current defaults, matching their own config (which likewise only '
                      'specifies nlive). (default: 1000)')
ap.add_argument('--zero-noise', action='store_true',
                 help='Zero-noise injection instead of a real noise draw (misc.zero_noise '
                      'in their config; default there is false).')
ap.add_argument('--seed', type=int, default=1234,
                 help='RNG seed, via bilby.core.utils.random.seed -- Cheung+2024 use 1234. '
                      '(default: 1234)')
ap.add_argument('--out-label', default=None,
                 help='Output label/directory name (default: auto-generated from --xc-fid/'
                      '--y-fid/--m-lz-fid).')
ap.add_argument('--output-dir', default=str(paths.RUNS_DIR / 'CIS_glworia_repro'),
                 help='Directory for bilby outdir. (default: outputs/runs/CIS_glworia_repro)')
args = ap.parse_args()

# --narrow-priors boxes (see its help). Values chosen 2026-09-27 from the 0.5-99.5% posterior
# range, max over glworia prod_7 weak y=2,3 and the four FIONA [50,150] runs, with margin.
NARROW_PRIORS = dict(
    MLz=(50.0, 150.0), luminosity_distance=(200.0, 1600.0),
    chirp_mass=(25.5, 27.5), mass_ratio=(0.45, 1.0), a_1=(0.0, 0.3), a_2=(0.0, 0.3),
    theta_jn=(0.0, 1.6), H1_time_halfwidth=0.01, zenith=(2.30, 2.43), azimuth=(2.1, 3.8),
)
if args.narrow_priors:
    args.m_lz_prior_min, args.m_lz_prior_max = NARROW_PRIORS['MLz']
    args.dl_prior_min, args.dl_prior_max = NARROW_PRIORS['luminosity_distance']

N_WORKERS = args.n_workers
M_LZ_FID = args.m_lz_fid
XC_FID = args.xc_fid
Y_FID = args.y_fid

label = args.out_label or f'CIS_glworia_repro_mlz{M_LZ_FID:g}_xc{XC_FID:g}_y{Y_FID:g}'
outdir = os.path.join(ROOT, args.output_dir, label)
os.makedirs(outdir, exist_ok=True)

bilby.core.utils.setup_logger(outdir=outdir, label=label, log_level='info')
bilby_random.seed(args.seed)

# ═══ Exact constants used by glworia's own waveform.py (Mtow = 8 pi G / c^3 * Msun) ══════
# Deliberately using scipy.constants (their own source) rather than infer_with_fiona.fisher.utils's
# own rounded G_SI/c_SI/M_sun_kg, for exact numerical parity in the w = Mtow*MLz*f formula.
MSUN_GLWORIA = 1.9884099e+30
MTOW_GLWORIA = 8.0 * np.pi * G_SI_SCIPY / C_SI ** 3 * MSUN_GLWORIA

# ═══ Section 1: exact injection parameters, from PE_submission_settings/CIS/ configs ═════
injection_parameters = dict(
    mass_1=30.0, mass_2=30.0,
    a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0,
    theta_jn=1.0471975511965976, psi=1.5707963267948966, phase=1.5707963267948966,
    geocent_time=1126259642.413, ra=1.375, dec=-1.2108,
    luminosity_distance=500.0,
    y=Y_FID, MLz=M_LZ_FID, lp=XC_FID,
)

duration = 4.0
sampling_frequency = 2048.0
minimum_frequency = 20.0

# ═══ Section 2: the F(w, y, x_c) table (cached FIONA table + GO residual + bGO switch; see
# amplification). Loaded, or built and cached, once below, before bilby forks its sampler pool. ═
MLZ_MIN, MLZ_MAX = args.m_lz_prior_min, args.m_lz_prior_max  # Cheung+2024's own full
                                                              # prior is (0.1,2000) -- see
                                                              # --m-lz-prior-min/max to
                                                              # narrow it for faster,
                                                              # higher-M_Lz-resolution tests
XC_MIN, XC_MAX = args.xc_prior_min, args.xc_prior_max   # Cheung+2024's own prior: (0.0,1.0)
Y_MIN, Y_MAX = args.y_prior_min, args.y_prior_max       # Cheung+2024's own prior: (0.1,5.0)
if args.n_logm is not None:
    print('[F] --n-logm is deprecated and ignored (the F table has no M_Lz axis).', flush=True)
F_MAX = sampling_frequency / 2.0
N_GRID = {'tiny': 21, 'low': 61, 'medium': 101, 'high': 201}[args.grid_res]
# 'high'=201 matches glworia's own N_grid_strong (their FINER, strong-lensing-side
# resolution) -- used here EVERYWHERE, not just near the caustic. See "F(w,y,x_c) table
# resolution" in the module docstring for why (a two-grid weak/strong attempt cost ~401 GB).
LN10 = np.log(10.0)


def y_crit_cis(xc):
    '''Analytic CIS critical curve y_crit(x_c), derived from the radial-critical-curve
    condition f2 = 1 - dalpha/dx = 0 (i.e. xc/(s*(s+xc)) = 1, solved for s, then
    y_crit = x_crit - alpha(x_crit)) -- NOT ported from glworia's own JAX-based numerical
    critical-curve tracer, but verified to agree with direct image-counting via
    infer_with_fiona.fisher.cis_go.cis_images to <1e-3 for x_c in [0.02, 0.2].'''
    xc = np.asarray(xc, dtype=float)
    xc_safe = np.maximum(xc, 1.0e-12)
    s_crit = 0.5 * (-xc_safe + np.sqrt(xc_safe ** 2 + 4.0 * xc_safe))
    x_crit = np.sqrt(np.maximum(s_crit ** 2 - xc_safe ** 2, 0.0))
    alpha_crit = x_crit / (s_crit + xc_safe)
    y_crit = np.abs(x_crit - alpha_crit)
    return np.where(xc < 1.0e-10, 1.0, y_crit)


_SHM_REFS = []  # keeps SharedMemory blocks alive for the process's lifetime (see
                # _alloc_shared_array) -- letting one get garbage-collected would unlink
                # the underlying memory out from under the F table using it.


def _alloc_shared_array(shape, dtype):
    '''Allocate a numpy array backed by a multiprocessing.shared_memory.SharedMemory block
    (RAM, not disk) instead of an ordinary numpy heap array. This is the fix for the OOM
    crash found 2026-09-21: bilby's dynesty sampler pool (multiprocessing.Pool with no
    explicit context -> "fork" on Linux) forks N_WORKERS worker processes off THIS process,
    each of which repeatedly queries the F(w) table over 10,000+ sampler
    iterations. An ordinary in-process array sitting in each worker's own private/anonymous
    memory gradually balloons in real memory use across that many long-running forked
    workers (confirmed: OOM after ~2.5h/~10,400 iterations of sampling, well past table-
    build, with the old ~320 GB 4D table). A SharedMemory-backed array is NOT part of
    any single process's private heap -- the table is used in place (no copy; verified
    2026-09-21 for the earlier interpolator, and table indexes it directly), and forked
    child processes automatically inherit the SAME underlying mapping (no explicit
    re-attach needed, since fork -- unlike spawn -- inherits open mappings), so all
    N_WORKERS workers share one real copy of the table instead of each independently
    growing their own.

    NOTE: this does NOT reduce the memory you need to REQUEST (--mem) -- the table (~5 GB
    at grid-res=high) plus per-worker overhead still needs to fit in
    the job's memory allocation. What this fixes is the uncontrolled GROWTH during
    sampling, not the baseline size.'''
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    _SHM_REFS.append(shm)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return arr


_F_PROVIDER = None


def _get_F():
    '''The F(w; x_c, y) provider (amplification.HybridF): the cached (x_c, y, w) FIONA table,
    loaded into shared memory (or built and cached on a miss), with the GO residual and the
    lens-dependent bGO switch. Created once in the parent process, before bilby forks its
    sampler pool, so workers share the table pages.'''
    global _F_PROVIDER
    if _F_PROVIDER is None:
        _F_PROVIDER = amplification.HybridF(
            XC_MIN, XC_MAX, Y_MIN, Y_MAX, N_GRID, MLZ_MIN, MLZ_MAX, F_MAX, GL_DIR, N_WORKERS,
            alloc=_alloc_shared_array, root=ROOT, gl_schedule=args.gl_schedule,
            switch_path=args.bgo_switch, switch_eps=args.bgo_switch_eps,
            time_reference=args.time_reference, fourier_convention=args.fourier_convention,
            cache_dir=None if args.no_table_cache else args.table_cache)
    return _F_PROVIDER


def F_of_theta(frequency_array, logM, xc, y):
    '''CIS amplification factor F(f) at (M_Lz = exp(logM), x_c, y), on bilby's frequency array.
    w uses glworia's own Mtow (MTOW_GLWORIA), matching their w = 8 pi G M_Lz f / c^3 exactly.
    f = 0 (bilby's first bin, where h_+/h_x vanish anyway) gets a tiny placeholder frequency;
    w below the table's range is clamped to its lowest node, where F ~ 1.'''
    f_safe = np.where(frequency_array <= 0.0, 1.0e-3, frequency_array)
    w_vec = MTOW_GLWORIA * np.exp(logM) * f_safe
    return _get_F()(w_vec, xc, y)


_get_F()   # load/build now, in the parent process
amplification.self_check(_get_F(), M_LZ_FID, XC_FID, Y_FID,
                      np.arange(minimum_frequency, F_MAX + 1e-9, 1.0 / duration),
                      MTOW_GLWORIA, GL_DIR, N_WORKERS)

# ═══ Section 3: bilby frequency-domain source model, mirroring glworia/pe/waveform.py's
# lal_binary_black_hole_lensed EXACTLY except F_interp -> our own F_of_theta (FIONA/bGO) ═══
def lal_binary_black_hole_lensed_fiona(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, MLz, y, lp, **kwargs):
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomXHM', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    waveform_dict = _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)
    logM = np.log(MLz)
    Fs = F_of_theta(frequency_array, logM, lp, y)
    waveform_dict["plus"] = waveform_dict["plus"] * Fs
    waveform_dict["cross"] = waveform_dict["cross"] * Fs
    return waveform_dict


waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=lal_binary_black_hole_lensed_fiona,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=dict(waveform_approximant='IMRPhenomXHM', reference_frequency=50.0,
                            minimum_frequency=20.0),
)

# ═══ Section 4: detectors + noise, exactly as glworia's PE_full.py does it ═══════════════
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
if args.zero_noise:
    ifos.set_strain_data_from_zero_noise(
        sampling_frequency=sampling_frequency, duration=duration,
        start_time=injection_parameters["geocent_time"] - 2,
    )
else:
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration,
        start_time=injection_parameters["geocent_time"] - 2,
    )
ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)

snr_h1 = ifos[0].meta_data['optimal_SNR'] if 'optimal_SNR' in ifos[0].meta_data else None
print('Injection complete. Network matched-filter SNRs:',
      [ifo.meta_data.get('optimal_SNR') for ifo in ifos])

# ═══ Section 5: priors, mirroring glworia/pe/prior.py + PE_full.py exactly ═══════════════
class Uniform2DMaskDist(BaseJointPriorDist):
    """A 2D joint prior with a mask around the caustic curve in the (y, lp) plane --
    ported verbatim from glworia/pe/prior.py (bilby-only dependencies, no glworia import
    needed)."""

    def __init__(self, names, bounds, crit_mask_settings=None, mask_boxes=[]):
        super(Uniform2DMaskDist, self).__init__(names=names, bounds=bounds)
        self.names = names
        self.bounds = bounds
        self.bounds_arr = np.array([bounds[names[0]], bounds[names[1]]])
        if crit_mask_settings is not None:
            self.crit_mask_settings = {'fac': 0.5, 'cap_high': 0.2, 'cap_low': 0.0}
            self.crit_mask_settings.update(crit_mask_settings)
            self.lens_param_to_y_crit = self.crit_mask_settings['lens_param_to_y_crit']
            self.fac = self.crit_mask_settings['fac']
            self.cap_high = self.crit_mask_settings['cap_high']
            self.cap_low = self.crit_mask_settings['cap_low']
        else:
            self.crit_mask_settings = None
        self.mask_boxes = mask_boxes

    def _rescale(self, samp, **kwargs):
        scaled_samps = self.bounds_arr[:, 0] + (self.bounds_arr[:, 1] - self.bounds_arr[:, 0]) * samp
        if self.crit_mask_settings is not None:
            for i in range(scaled_samps.shape[0]):
                samp_i = scaled_samps[i, :]
                masked = self.check_mask(samp_i)
                while masked:
                    samp_new = (bilby_random.rng.uniform(0, 1, 2)
                               * (self.bounds_arr[:, 1] - self.bounds_arr[:, 0])
                               + self.bounds_arr[:, 0])
                    masked = self.check_mask(samp_new)
                    scaled_samps[i, :] = samp_new
        return scaled_samps

    def _sample(self, size, **kwargs):
        samps = np.zeros((size, 2))
        for i in range(size):
            vals = bilby_random.rng.uniform(0, 1, 2)
            samp = np.atleast_1d(self.rescale(vals))
            samps[i, :] = samp
        return samps

    def _ln_prob(self, samp, lnprob, outbounds):
        lnprob = np.logaddexp(lnprob, 0)
        lnprob[outbounds] = -np.inf
        for i in range(samp.shape[0]):
            if self.check_mask(samp[i, :]):
                lnprob[i] = -np.inf
        return lnprob

    def check_mask(self, samp):
        y, lp = samp[0], samp[1]
        y_crit = self.lens_param_to_y_crit(lp)
        masked = np.abs(y - y_crit) < np.max((np.min((self.fac * y_crit, self.cap_high)), self.cap_low))
        for box in self.mask_boxes:
            masked = masked or (box[0][0] < y < box[0][1] and box[1][0] < lp < box[1][1])
        return masked


class Uniform2DMask(JointPrior):
    def __init__(self, dist, name=None, latex_label=None, unit=None):
        if not isinstance(dist, Uniform2DMaskDist):
            raise JointPriorDistError("dist object must be instance of Uniform2DMaskDist")
        super(Uniform2DMask, self).__init__(dist=dist, name=name, latex_label=latex_label, unit=unit)


priors = bilby.gw.prior.BBHPriorDict()

if args.m_lz_prior_type == 'log-uniform':
    priors["MLz"] = bilby.core.prior.LogUniform(
        minimum=MLZ_MIN, maximum=MLZ_MAX, name="MLz", latex_label="$M_{Lz}$", unit="$M_\\odot$",
    )
else:
    priors["MLz"] = bilby.core.prior.Uniform(
        minimum=MLZ_MIN, maximum=MLZ_MAX, name="MLz", latex_label="$M_{Lz}$", unit="$M_\\odot$",
    )

# Exactly what glworia's released scripts/PE_full.py does for the CIS benchmarks: the
# config's crit_mask_settings (fac 1.0, caps 0.15/0.15) with glworia's OWN y_crit interpolator
# (pickled interp1d; it returns a ~-99 sentinel where CIS has no caustic, x_c >~ 0.49 and
# x_c = 0, which switches the mask off there), and NO mask box: the configs list mask_boxes
# [[[0.7, 1.1], [0.0, 0.05]]], but PE_full.py never passes them to Uniform2DMaskDist.
# table.cis_prior_masked implements the same mask; the F table's row trimming uses it.
crit_mask_settings = dict(fac=1.0, cap_high=0.15, cap_low=0.15,
                          lens_param_to_y_crit=amplification.FWT.glworia_y_crit)
mask_boxes = []

ylp_masked = Uniform2DMaskDist(
    names=['y', 'lp'],
    bounds={'y': (Y_MIN, Y_MAX), 'lp': (XC_MIN, XC_MAX)},
    crit_mask_settings=crit_mask_settings, mask_boxes=mask_boxes,
)
priors['y'] = Uniform2DMask(ylp_masked, name='y', latex_label='$y$', unit=None)
priors['lp'] = Uniform2DMask(ylp_masked, name='lp', latex_label='$x_c$', unit=None)

priors["luminosity_distance"] = bilby.core.prior.Uniform(
    minimum=args.dl_prior_min, maximum=args.dl_prior_max, name="luminosity_distance", latex_label="$d_L$", unit="Mpc",
)

time_delay = ifos[0].time_delay_from_geocenter(
    injection_parameters["ra"], injection_parameters["dec"], injection_parameters["geocent_time"],
)
priors["H1_time"] = bilby.core.prior.Uniform(
    minimum=injection_parameters["geocent_time"] + time_delay - 0.1,
    maximum=injection_parameters["geocent_time"] + time_delay + 0.1,
    name="H1_time", latex_label="$t_H$", unit="$s$",
)
del priors["ra"], priors["dec"]
priors["zenith"] = bilby.core.prior.Sine(latex_label="$\\kappa_z$")
priors["azimuth"] = bilby.core.prior.Uniform(
    minimum=0, maximum=2 * np.pi, latex_label="$\\epsilon_a$", boundary="periodic",
)

for key in ["tilt_1", "tilt_2", "phi_12", "phi_jl"]:
    priors[key] = injection_parameters[key]

if args.narrow_priors:
    NP = NARROW_PRIORS
    priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(
        minimum=NP['chirp_mass'][0], maximum=NP['chirp_mass'][1], name='chirp_mass',
        latex_label='$\\mathcal{M}$')
    priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(
        minimum=NP['mass_ratio'][0], maximum=NP['mass_ratio'][1], name='mass_ratio', latex_label='$q$')
    for k in ('a_1', 'a_2'):
        priors[k] = bilby.core.prior.Uniform(minimum=NP[k][0], maximum=NP[k][1], name=k,
                                             latex_label=f'$a_{k[-1]}$')
    priors['theta_jn'] = bilby.core.prior.Sine(minimum=NP['theta_jn'][0], maximum=NP['theta_jn'][1],
                                               name='theta_jn', latex_label='$\\theta_{JN}$')
    t_h1 = injection_parameters["geocent_time"] + time_delay
    priors["H1_time"] = bilby.core.prior.Uniform(
        minimum=t_h1 - NP['H1_time_halfwidth'], maximum=t_h1 + NP['H1_time_halfwidth'],
        name="H1_time", latex_label="$t_H$", unit="$s$")
    priors["zenith"] = bilby.core.prior.Sine(minimum=NP['zenith'][0], maximum=NP['zenith'][1],
                                             name='zenith', latex_label="$\\kappa_z$")
    priors["azimuth"] = bilby.core.prior.Uniform(minimum=NP['azimuth'][0], maximum=NP['azimuth'][1],
                                                 name='azimuth', latex_label="$\\epsilon_a$")
    print('[priors] --narrow-priors:', {k: priors[k] for k in
          ('MLz', 'luminosity_distance', 'chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'theta_jn',
           'H1_time', 'zenith', 'azimuth')}, flush=True)

priors.validate_prior(duration, minimum_frequency)

# ═══ Section 6: likelihood, exactly as glworia's PE_full.py configures it ════════════════
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    priors=priors,
    distance_marginalization=False,
    phase_marginalization=False,
    time_marginalization=True,
    reference_frame="H1L1",
    time_reference="H1",
)

# ═══ Section 7: sampler -- bilby.run_sampler(sampler='dynesty', nlive=...), everything
# else left at bilby's OWN current defaults (their config only ever specifies nlive too,
# and no bilby/dynesty version pin exists anywhere in their repo to match more precisely) ═
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler='dynesty',
    nlive=args.nlive,
    npool=N_WORKERS,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    result_class=bilby.gw.result.CBCResult,
)

result.plot_corner()
print('\nDone. Results in', outdir)
