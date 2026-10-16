#!/usr/bin/env python3
"""
Bayesian PE for the dark-matter lens profiles of the paper's Sec. "Constraints on dark matter
models": the ultralight-axion (ULA) soliton (--lens ULA) and the primordial-black-hole point
lens (--lens PBH). The SIDM case is the CIS lens (run_inference_cis.py).

Built from run_inference_cis.py (2026-10-09) and identical to it in everything not
specific to the lens: injection (masses, spins, angles, distance, sky, time), bilby noise
realisation (seed 1234), PSD, waveform (IMRPhenomXHM, f_ref 50 Hz), likelihood (H1L1 frame,
time_reference H1), sampler (dynesty, nlive 1000), the C24 priors on all source parameters,
and every correction of the CIS pipeline:
  - F from the cached (shape, y, w) FIONA table with the GO residual and a lens-dependent
    bGO switch measured for THIS lens (derive_bgo_switch_ula_pbh.py); amplification.HybridF.
    GL schedule v4x4 (4x the CIS-derived v4): both lenses have a far steeper core than any
    CIS in the v4 derivation (point lens: softening 1e-5; ULA at r_s = 0.01: central
    convergence ~6e4), and FIONA's error there halves per doubling of the node count. Against
    the exact point-mass F, v4 / v4x2 / v4x4 reach 3.2e-3 / 1.6e-3 / 8.0e-4
    (check_point_lens_exact.py). Table validated by validate_table_ula_pbh.py.
  - F in LAL's Fourier convention (conj F) and glworia's minimum-image time reference.
  - M_Lz: log-uniform over [0.1, 2000] Msun by default (the paper's prior), w up to 254.

Lens specifics:
  ULA: shape parameter r_s (soliton core radius, fiona.lenses.ULASoliton, alpha(1) = 1),
       prior uniform in [0.01, 1] (as the earlier ULA pipeline), injected r_s = 0.1. The
       joint (y, r_s) prior masks |y - y_crit(r_s)| < 0.15 around the radial caustic, the same
       band as the CIS mask (the table's GO residual is not valid there). For r_s < ~0.3 the
       caustic lies above y = 5, so the mask only acts for r_s ~0.3-1.
  PBH: point lens with the fixed Plummer softening 1e-5 (not sampled), y uniform in [0.1, 5].
       The shape parameter 'lp' is held fixed at the softening (bilby fixed prior).

The sampled shape parameter keeps the CIS script's name 'lp' (glworia's "lens parameter").

IMPORTANT: `finufft` must be imported before any infer_with_fiona.fisher/fiona import.
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
ap.add_argument('--lens', choices=['ULA', 'PBH'], required=True)
ap.add_argument('--m-lz-fid', type=float, default=100.0,
                 help='Fiducial (injected) M_Lz [Msun]. (default: 100.0)')
ap.add_argument('--shape-fid', type=float, default=0.1,
                 help='Fiducial (injected) ULA core radius r_s (ignored for PBH). (default: 0.1)')
ap.add_argument('--y-fid', type=float, default=0.3,
                 help='Fiducial (injected) source position y. (default: 0.3)')
ap.add_argument('--grid-res', choices=['tiny', 'low', 'medium', 'high'], default='high',
                 help="(shape, y) density of the F table: 'high'=201 (validated), 'medium'=101, "
                      "'low'=61, 'tiny'=21 (smoke tests only). (default: high)")
ap.add_argument('--shape-prior-min', type=float, default=0.01,
                 help='ULA r_s prior lower bound. (default: 0.01)')
ap.add_argument('--shape-prior-max', type=float, default=1.0,
                 help='ULA r_s prior upper bound. (default: 1.0)')
ap.add_argument('--y-prior-min', type=float, default=0.1)
ap.add_argument('--y-prior-max', type=float, default=5.0)
ap.add_argument('--dl-prior-min', type=float, default=1.0)
ap.add_argument('--dl-prior-max', type=float, default=5000.0)
ap.add_argument('--m-lz-prior-min', type=float, default=0.1)
ap.add_argument('--m-lz-prior-max', type=float, default=2000.0)
ap.add_argument('--m-lz-prior-type', choices=['uniform', 'log-uniform'], default='log-uniform',
                 help="(default: log-uniform, the paper's prior)")
ap.add_argument('--gl-schedule', default='steep_core', choices=quadrature.CHOICES,
                 help='Gauss-Legendre node schedule (infer_with_fiona.quadrature). steep_core is 4x '
                      'the cis schedule: cis itself, derived for CIS, is off by up to ~4e-3 for these '
                      'steep-core lenses. (default: steep_core)')
ap.add_argument('--table-cache', default=str(paths.TABLE_CACHE_DIR))
ap.add_argument('--no-table-cache', action='store_true')
ap.add_argument('--bgo-switch', default=None,
                 help='derive_bgo_switch_ula_pbh.py output (default: data/bgo_switch/ula.npz '
                      'or data/bgo_switch/pbh.npz).')
ap.add_argument('--bgo-switch-eps', type=float, default=1e-3)
ap.add_argument('--fourier-convention', choices=['lal', 'fiona'], default='lal')
ap.add_argument('--time-reference', choices=['min_image', 'absolute'], default='min_image')
ap.add_argument('--n-workers', type=int, default=112)
ap.add_argument('--nlive', type=int, default=1000)
ap.add_argument('--zero-noise', action='store_true')
ap.add_argument('--seed', type=int, default=1234)
ap.add_argument('--out-label', default=None)
ap.add_argument('--output-dir', default=None,
                 help='(default: outputs/runs/<lens>_glworia_repro)')
args = ap.parse_args()

LENS = args.lens
LENS_MODEL = {'ULA': 'ULA', 'PBH': 'PointLens'}[LENS]
if args.bgo_switch is None:
    args.bgo_switch = str(paths.BGO_SWITCH[LENS])
if args.output_dir is None:
    args.output_dir = str(paths.RUNS_DIR / f'{LENS}_glworia_repro')
if LENS == 'PBH':
    args.shape_fid = args.shape_prior_min = args.shape_prior_max = \
        amplification.FWT.POINT_LENS_SOFTENING

N_WORKERS = args.n_workers
M_LZ_FID = args.m_lz_fid
XC_FID = args.shape_fid
Y_FID = args.y_fid

label = args.out_label or (f'{LENS}_glworia_repro_mlz{M_LZ_FID:g}'
                           + (f'_rs{XC_FID:g}' if LENS == 'ULA' else '') + f'_y{Y_FID:g}')
outdir = os.path.join(ROOT, args.output_dir, label)
os.makedirs(outdir, exist_ok=True)

bilby.core.utils.setup_logger(outdir=outdir, label=label, log_level='info')
bilby_random.seed(args.seed)

# ═══ Exact constants used by glworia's own waveform.py (Mtow = 8 pi G / c^3 * Msun) ══════
# Deliberately using scipy.constants (their own source) rather than infer_with_fiona.fisher.utils's
# own rounded G_SI/c_SI/M_sun_kg, for exact numerical parity in the w = Mtow*MLz*f formula.
MSUN_GLWORIA = 1.9884099e+30
MTOW_GLWORIA = 8.0 * np.pi * G_SI_SCIPY / C_SI ** 3 * MSUN_GLWORIA

# ═══ Section 1: injection parameters, as the CIS benchmarks (PE_submission_settings/CIS/) ═
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

# ═══ Section 2: the F(w, y, shape) table (cached FIONA table + GO residual + bGO switch; see
# amplification). Loaded, or built and cached, once below, before bilby forks its sampler pool. ═
MLZ_MIN, MLZ_MAX = args.m_lz_prior_min, args.m_lz_prior_max  # (0.1, 2000) by default
XC_MIN, XC_MAX = args.shape_prior_min, args.shape_prior_max   # shape-parameter prior
Y_MIN, Y_MAX = args.y_prior_min, args.y_prior_max       # Cheung+2024's own prior: (0.1,5.0)
F_MAX = sampling_frequency / 2.0
N_GRID = {'tiny': 21, 'low': 61, 'medium': 101, 'high': 201}[args.grid_res]
# 'high'=201 matches glworia's own N_grid_strong (their FINER, strong-lensing-side
# resolution) -- used here EVERYWHERE, not just near the caustic. See "F(w,y,x_c) table
# resolution" in the module docstring for why (a two-grid weak/strong attempt cost ~401 GB).
LN10 = np.log(10.0)


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
            cache_dir=None if args.no_table_cache else args.table_cache,
            lens_model=LENS_MODEL)
    return _F_PROVIDER


def F_of_theta(frequency_array, logM, xc, y):
    '''Amplification factor F(f) at (M_Lz = exp(logM), shape, y), on bilby's frequency array.
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

if LENS == 'ULA':
    # Same joint prior as the CIS runs (glworia's Uniform2DMaskDist, fac 1.0, caps 0.15/0.15),
    # with the ULA radial caustic y_crit(r_s): the band |y - y_crit| < 0.15 is excluded.
    # table.ula_prior_masked implements the same mask; the F table's row trimming uses it.
    crit_mask_settings = dict(fac=1.0, cap_high=0.15, cap_low=0.15,
                              lens_param_to_y_crit=amplification.FWT.ula_y_crit_fast)
    ylp_masked = Uniform2DMaskDist(
        names=['y', 'lp'],
        bounds={'y': (Y_MIN, Y_MAX), 'lp': (XC_MIN, XC_MAX)},
        crit_mask_settings=crit_mask_settings, mask_boxes=[],
    )
    priors['y'] = Uniform2DMask(ylp_masked, name='y', latex_label='$y$', unit=None)
    priors['lp'] = Uniform2DMask(ylp_masked, name='lp', latex_label='$r_s$', unit=None)
else:
    priors['y'] = bilby.core.prior.Uniform(minimum=Y_MIN, maximum=Y_MAX, name='y',
                                           latex_label='$y$')
    priors['lp'] = XC_FID          # fixed point-lens softening (not sampled)

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
