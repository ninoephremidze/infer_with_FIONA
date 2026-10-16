#!/usr/bin/env python
"""Check that F multiplies LAL/bilby waveforms in the right Fourier convention: in the time
domain, the later lensed image must arrive LATER than the first one.

Builds the lensed waveform exactly as the CIS PE scripts do (bilby IMRPhenomXHM h_+(f) x F, with
the min-image time reference), for the strong-lensing benchmark lens (x_c = 0.02, y = 0.3) at
M_Lz = 2000 Msun, where the saddle image arrives 23 ms after the minimum image. Compares the
Hilbert envelope at t0 + dt and t0 - dt around the unlensed merger time t0.

  default : F from the geometric-optics sum, converted with amplification's own functions
            (time_reference_factor, to_fourier_convention) -- seconds, login node is fine.
  --full  : F from the production amplification.HybridF (cached F table, ~5 GB in memory).

Expected (fourier_convention='lal', the default since 2026-10-07): envelope at t0 + dt ~ 1.5
(the saddle image, sqrt|mu| = 1.53) and no merger burst at t0 - dt beyond the first image's
inspiral. With 'fiona' (e^{+iwT}, glworia's convention) the saddle burst appears at t0 - dt.
Exit status 1 if the 'lal' convention fails.
"""
import argparse
import os
import sys

import numpy as np
from scipy.signal import hilbert

import finufft  # noqa: F401,E402 -- import order used by the PE scripts
from infer_with_fiona import paths  # noqa: E402 -- after finufft
from infer_with_fiona import table as FWT  # noqa: E402
from infer_with_fiona import amplification  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--full', action='store_true', help='use the production HybridF (cached F table)')
args = ap.parse_args()

import bilby  # noqa: E402
bilby.core.utils.logger.setLevel('ERROR')

XC, Y, M_LZ = 0.02, 0.3, 2000.0
FS, T = 4096.0, 8.0
wg = bilby.gw.WaveformGenerator(
    duration=T, sampling_frequency=FS, frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=dict(waveform_approximant='IMRPhenomXHM', reference_frequency=50.0, minimum_frequency=20.0))
par = dict(mass_1=30., mass_2=30., a_1=0., a_2=0., tilt_1=0., tilt_2=0., phi_12=0., phi_jl=0.,
           luminosity_distance=500., theta_jn=0., psi=0., phase=0., geocent_time=0., ra=0., dec=0.)
hp = wg.frequency_domain_strain(par)['plus']
f = wg.frequency_array
w = FWT.W_TO_MF * M_LZ * np.where(f > 0, f, 1.0e-3)

phi, mu, n = FWT.cis_image_set(XC, Y)
i_sad = int(np.where(n == 0.5)[0][0])
dt = 4 * 4.925490947641267e-06 * M_LZ * (phi[i_sad] - phi.min())
amp_sad = np.sqrt(abs(mu[i_sad]))

if args.full:
    hyb = {c: amplification.HybridF(0.0, 1.0, 0.1, 5.0, 201, 0.1, 2000.0, 1024.0,
                                 str(paths.GL_RULES_DIR), 8, fourier_convention=c)
           for c in ('lal', 'fiona')}
    F_of = lambda conv: hyb[conv](np.clip(w, None, hyb[conv].w_max), XC, Y)
    source = 'production HybridF (F table + bGO switch)'
else:
    ref = amplification.time_reference_factor(w, XC, Y, 'min_image')
    F_of = lambda conv: amplification.to_fourier_convention(FWT.F_go(w, phi, mu, n) * ref, conv)
    source = 'geometric-optics F with amplification conversions'


def envelope(hf):
    ht = np.fft.irfft(hf) * FS
    t = np.arange(len(ht)) / FS
    t = np.where(t > T / 2, t - T, t)
    o = np.argsort(t)
    return t[o], np.abs(hilbert(ht[o]))


t, e0 = envelope(hp)
t0 = t[np.argmax(e0)]
at = lambda e, tt: e[np.argmin(np.abs(t - tt))] / e0.max()
print(f'F: {source}.  Lens x_c={XC}, y={Y}, M_Lz={M_LZ:g}: saddle image (sqrt|mu| = {amp_sad:.2f}) '
      f'should arrive {1e3 * dt:.1f} ms AFTER the minimum image.')
print(f'  unlensed: envelope at t0 - dt / t0 + dt = {at(e0, t0 - dt):.2f} / {at(e0, t0 + dt):.2f}')
ok = {}
for conv in ('lal', 'fiona'):
    Fw = F_of(conv)
    Fw[f < 20] = 1.0
    _, e = envelope(hp * Fw)
    late, early = at(e, t0 + dt), at(e, t0 - dt)
    ok[conv] = abs(late - amp_sad) < 0.2 * amp_sad          # saddle burst found where it belongs
    print(f'  {conv:5s}: envelope at t0 - dt = {early:.2f}, at t0 = {at(e, t0):.2f}, at t0 + dt = {late:.2f}'
          f'  -> delayed image {"AFTER the first image (correct)" if ok[conv] else "NOT at t0 + dt"}')
if not ok['lal'] or ok['fiona']:
    print('FAIL: expected the delayed image at t0 + dt for lal and not for fiona')
    sys.exit(1)
print('PASS: fourier_convention="lal" puts the delayed image after the first image.')
