"""Native (unnormalized) SNR of the paper's LISA sources (last month before merger, z_s = 3), with the same
waveform, frequency grid, PSD and band (LISA_BAND=imr: full IMRPhenomD band, the paper) as scan_detect.py.
fiona_cdm_rate_post.py with NATIVE_SNR=1 uses it to rescale runs made at SNR = 1e3 to the native SNR. Runs made
with NATIVE_SNR=1 (scan_detect.py) are already at native SNR and must be post-processed without it."""
import sys, os, numpy as np
from infer_with_fiona.fisher import imrphenomd_pycbc_amplitude, lisa_psd, f_isco, f_1month, luminosity_distance
from infer_with_fiona.fisher.fisher import _inner_product
out = {}
for name, m in (('SMBHB_1e4', 1e4), ('SMBHB_1e5', 1e5), ('SMBHB_1e6', 1e6), ('SMBHB_1e7', 1e7)):
    z_s = 3.0; M_tot, eta = 2 * m, 0.25
    f_lo, f_hi = max(f_1month(M_tot, eta, z_s), 1e-4), min(f_isco(M_tot, z_s), 1.0)
    if os.environ.get('LISA_BAND', 'imr') == 'imr':
        f_hi = min(0.2 / (M_tot * (1 + z_s) * 4.925491e-6), 1.0)
    f = np.geomspace(f_lo, f_hi, 768)
    h0 = imrphenomd_pycbc_amplitude(f, m, m, z_s, luminosity_distance(z_s), f_lo, f_hi, delta_f=1e-7)
    S = lisa_psd(f); k = h0 > 0
    out[name] = float(np.sqrt(_inner_product(h0[k], h0[k], S[k], f[k])))
    print(name, 'native SNR %.1f' % out[name])
from infer_with_fiona import paths
os.makedirs(paths.DATA_DIR / 'cdm', exist_ok=True)
np.savez(paths.DATA_DIR / 'cdm' / 'native_snr.npz', **out)
