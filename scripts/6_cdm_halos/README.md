# Cold-dark-matter halos (Sec. V A, Fig. `cdm_nfw`)

Detectability of NFW halos on the Diemer & Joyce (2019) concentration-mass relation, for the LIGO benchmark and the
four LISA binaries of the Fisher forecasts (full IMRPhenomD band, native signal-to-noise ratio).

The shipped results in `data/cdm/` reproduce the paper's figure:

```bash
TAG=_imr python scripts/6_cdm_halos/plot_cdm_halos.py        # -> outputs/figures/cdm_nfw_fisher_imr.{png,pdf}
```

To recompute (raw FIONA runs go to `outputs/cdm/`, or `$CDM_RAW`; each LISA run takes 10-40 min on 112 cores):

```bash
export LISA_BAND=imr NATIVE_SNR=1                             # full band, native SNR
for S in steep_core cis_x2; do                                # two quadrature schedules (4x and 2x the CIS one)
  python scripts/6_cdm_halos/fiona_cdm_rate.py SMBHB_1e5 --sched $S \
      --out outputs/cdm/fiona_cdm_rate_SMBHB_1e5_${S}_imr.npz
done
# third check with 8x the CIS schedule for M200 >= 1e9 Msun (grid indices 24-36)
python scripts/6_cdm_halos/fiona_cdm_rate_x8.py SMBHB_1e5 --midx 29,30,31,32,33,34,35,36 \
    --out outputs/cdm/fiona_cdm_rate_SMBHB_1e5_x8_imr.npz
python scripts/6_cdm_halos/fiona_cdm_rate_x8.py SMBHB_1e5 --midx 24,25,26,27,28 \
    --out outputs/cdm/fiona_cdm_rate_SMBHB_1e5_x8lo_imr.npz
python scripts/6_cdm_halos/merge_x8.py SMBHB_1e5               # -> *_imrhi.npz: 8x vs 4x where rerun, else 4x vs 2x
IN_TAG=_imrhi TAG=_imr python scripts/6_cdm_halos/fiona_cdm_rate_post.py SMBHB_1e5   # -> data/cdm/
```

Do not set `NATIVE_SNR=1` for the post-processing of runs that were already made with `NATIVE_SNR=1`: it rescales
the signal from SNR 1e3 to the values in `data/cdm/native_snr.npz` (`native_snr.py`). The paper's 10^7+10^7 binary
was run at SNR 1e3 (`IN_TAG=_imr1e3`) and post-processed with `NATIVE_SNR=1`.

Convergence: only halos whose Delta chi^2 agrees to 2% between two schedules (and Fisher errors to 5%) are used.
Agreement between 2x and 4x alone is not sufficient above the masses quoted in the paper: there the 8x schedule
can differ by large factors, so the 8x check is part of the pipeline.

Files: `scan_detect.py` (sources, band, SNR), `halo_born.py` (halo parameters), `nfw_stable.py` (NFW potential in a
form stable at x << r_s), `fiona_cdm.py` (LIGO), `fiona_cdm_rate.py` (LISA, all lens redshifts), `fiona_cdm_rate_post.py`
(intrinsic-parameter marginalization, likelihood-ratio and Bayes-factor criteria, optical depth).
