# Comparison with glworia

These files reproduce the reference side of the paper's pipeline validation: C24's released
[`glworia`](https://github.com/mhycheung/glworia) code, run on the same data, priors and sampler
settings as this pipeline.

| File | Contents |
|---|---|
| `glworia_base_commit.txt` | The `glworia` commit the comparison used. |
| `glworia_local_changes.patch` | The changes applied to it (see below). |
| `configs/` | The four CIS benchmark configurations, with the log-uniform $M_{Lz}$ prior. |
| `submit_glworia.sh` | Slurm script that runs glworia's `scripts/PE_full.py` on one configuration. |
| `check_time_reference.py`, `check_time_reference.sh` | Compares FIONA's $F$ with glworia's interpolated $F$ as complex numbers. |

The patch makes three changes.

- **Fourier convention.** It applies $F^*$ to the LAL waveforms, as this pipeline does. Without it, later images would arrive earlier.
- **Image-finding settings.** It sets four image-finding parameters that the released interpolation settings omit, using the values of the matching critical-curve settings in the same file.
- **JAX import.** It updates one import for current JAX releases.

## Setup

```bash
git clone https://github.com/mhycheung/glworia.git
cd glworia
git checkout $(cat ../infer_with_FIONA/glworia_comparison/glworia_base_commit.txt)
git apply ../infer_with_FIONA/glworia_comparison/glworia_local_changes.patch
cp ../infer_with_FIONA/glworia_comparison/configs/*.json PE_submission_settings/CIS/
```

Build glworia's CIS interpolation table with its own `scripts/do_interpolation.py`, in an environment with
the JAX version glworia requires. Then run each benchmark from the `infer_with_FIONA` root:

```bash
export GLWORIA_PREFIX=/path/to/glworia
sbatch glworia_comparison/submit_glworia.sh PE_submission_settings/CIS/MLz_100.0_dL_500.0_y_0.3000_xc_0.0200_lal_loguniform.json
```

glworia writes its results under `$GLWORIA_PREFIX/outdir/`. The figure scripts read them with
`--from-runs --glworia-dir $GLWORIA_PREFIX/outdir`.
