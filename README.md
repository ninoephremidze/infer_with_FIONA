# GW-lensing inference with FIONA

Code and data products for the paper

> **The power of gravitational-wave lensing to constrain dark matter: accelerated inference with FIONA**
> Nino Ephremidze, Cora Dvorkin, and Marc Kamionkowski

This repository contains the code behind every Fisher-forecast and Bayesian-inference result in the paper. It uses [FIONA](https://github.com/ninoephremidze/FIONA) ([arXiv:2603.12333](https://arxiv.org/abs/2603.12333)), a wave-optics solver that evaluates the gravitational-wave lensing amplification factor $F(w, y)$ for a whole batch of source positions and lens shapes with a single non-uniform fast Hankel transform.

## Where to start

| I want to... | Run |
|---|---|
| Get a guided tour of the results | the notebooks in `notebooks/` (below) |
| Remake the paper's figures | `scripts/5_figures/*.py`, which read the posterior samples shipped in `results/posteriors/` |
| Reproduce the Fisher forecasts | `scripts/1_fisher/fisher_forecasts.py` |
| Run Bayesian inference on a lensed signal | `scripts/4_inference/run_inference_cis.py` or `run_inference_ula_pbh.py` |
| Evaluate $F(w, y, l)$ in my own code | `infer_with_fiona.amplification.HybridF`, or FIONA itself in `infer_with_fiona.fiona` |

## What the package does

Gravitational waves passing a compact dark-matter lens pick up a frequency-dependent amplification factor $F(w, y, l)$. Here $w = 8\pi G M_{Lz} f / c^3$ is the dimensionless frequency, $y$ is the source position, and $l$ is a lens-shape parameter. The package provides two analyses built on FIONA.

1. **Fisher forecasts.** It computes Fisher-matrix constraints on $(\ln M_{Lz}, y, x_c)$ for a cored isothermal sphere (CIS) lens, for LISA and LIGO sources. Derivatives in $y$ and in the lens shape are batched into a single FIONA call.
2. **Full Bayesian parameter estimation.** It runs a `bilby` + `dynesty` analysis of a lensed GW150914-like event in a simulated H1+L1 network with O4 noise. It samples the 11 source parameters together with the lens parameters $(M_{Lz}, y, l)$, for three lens profiles:
   - **CIS**, also reinterpreted as a self-interacting-dark-matter (SIDM) core,
   - **ULA**, an ultralight-axion soliton,
   - **PBH**, a point-mass primordial black hole.

The likelihood never calls FIONA directly. Instead, $F$ is precomputed once on a three-dimensional table in $(x_c, y, w)$ and interpolated during sampling, which takes about 0.4 ms per frequency array. The table has no separate mass axis, because $F$ depends on $M_{Lz}$ and $f$ only through $w$.

### Main ingredients

- **A convergence-tested quadrature schedule.** The number of Gauss-Legendre nodes $n(w)$ was set by doubling $n = 250 \times 2^k$, up to 1,024,000, until $F$ changed by less than $10^{-3}$ for every lens in the prior. The schedule spans $w = 2\times10^{-4}$ to $254$.
- **A beyond-geometric-optics (bGO) switch.** Above a lens-dependent frequency $w_s(x_c, y)$, $F$ is evaluated with the closed-form bGO sum, which there agrees with converged FIONA to $10^{-3}$. The two are blended smoothly over $w_s \le w \le 1.2\,w_s$.
- **A geometric-optics residual table.** Above $w \sim 1$ the table stores $F - s(w) F_{\rm GO}$, and $F_{\rm GO}$ is added back exactly at the queried point. This keeps bilinear interpolation in $(x_c, y)$ accurate where $F$ oscillates rapidly.
- **Correct conventions for LAL waveforms.** The lensed waveform uses $F^*$, because LALSuite and bilby use the opposite Fourier sign convention. $F$ is also multiplied by $e^{-iw\phi_{\min}}$, so that arrival times are measured from the first image.

Against direct, converged FIONA evaluations at 130 test lenses across the CIS prior, the interpolated $F$ has these errors relative to each lens's median $|F|$:

| Statistic | Error |
|---|---|
| Median | 0.03% |
| 95th percentile | 0.2% |
| Maximum | 1.0% |

On the four CIS benchmarks of Cheung et al. (2024), hereafter C24, this pipeline and the released [`glworia`](https://github.com/mhycheung/glworia) code give posteriors whose medians agree to within $0.06\sigma$ in every parameter, on identical data. This pipeline completes each run 2.3 to 2.7 times faster.

## Notebooks

Three short notebooks walk through the results. They read the shipped data, so they run in a minute or two.

| Notebook | Contents |
|---|---|
| `notebooks/01_fisher_forecasts.ipynb` | FIONA's $F(w, y)$ against geometric optics, which part of $F$ each LISA and LIGO source probes, and the Fisher constraints and ellipses |
| `notebooks/02_fiona_vs_glworia.ipynb` | The validation against `glworia` on C24's four benchmarks: lens posteriors, agreement across all 14 parameters, and run cost |
| `notebooks/03_mass_scans.ipynb` | Inference for CIS, ULA and PBH lenses at $M_{Lz} = 3$ to $1000\,M_\odot$: what the data constrain as the lens mass grows |
| `notebooks/04_core_size_calibration.ipynb` | Fisher forecasts of the core size $x_c$ against the mass-scan posteriors, with the Fisher Gaussian restricted to the inference prior |

## Repository layout

```
infer_with_FIONA/
├── src/infer_with_fiona/       the installable package
│   ├── fiona/                  the FIONA solver, in the exact version used for the paper
│   ├── fisher/                 Fisher-matrix tools: waveforms, noise curves, derivatives of F, image solvers, plots
│   ├── quadrature/             Gauss-Legendre node schedules n(w) and the node cache
│   ├── table.py                the F(x_c, y, w) table, GO residual, bGO switch and caustic prior mask
│   ├── amplification.py        HybridF: the F used in the likelihood, with the LAL and time conventions
│   ├── posteriors.py           loaders for the shipped samples and for fresh bilby output
│   ├── dm_conversions.py       lens parameters to SIDM cross section and axion mass
│   └── paths.py                where data, caches and outputs live
├── scripts/
│   ├── 1_fisher/               Fisher forecasts for LISA and LIGO
│   ├── 2_convergence/          derive the quadrature schedule and the bGO switch; FIONA vs exact point lens
│   ├── 3_table/                build and validate the F table; check the Fourier convention
│   ├── 4_inference/            Bayesian inference drivers and posterior extraction
│   └── 5_figures/              the paper's posterior figures
├── data/                       inputs the code needs (described below)
├── notebooks/                  guided tours of the results
├── results/                    posterior samples of every run, and run statistics
├── slurm/                      one Slurm script per entry point
├── glworia_comparison/         how the glworia reference runs were made
└── outputs/                    everything the pipeline writes (git-ignored)
```

### Data files

| Path | Contents |
|---|---|
| `data/gl_rules/` | Precomputed Gauss-Legendre rules, $n = 250 \times 2^k$ up to 1,024,000. The largest took about 8 hours to generate. |
| `data/bgo_switch/{cis,ula,pbh}.npz` | The measured bGO switch maps $w_s$ for each lens. |
| `data/caustic_cis_ycrit.pkl` | `glworia`'s tabulated caustic $y_{\rm crit}(x_c)$, which defines the CIS prior mask. |
| `data/table_validation_points.json` | The test lenses for the CIS table validation. |
| `data/fisher_covariances.npz` | The Fisher covariances behind the two Fisher figures, with $F$ referenced to the first image. |
| `data/core_fisher_LIGO_CIS.npz` | Fisher grid of $(\ln M_{Lz}, y, x_c)$ over core size, lens mass and source position for the LIGO inference benchmark, used by notebook 04. |
| `data/nufht_tables.cache` | Precomputed tables of the compiled NUFHT, which it otherwise regenerates in every process. |

### Outputs

Every script writes under `outputs/`, so the repository itself stays clean.

| Path | Contents |
|---|---|
| `outputs/table_cache/` | Interpolation tables, keyed by a hash of their configuration. |
| `outputs/runs/` | `bilby` output of inference runs. |
| `outputs/figures/` | Figures. |
| `outputs/checks/` | Convergence and validation results. |
| `outputs/logs/` | Slurm logs. |

## Installation

```bash
git clone https://github.com/ninoephremidze/infer_with_FIONA.git
cd infer_with_FIONA
conda create -n gwlens python=3.11
conda activate gwlens
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins the versions used for the paper.

| Package | Version |
|---|---|
| Python | 3.11.14 |
| numpy | 2.3.5 |
| scipy | 1.16.2 |
| bilby | 2.8.1 |
| dynesty | 3.0.0 |
| lalsuite | 7.26.1 |
| PyCBC | 2.9.0 |
| finufft | 2.4.1 |

FIONA's axisymmetric solver also needs the compiled batched Hankel-transform extension `_pynufht`. Build it from the [NUFHT](https://github.com/marckamion/NUFHT) repository, following its README, and put the resulting module on your `PYTHONPATH` or install it into the environment.

```bash
git clone https://github.com/marckamion/NUFHT.git
cd NUFHT/python
python setup.py build_ext --inplace
```

The package vendors FIONA as `infer_with_fiona.fiona`, so it does not clash with a separate FIONA installation. This copy adds the ULA soliton lens and the batched multi-lens solver that the pipeline needs, which are not in the currently released FIONA. Always `import finufft` before importing FIONA, as the scripts do.

All paths are resolved from the repository root, so scripts can be run from any directory. To keep data or outputs elsewhere, set `INFER_WITH_FIONA_ROOT`.

## Reproducing the results

The commands below are run from the repository root.

### 1. Fisher forecasts

```bash
python scripts/1_fisher/fisher_forecasts.py plot      # fisher_lisa.png, fisher_ligo.png from the shipped covariances
python scripts/1_fisher/fisher_forecasts.py compute   # rerun the Fisher matrices with FIONA
sbatch slurm/core_size_fisher.sh                      # Fisher grid of the core size for the LIGO benchmark (notebook 04)
```

The lens is a CIS with $y = 0.3$ and $x_c = 0.01$. Each source is paired with the lens mass at which $w = 10$ at $f_*$, the frequency below which 90% of its unlensed SNR$^2$ accumulates (`--w-star`), so that its band covers the wave-optics regime $w \sim 1$. $F$ and its derivatives come from FIONA with the converged `cis` schedule, blended into the beyond-geometric-optics sum above the measured switch frequency $w_s(x_c, y)$, with the same weight on $F$ and every derivative. LISA forecasts are normalized to SNR $10^3$ and LIGO forecasts to SNR $10^2$. As in Tambalo et al. (2023), $F$ is referenced to the first-arriving image, so a lens-dependent overall time shift does not count as information about the lens. `plot` reproduces the paper's figures exactly, and `compute` reproduces the shipped covariances exactly.

### 2. Quadrature schedule and bGO switch

These steps are optional, because their results ship in `src/infer_with_fiona/quadrature/schedule_cis.py` and `data/bgo_switch/`. Their outputs go to `outputs/checks/`.

```bash
python scripts/2_convergence/derive_quadrature_schedule.py --n-ref 1024000 --w-high-ext   # node doubling over 66 lenses
python scripts/2_convergence/derive_bgo_switch_cis.py --fine --out bgo_switch_cis         # w_s(x_c, y) for CIS
python scripts/2_convergence/derive_bgo_switch_ula_pbh.py --lens ULA --gl-mult 4          # ULA soliton
python scripts/2_convergence/derive_bgo_switch_ula_pbh.py --lens PointLens --gl-mult 4    # PBH point lens
python scripts/2_convergence/check_point_lens_exact.py                                    # FIONA vs the exact point-mass F
```

The schedules are selected by name: `cis` for the CIS lens, and `steep_core`, four times the `cis` node count, for the ULA and PBH lenses.

### 3. Interpolation table

```bash
python scripts/3_table/build_table.py               # CIS table for the paper's prior, into outputs/table_cache/
python scripts/3_table/validate_table_cis.py --stages B --go-residual 1.0 --switch data/bgo_switch/cis.npz \
    --w-max 253.7 --w-split 100.5 --dw-high 0.05 --table-cache outputs/table_cache
python scripts/3_table/validate_table_ula_pbh.py --lens ULA      # also --lens PBH; builds those tables too
```

The CIS table covers 201 × 201 × 8452 nodes in $(x_c, y, w)$. It takes about 46 minutes to build on one 112-core node and occupies about 5 GB. Each inference run loads it from the cache, or builds it on first use, and compares it with direct FIONA at the injected parameters before sampling starts.

### 4. Bayesian inference

The runs were made on 112-core nodes with about 250 GB of memory, and each took 13 to 26 hours. The Slurm scripts record the exact resources and arguments. Submit them from the repository root, and set `PYTHON` if your interpreter is not the default `python`.

The four CIS benchmarks, with the log-uniform $M_{Lz}$ prior over $[0.1, 2000]\,M_\odot$ used in the paper:

```bash
#                                  x_c   y    M_Lz grid nlive prior range  (unused) prior type
sbatch slurm/run_inference_cis.sh 0.02 0.3 100 high 1000 0.1 2000.0 "" log-uniform
sbatch slurm/run_inference_cis.sh 0.2  1.0 100 high 1000 0.1 2000.0 "" log-uniform
sbatch slurm/run_inference_cis.sh 0.2  2.0 100 high 1000 0.1 2000.0 "" log-uniform
sbatch slurm/run_inference_cis.sh 0.2  3.0 100 high 1000 0.1 2000.0 "" log-uniform
```

The prior type must be given explicitly, because the CIS script defaults to a uniform $M_{Lz}$ prior, as in C24. The strongly lensed run at $x_c = 0.02$, $y = 0.3$ is also the SIDM posterior.

The ULA and PBH lenses, which default to the log-uniform prior:

```bash
sbatch slurm/run_inference_ula_pbh.sh ULA 0.3 0.1 100    # r_s = 0.1, y = 0.3, M_Lz = 100 Msun
sbatch slurm/run_inference_ula_pbh.sh PBH 0.3 0.1 100    # y = 0.3, M_Lz = 100 Msun (no shape parameter)
```

Results go to `outputs/runs/`. Runs checkpoint automatically, and resubmitting the same command resumes them. The drivers can also be run directly; see `python scripts/4_inference/run_inference_cis.py --help`.

### 5. Figures

| Paper figure | Script |
|---|---|
| Fisher forecasts for LISA and LIGO | `scripts/1_fisher/fisher_forecasts.py plot` |
| Fisher vs inference width of the core size | `notebooks/04_core_size_calibration.ipynb` |
| CIS lens-parameter posteriors, this pipeline vs `glworia` | `scripts/5_figures/plot_cis_lens_corners.py` |
| PBH and ULA lens-parameter posteriors | `scripts/5_figures/plot_ula_pbh_corners.py` |
| Full 14-parameter CIS posteriors (appendix) | `scripts/5_figures/plot_cis_full_corners.py` |
| SIDM cross-section posteriors | `scripts/5_figures/plot_sidm_cross_section.py` |

By default, the figure scripts read the shipped samples in `results/posteriors/` and write to `outputs/figures/`. With `--from-runs`, they read fresh `bilby` output from `outputs/runs/` instead, and glworia output from `--glworia-dir`.

### 6. Comparison with glworia

`glworia_comparison/README.md` describes how the reference runs were made: the glworia commit, the patch applied to it, the four run configurations and the Slurm script.

## Posterior samples

`results/posteriors/` contains the equal-weight posterior samples of every inference run, as compressed NumPy archives.

- **`fiona/`** holds the four CIS benchmarks and the mass scans: CIS, ULA and PBH lenses at $M_{Lz} = 3, 10, 30, 100, 300$ and $1000\,M_\odot$, all with $y = 0.3$ and the same noise realization.
- **`glworia/`** holds the four CIS benchmarks from `glworia`, run on identical data.

`results/run_summary.json` lists each run's sampling time, number of likelihood evaluations and evidence.

```python
from infer_with_fiona import posteriors

samples, info = posteriors.load_fiona(posteriors.fiona_label('CIS', y=0.3, shape=0.02))
print(info['status'], samples['MLz'].mean(), samples['y'].mean(), samples['lp'].mean())
```

The sampled lens-shape parameter is called `lp`: $x_c$ for CIS and $r_s$ for ULA.

### Run labels

Run labels keep the tags used during the work, so they match the shipped files.

- **`wtab`** means $F$ came from the interpolation table.
- **`v4`** is the `cis` quadrature schedule, and **`v4x4`** is `steep_core`.
- **`lal`** means $F^*$ was applied to LAL waveforms, the correct convention. Every result in the paper uses it.
- **`glworia_repro`** marks runs whose data, priors and sampler settings match `glworia`'s released benchmarks.

## Citation

If you use this code, please cite the paper above and the FIONA paper:

```bibtex
@misc{FIONA,
  author        = {Ephremidze, Nino and Kamionkowski, Marc and Dvorkin, Cora},
  title         = {Fast {F}ourier transform evaluation of the {F}resnel integral for gravitational-wave lensing},
  year          = {2026},
  eprint        = {2603.12333},
  archivePrefix = {arXiv}
}
```

The Bayesian pipeline reproduces the methodology of C24 and the released `glworia` code. The Fisher forecasts follow Tambalo et al. (2023). The NUFHT algorithm is from Beckman and O'Neil ([arXiv:2411.09583](https://arxiv.org/abs/2411.09583)).
