# GW-lensing inference with FIONA

Code and data products for the paper

> **The power of gravitational-wave lensing to constrain dark matter: accelerated inference with FIONA**
> Nino Ephremidze, Cora Dvorkin, and Marc Kamionkowski

This repository contains the final code used to produce every Fisher-forecast and Bayesian-inference result in the paper. It uses [FIONA](https://github.com/ninoephremidze/FIONA) ([arXiv:2603.12333](https://arxiv.org/abs/2603.12333)), a wave-optics solver that evaluates the gravitational-wave lensing amplification factor $F(w, y)$ for a whole batch of source positions and lens shapes with a single non-uniform fast Hankel transform.

## What the package does

Gravitational waves passing a compact dark-matter lens pick up a frequency-dependent amplification factor $F(w, y, l)$. Here $w = 8\pi G M_{Lz} f / c^3$ is the dimensionless frequency, $y$ is the source position, and $l$ is a lens-shape parameter. This repository provides two analyses built on FIONA.

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

On the four CIS benchmarks of Cheung et al. (2024), this pipeline and the released [`glworia`](https://github.com/mhycheung/glworia) code give posteriors whose medians agree to within $0.06\sigma$ in every parameter, on identical data. This pipeline completes each run 2.3 to 2.7 times faster.

## Repository layout

All Python files sit at the repository root and are run from there. The scripts import one another and load their data files by relative path.

| Path | Contents |
|---|---|
| `fiona/` | The FIONA solver, in the exact version used for the paper. |
| `fisherWithFIONA/` | Fisher-matrix tools: waveforms, noise curves, finite-difference derivatives of $F$, and geometric-optics image solvers. |
| `fiona_adaptive_gl_v4.py` | The convergence-tested quadrature schedule $n(w)$ for CIS lenses. |
| `fiona_adaptive_gl_v4x4.py` | The same schedule with four times the nodes, used for the steeper ULA and PBH cores. |
| `fiona_adaptive_gl_v3.py`, `fiona_adaptive_gl_v4x2.py` | Helpers: the shared node-cache writer, and an intermediate schedule used in the PBH convergence check. |
| `fiona_w_table.py` | Builds and interpolates the $F(x_c, y, w)$ table, with the GO residual, the bGO switch and the caustic prior mask. |
| `fiona_pe_F.py` | `HybridF`, the amplification factor used inside the likelihood. It caches the table on disk and in shared memory, and applies the Fourier and time conventions. |
| `PE_CIS_glworia_reproduction.py` | Bayesian inference for the CIS lens, which also gives the SIDM result. |
| `PE_DM_lens.py` | Bayesian inference for the ULA soliton and PBH lenses. |
| `build_F_table.py` | Builds the interpolation table ahead of a set of runs. |
| `regen_fisher_figures.py` | LISA and LIGO Fisher forecasts. |
| `derive_gl_schedule_v4.py` | Derives the quadrature schedule by node doubling. |
| `derive_bgo_switch.py`, `derive_bgo_switch_lens.py` | Measure the bGO switch frequency $w_s(x_c, y)$ for CIS, ULA and PBH. |
| `validate_w_table.py`, `validate_w_table_lens.py` | Validate the table against direct FIONA evaluations. |
| `check_*.py` | Checks of the Fourier convention, the time reference, and FIONA against the exact point-mass $F$. |
| `dm_conversions.py` | Converts lens parameters to dark-matter parameters: the SIDM cross section and the axion mass. |
| `plot_*.py`, `extract_all_posteriors.py` | Paper figures and posterior extraction. |
| `bgo_switch_*.npz` | The measured bGO switch maps for CIS, ULA and PBH. |
| `fisher_fig12_cov.npz` | Cached Fisher covariances behind the two Fisher figures. |
| `fiona_gl_cache/` | Precomputed Gauss-Legendre rules. The largest took about 8 hours to generate. |
| `glworia_repo/interpolation/crit_funcs/` | `glworia`'s tabulated caustic curve $y_{\rm crit}(x_c)$, which defines the CIS prior mask. |
| `glworia_comparison/` | The changes made to `glworia` for the comparison, and its run configurations. |
| `results/posteriors/` | Posterior samples of every run in the paper. |
| `slurm/` | The Slurm submission scripts used on the Harvard FASRC cluster. |

### Naming conventions

Run labels and file names carry a few internal tags.

- **`v4`** is the convergence-tested quadrature schedule described above. **`v4x4`** is the same schedule with four times the nodes.
- **`wtab`** means $F$ came from the interpolation table.
- **`lal`** means $F^*$ was applied to LAL waveforms, which is the correct convention. Every result in the paper uses it.
- **`glworia_repro`** marks runs whose data, priors and sampler settings match `glworia`'s released benchmarks.
- **`lp`** is the sampled lens-shape parameter: $x_c$ for CIS and $r_s$ for ULA.

## Installation

The paper's results were produced with this environment.

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

`requirements.txt` lists the full set.

```bash
git clone https://github.com/ninoephremidze/infer_with_FIONA.git
cd infer_with_FIONA
conda create -n gwlens python=3.11
conda activate gwlens
pip install -r requirements.txt
```

FIONA's axisymmetric solver also needs the compiled batched Hankel-transform extension `_pynufht`. Build it from the [NUFHT](https://github.com/marckamion/NUFHT) repository, following its README:

```bash
git clone https://github.com/marckamion/NUFHT.git
cd NUFHT/python
python setup.py build_ext --inplace
```

Then put the resulting `_pynufht` module on your `PYTHONPATH`, or install it into the environment. Always `import finufft` before importing `fiona`, as the scripts do.

The included `fiona/` folder is the version of FIONA used for the paper. It adds the ULA soliton lens and the batched multi-lens solver that this pipeline needs, which are not in the currently released FIONA.

## Reproducing the results

The Bayesian runs were made on 112-core nodes with about 250 GB of memory, and each took 13 to 26 hours. The scripts in `slurm/` record the exact resources and arguments. They contain absolute paths to the original cluster directory and Python environment, so edit the `cd` and Python lines before using them elsewhere.

### 1. Fisher forecasts

```bash
python regen_fisher_figures.py compute                     # Fisher runs -> fisher_fig12_cov.npz
mkdir -p figures
python regen_fisher_figures.py plot --out-dir figures      # LISA_fisher.png, LIGO_fisher.png
```

The shipped `fisher_fig12_cov.npz` lets you run `plot` without recomputing. LISA forecasts are normalized to SNR $10^3$ and LIGO forecasts to SNR $10^2$.

### 2. Quadrature schedule and bGO switch

These steps are optional, because their outputs are already in the repository.

```bash
python derive_gl_schedule_v4.py --n-ref 1024000 --w-high-ext   # node-doubling convergence test
python derive_bgo_switch.py --fine --out bgo_switch_fine      # w_s(x_c, y) for CIS
python derive_bgo_switch_lens.py --lens ULA --gl-mult 4        # ULA soliton
python derive_bgo_switch_lens.py --lens PointLens --gl-mult 4  # PBH point lens
python check_pbh_vs_exact.py                                   # FIONA vs the exact point-mass F
```

The resulting schedule is encoded in `fiona_adaptive_gl_v4.py`. The switch maps are the `bgo_switch_*.npz` files.

### 3. Interpolation table

```bash
python build_F_table.py --m-lz-prior-min 0.1 --m-lz-prior-max 2000
python validate_w_table.py          # CIS: table vs direct FIONA at test lenses across the prior
python validate_w_table_lens.py --lens ULA   # also --lens PBH
```

The CIS table covers 201 × 201 × 8452 nodes in $(x_c, y, w)$. It takes about 46 minutes to build on one 112-core node and occupies about 5 GB in `fiona_w_table_cache/`. The cache is keyed by a hash of the table configuration, so every run with the same configuration reuses it. Each inference run also compares the table with direct FIONA at its injected parameters before sampling starts.

### 4. Bayesian inference

The four CIS benchmarks, with the log-uniform $M_{Lz}$ prior over $[0.1, 2000]\,M_\odot$ used in the paper:

```bash
#                                               x_c   y    M_Lz grid nlive prior range  (unused) prior type
sbatch slurm/submit_PE_CIS_glworia_reproduction.sh 0.02 0.3 100 high 1000 0.1 2000.0 "" log-uniform
sbatch slurm/submit_PE_CIS_glworia_reproduction.sh 0.2  1.0 100 high 1000 0.1 2000.0 "" log-uniform
sbatch slurm/submit_PE_CIS_glworia_reproduction.sh 0.2  2.0 100 high 1000 0.1 2000.0 "" log-uniform
sbatch slurm/submit_PE_CIS_glworia_reproduction.sh 0.2  3.0 100 high 1000 0.1 2000.0 "" log-uniform
```

The prior type must be given explicitly, because this script defaults to a uniform $M_{Lz}$ prior. The strongly lensed run at $x_c = 0.02$, $y = 0.3$ is also the SIDM posterior. Its core radius is converted to a cross section with `dm_conversions.py` and `plot_sidm_cross_section_posteriors.py`.

The ULA and PBH lenses:

```bash
sbatch slurm/submit_PE_DM_lens.sh ULA 0.3 0.1 100    # r_s = 0.1, y = 0.3, M_Lz = 100 Msun
sbatch slurm/submit_PE_DM_lens.sh PBH 0.3 0.1 100    # y = 0.3, M_Lz = 100 Msun (no shape parameter)
```

`PE_DM_lens.py` defaults to the log-uniform prior. All runs checkpoint automatically, and resubmitting the same command resumes them.

### 5. Comparison with glworia

The comparison runs used `glworia` at the commit in `glworia_comparison/glworia_base_commit.txt`, with the changes in `glworia_local_changes.patch`. The main change applies $F^*$ to the LAL waveforms, matching this pipeline. The patch also sets four image-finding parameters that are missing from the released interpolation settings, and updates one JAX import.

```bash
git clone https://github.com/mhycheung/glworia.git
cd glworia
git checkout $(cat ../glworia_comparison/glworia_base_commit.txt)
git apply ../glworia_comparison/glworia_local_changes.patch
cp ../glworia_comparison/configs/*.json PE_submission_settings/CIS/
```

Build `glworia`'s CIS interpolation table with its own scripts, then run `scripts/PE_full.py` on each configuration, as in `slurm/submit_glworia_PE_full.sh`.

### 6. Figures

| Paper figure | Script |
|---|---|
| Fisher forecasts for LISA and LIGO | `regen_fisher_figures.py plot` |
| CIS lens-parameter posteriors, this pipeline vs `glworia` | `plot_final_cis_lensing_corners.py` |
| PBH and ULA lens-parameter posteriors | `plot_provisional_dm_lens.py` |
| Full 14-parameter CIS posteriors (appendix) | `plot_full_corners_appendix.py` |

The plotting scripts read the full `bilby` result files under `full_parameter_inference/` and `glworia_repo/outdir/`, which the inference runs write. These files are 35 to 55 MB each, so they are not included here. Their equal-weight posterior samples are included instead.

## Posterior samples

`results/posteriors/` contains the posterior samples of every run in the paper, as compressed NumPy archives.

- **`fiona/`** holds the four CIS benchmarks, the ULA run and the PBH run from this pipeline.
- **`glworia/`** holds the four CIS benchmarks from `glworia`, run on identical data.

Each file holds one array per sampled parameter and a JSON string `info` with the run status and log evidence.

```python
import json
import numpy as np

d = np.load('results/posteriors/fiona/CIS_glworia_repro_mlz100_xc0.02_y0.3_high_mlzprior0.1-2000.0_log-uniform_wtabv4lal_0.npz')
print(json.loads(str(d['info'])))
print(np.median(d['MLz']), np.median(d['y']), np.median(d['lp']))
```

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

The Bayesian pipeline reproduces the methodology of Cheung et al. (2024) and the released `glworia` code. The Fisher forecasts follow Tambalo et al. (2023). The NUFHT algorithm is from Beckman and O'Neil ([arXiv:2411.09583](https://arxiv.org/abs/2411.09583)).
