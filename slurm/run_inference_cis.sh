#!/bin/bash
# Slurm resources below are those used on Harvard FASRC (112-core sapphire nodes); adjust for your cluster.
#SBATCH -n 112        # Number of cores (single node)
#SBATCH -N 1          # one node, so bilby/dynesty's npool actually gets 112 real cores
#SBATCH -t 3-00:00    # Runtime in D-HH:MM. Checkpoints automatically; resubmit to resume.
#SBATCH -p sapphire,dvorkin  # 112-core, 1 TB nodes (shared's 48-core, 188 GB nodes cannot fit these jobs)
#SBATCH --mem=250000  # MB. F table ~5 GB (grid_res high, in shared memory, counted once) +
                      # dynesty/bilby state, which grows with iterations: glworia's own runs
                      # with identical sampler settings peaked at 47-73 GB (MaxRSS) over
                      # 44k-56k iterations. The old 4D table needed 440000 and still OOM'd.
#SBATCH -o outputs/logs/%x_%j.out
#SBATCH -e outputs/logs/%x_%j.err
#SBATCH --array=0

# Submission script for scripts/4_inference/run_inference_cis.py -- a faithful reproduction of
# Cheung+2024's ("glworia") actual released CIS pipeline (bilby detectors/noise/priors/
# likelihood/sampler), using FIONA for the lensing amplification factor F(w,y,x_c).
#
# F(w, y, x_c): the cached (x_c, y, w) FIONA table + GO residual + lens-dependent bGO switch
# (src/infer_with_fiona/amplification.py), validated to <= 1.02% of direct FIONA outside the prior's caustic mask.
# With the default grid_res=high and M_Lz prior [0.1, 2000] it LOADS the cached table
# (outputs/table_cache/07cd8bb159006ece, seconds). Any other grid_res or m_lz_prior_max
# builds and caches a new table first (~46 min at high on 112 cores; much less for smaller
# grids or a narrower M_Lz range).
#
# Usage: sbatch slurm/run_inference_cis.sh <xc_fid> <y_fid> [m_lz_fid] [grid_res] [nlive] [m_lz_prior_min] [m_lz_prior_max] [n_logm] [m_lz_prior_type]
#   xc_fid:    fiducial (injected) CIS core radius x_c (required). Cheung+2024 use 0.02 for
#              their strong-lensing benchmark, 0.2 for weak-lensing.
#   y_fid:     fiducial (injected) source position y (required). Cheung+2024 use 0.3 for
#              strong-lensing, 1/2/3 for weak-lensing.
#   m_lz_fid:  fiducial M_Lz [Msun] (default: 100, matching every Cheung+2024 CIS config).
#   grid_res:  tiny | low | medium | high (default: high = 201 x 201, the validated and
#              cached configuration; the others are for smoke tests and are not validated).
#   nlive:     dynesty nlive via bilby.run_sampler (default: 1000, matching Cheung+2024).
#   m_lz_prior_min, m_lz_prior_max: M_Lz prior bounds [Msun] (default: 0.1, 2000.0,
#              Cheung+2024's own full range).
#   n_logm:    DEPRECATED, ignored (the F table has no M_Lz axis); kept so old commands work.
#   m_lz_prior_type: uniform (default) | log-uniform. 'uniform' matches Cheung+2024.
#
# Examples:
#   sbatch -p test -t 02:00:00 slurm/run_inference_cis.sh 0.02 0.3 100 high 100   # smoke test (cached table)
#   sbatch slurm/run_inference_cis.sh 0.02 0.3        # strong-lensing benchmark
#   sbatch slurm/run_inference_cis.sh 0.2  1.0        # weak-lensing y=1 benchmark (also y=2, y=3)
#
# Optional environment variables: PE_EXTRA_ARGS (extra scripts/4_inference/run_inference_cis.py
# arguments, e.g. "--narrow-priors") and PE_LABEL_TAG (appended to the output label), e.g.
#   sbatch --export=ALL,PE_EXTRA_ARGS=--narrow-priors,PE_LABEL_TAG=narrow slurm/run_inference_cis.sh 0.02 0.3 100 high 1000 50 150
#
# "wtabv4lal" in the label (2026-10-07): F in LAL's Fourier convention, conj(F) -- see
# amplification.py, FOURIER CONVENTION. Runs labelled "wtabv4tmin" used e^{+iwT} (images in the
# wrong time order) and must NOT be resumed into these.
# "wtabv4tmin": F with glworia's time reference (minimum image), introduced
# 2026-09-28. Runs labelled "wtabv4" used FIONA's absolute time reference and must NOT be
# resumed into these (the label change makes each resubmit start its own new run).
#
# If this job is killed by walltime/preemption/node failure, just resubmit the SAME
# command -- bilby's own dynesty wrapper checkpoints and resumes automatically.

USAGE="Usage: sbatch slurm/run_inference_cis.sh <xc_fid> <y_fid> [m_lz_fid] [grid_res] [nlive] [m_lz_prior_min] [m_lz_prior_max] [n_logm] [m_lz_prior_type]"
XC_FID=${1:?"$USAGE"}
Y_FID=${2:?"$USAGE"}
M_LZ_FID=${3:-100}
GRID_RES=${4:-high}
NLIVE=${5:-1000}
M_LZ_PRIOR_MIN=${6:-0.1}
M_LZ_PRIOR_MAX=${7:-2000.0}
N_LOGM=${8:-}            # deprecated, ignored
M_LZ_PRIOR_TYPE=${9:-uniform}

# Run from the repository root (or set INFER_WITH_FIONA_ROOT); PYTHON selects the interpreter.
cd "${INFER_WITH_FIONA_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON=${PYTHON:-python}

echo "xc_fid=${XC_FID}  y_fid=${Y_FID}  m_lz_fid=${M_LZ_FID}  grid_res=${GRID_RES}  nlive=${NLIVE}  m_lz_prior=[${M_LZ_PRIOR_MIN},${M_LZ_PRIOR_MAX}] (${M_LZ_PRIOR_TYPE})"
[ -n "${N_LOGM}" ] && echo "note: n_logm=${N_LOGM} given but ignored (the F table has no M_Lz axis)"

# Call the env's python by absolute path: `module load` / `conda activate` are not needed.
$PYTHON scripts/4_inference/run_inference_cis.py \
  --m-lz-fid ${M_LZ_FID} \
  --xc-fid ${XC_FID} \
  --y-fid ${Y_FID} \
  --grid-res ${GRID_RES} \
  --nlive ${NLIVE} \
  --m-lz-prior-min ${M_LZ_PRIOR_MIN} \
  --m-lz-prior-max ${M_LZ_PRIOR_MAX} \
  --m-lz-prior-type ${M_LZ_PRIOR_TYPE} \
  --n-workers 112 \
  ${PE_EXTRA_ARGS} \
  --out-label "CIS_glworia_repro_mlz${M_LZ_FID}_xc${XC_FID}_y${Y_FID}_${GRID_RES}_mlzprior${M_LZ_PRIOR_MIN}-${M_LZ_PRIOR_MAX}_${M_LZ_PRIOR_TYPE}_wtabv4lal${PE_LABEL_TAG:+_${PE_LABEL_TAG}}_${SLURM_ARRAY_TASK_ID}"
