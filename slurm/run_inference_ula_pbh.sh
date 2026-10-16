#!/bin/bash
# Slurm resources below are those used on Harvard FASRC (112-core sapphire nodes); adjust for your cluster.
#SBATCH -n 112        # Number of cores (single node)
#SBATCH -N 1
#SBATCH -t 3-00:00    # Checkpoints automatically; resubmit the same command to resume.
#SBATCH -p sapphire,dvorkin
#SBATCH --mem=250000  # MB, as the CIS runs (table <= ~5 GB in shared memory + dynesty state)
#SBATCH -o outputs/logs/%x_%j.out
#SBATCH -e outputs/logs/%x_%j.err
#SBATCH --array=0

# PE for the ULA soliton or PBH point lens (scripts/4_inference/run_inference_ula_pbh.py): same pipeline, injection, noise
# and priors as the corrected CIS runs (F* in LAL's convention, minimum-image time reference,
# log-uniform M_Lz in [0.1, 2000], cached F table with GL v4x4 and a lens-specific bGO switch).
#
# Usage: sbatch slurm/run_inference_ula_pbh.sh <ULA|PBH> [y_fid] [shape_fid] [m_lz_fid] [grid_res] [nlive]
#   ULA defaults: y 0.3, r_s 0.1, M_Lz 100 (the earlier ULA benchmark); PBH: y 0.3, M_Lz 100.
# Smoke test: sbatch -p test -t 02:00:00 slurm/run_inference_ula_pbh.sh ULA 0.3 0.1 100 high 100
#
# "wtabv4x4lal" in the label: GL v4x4, F in LAL's convention. Resubmitting the same command
# resumes from the checkpoint.

USAGE="Usage: sbatch slurm/run_inference_ula_pbh.sh <ULA|PBH> [y_fid] [shape_fid] [m_lz_fid] [grid_res] [nlive]"
LENS=${1:?"$USAGE"}
Y_FID=${2:-0.3}
SHAPE_FID=${3:-0.1}
M_LZ_FID=${4:-100}
GRID_RES=${5:-high}
NLIVE=${6:-1000}

# Run from the repository root (or set INFER_WITH_FIONA_ROOT); PYTHON selects the interpreter.
cd "${INFER_WITH_FIONA_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON=${PYTHON:-python}
if [ "$LENS" = "ULA" ]; then SHAPE_TAG="_rs${SHAPE_FID}"; else SHAPE_TAG=""; fi
echo "lens=${LENS} y_fid=${Y_FID} shape_fid=${SHAPE_FID} m_lz_fid=${M_LZ_FID} grid_res=${GRID_RES} nlive=${NLIVE}"

$PYTHON scripts/4_inference/run_inference_ula_pbh.py \
  --lens ${LENS} \
  --m-lz-fid ${M_LZ_FID} \
  --shape-fid ${SHAPE_FID} \
  --y-fid ${Y_FID} \
  --grid-res ${GRID_RES} \
  --nlive ${NLIVE} \
  --n-workers 112 \
  ${PE_EXTRA_ARGS} \
  --out-label "${LENS}_glworia_repro_mlz${M_LZ_FID}${SHAPE_TAG}_y${Y_FID}_${GRID_RES}_mlzprior0.1-2000.0_log-uniform_wtabv4x4lal${PE_LABEL_TAG:+_${PE_LABEL_TAG}}_${SLURM_ARRAY_TASK_ID}"
