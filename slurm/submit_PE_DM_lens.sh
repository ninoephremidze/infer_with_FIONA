#!/bin/bash
#SBATCH -n 112        # Number of cores (single node)
#SBATCH -N 1
#SBATCH -t 3-00:00    # Checkpoints automatically; resubmit the same command to resume.
#SBATCH -p sapphire,dvorkin
#SBATCH --mem=250000  # MB, as the CIS runs (table <= ~5 GB in shared memory + dynesty state)
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nino_ephremidze@college.harvard.edu
#SBATCH --array=0

# PE for the ULA soliton or PBH point lens (PE_DM_lens.py): same pipeline, injection, noise
# and priors as the corrected CIS runs (F* in LAL's convention, minimum-image time reference,
# log-uniform M_Lz in [0.1, 2000], cached F table with GL v4x4 and a lens-specific bGO switch).
#
# Usage: sbatch submit_PE_DM_lens.sh <ULA|PBH> [y_fid] [shape_fid] [m_lz_fid] [grid_res] [nlive]
#   ULA defaults: y 0.3, r_s 0.1, M_Lz 100 (the earlier ULA benchmark); PBH: y 0.3, M_Lz 100.
# Smoke test: sbatch -p test -t 02:00:00 submit_PE_DM_lens.sh ULA 0.3 0.1 100 high 100
#
# "wtabv4x4lal" in the label: GL v4x4, F in LAL's convention. Resubmitting the same command
# resumes from the checkpoint.

USAGE="Usage: sbatch submit_PE_DM_lens.sh <ULA|PBH> [y_fid] [shape_fid] [m_lz_fid] [grid_res] [nlive]"
LENS=${1:?"$USAGE"}
Y_FID=${2:-0.3}
SHAPE_FID=${3:-0.1}
M_LZ_FID=${4:-100}
GRID_RES=${5:-high}
NLIVE=${6:-1000}

cd /n/netscratch/dvorkin_lab/Lab/nephremidze/2-LISA/0-dark_matter_inference
if [ "$LENS" = "ULA" ]; then SHAPE_TAG="_rs${SHAPE_FID}"; else SHAPE_TAG=""; fi
echo "lens=${LENS} y_fid=${Y_FID} shape_fid=${SHAPE_FID} m_lz_fid=${M_LZ_FID} grid_res=${GRID_RES} nlive=${NLIVE}"

/n/home02/nephremidze/.conda/envs/glow311/bin/python PE_DM_lens.py \
  --lens ${LENS} \
  --m-lz-fid ${M_LZ_FID} \
  --shape-fid ${SHAPE_FID} \
  --y-fid ${Y_FID} \
  --grid-res ${GRID_RES} \
  --nlive ${NLIVE} \
  --n-workers 112 \
  ${PE_EXTRA_ARGS} \
  --out-label "${LENS}_glworia_repro_mlz${M_LZ_FID}${SHAPE_TAG}_y${Y_FID}_${GRID_RES}_mlzprior0.1-2000.0_log-uniform_wtabv4x4lal${PE_LABEL_TAG:+_${PE_LABEL_TAG}}_${SLURM_ARRAY_TASK_ID}"
