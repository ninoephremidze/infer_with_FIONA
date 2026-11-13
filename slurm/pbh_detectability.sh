#!/bin/bash
# Point-lens detectability grid for the LIGO PE benchmark (scripts/1_fisher/pbh_detectability.py).
# Resource lines are FASRC-specific. About 10 minutes on 112 cores.
#SBATCH -p dvorkin
#SBATCH -n 112
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH --mem=50000
#SBATCH -o outputs/logs/pbh_detectability_%j.out
cd "${INFER_WITH_FIONA_ROOT:-$SLURM_SUBMIT_DIR}"
${PYTHON:-python} -u scripts/1_fisher/pbh_detectability.py 2>&1 | grep --line-buffered -v "Loaded tables"
