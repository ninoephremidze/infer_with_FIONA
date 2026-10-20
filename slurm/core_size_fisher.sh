#!/bin/bash
# Fisher grid of the CIS core uncertainty for the LIGO PE benchmark (scripts/1_fisher/core_size_fisher.py).
# Resource lines are FASRC-specific. About 15 minutes on 112 cores.
#SBATCH -p test
#SBATCH -n 112
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --mem=200000
#SBATCH -o outputs/logs/core_size_fisher_%j.out
cd "${INFER_WITH_FIONA_ROOT:-$SLURM_SUBMIT_DIR}"
${PYTHON:-python} -u scripts/1_fisher/core_size_fisher.py "${@:-LIGO}"
