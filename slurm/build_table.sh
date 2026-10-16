#!/bin/bash
# Slurm resources below are those used on Harvard FASRC (112-core sapphire nodes); adjust for your cluster.
#SBATCH -n 112
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -p sapphire,dvorkin
#SBATCH --mem=400000
#SBATCH -o outputs/logs/%x_%j.out
#SBATCH -e outputs/logs/%x_%j.err
# Builds the cached F table for a given M_Lz prior (see scripts/3_table/build_table.py).
# Usage: sbatch slurm/build_table.sh <m_lz_prior_min> <m_lz_prior_max> [grid_res]
# Run from the repository root (or set INFER_WITH_FIONA_ROOT); PYTHON selects the interpreter.
cd "${INFER_WITH_FIONA_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON=${PYTHON:-python}
$PYTHON -u scripts/3_table/build_table.py \
    --m-lz-prior-min "${1:?min}" --m-lz-prior-max "${2:?max}" --grid-res "${3:-high}"
