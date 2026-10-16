#!/bin/bash
# Slurm resources below are those used on Harvard FASRC (112-core sapphire nodes); adjust for your cluster.
#SBATCH -n 24
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -p test
#SBATCH --mem=400000
#SBATCH -o outputs/logs/%x_%j.out
#SBATCH -e outputs/logs/%x_%j.err
# Cache posteriors of all FIONA lal runs (final or checkpoint) for the 0-all_*_results notebooks.
# Run from the repository root (or set INFER_WITH_FIONA_ROOT); PYTHON selects the interpreter.
cd "${INFER_WITH_FIONA_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON=${PYTHON:-python}
$PYTHON -u scripts/4_inference/extract_posteriors.py "$@"
