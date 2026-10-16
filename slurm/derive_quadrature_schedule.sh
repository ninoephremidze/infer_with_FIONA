#!/bin/bash
# Slurm resources below are those used on Harvard FASRC (112-core sapphire nodes); adjust for your cluster.
#SBATCH -n 112
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -p dvorkin,shared,sapphire
#SBATCH --mem=128000
#SBATCH -o outputs/logs/%x_%j.out
#SBATCH -e outputs/logs/%x_%j.err
# Re-derives the FIONA GL schedule by convergence against a high-n reference, plus bGO
# accuracy (scripts/2_convergence/derive_quadrature_schedule.py).
# Usage: sbatch slurm/derive_quadrature_schedule.sh [extra args]
# Smoke test: sbatch -p test -t 01:00:00 slurm/derive_quadrature_schedule.sh --quick
# Run from the repository root (or set INFER_WITH_FIONA_ROOT); PYTHON selects the interpreter.
cd "${INFER_WITH_FIONA_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON=${PYTHON:-python}
$PYTHON -u scripts/2_convergence/derive_quadrature_schedule.py "$@"
