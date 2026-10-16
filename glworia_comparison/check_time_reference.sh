#!/bin/bash
# Slurm resources below are those used on Harvard FASRC (112-core sapphire nodes); adjust for your cluster.
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -p test
#SBATCH --mem=64000
#SBATCH -o outputs/logs/%x_%j.out
#SBATCH -e outputs/logs/%x_%j.err
# FIONA F (min-image time reference) vs glworia F, complex (glworia_comparison/check_time_reference.py)
# Run from the repository root (or set INFER_WITH_FIONA_ROOT); PYTHON selects the interpreter.
cd "${INFER_WITH_FIONA_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON=${PYTHON:-python}
export GLWORIA_PREFIX=${GLWORIA_PREFIX:?set GLWORIA_PREFIX to a patched glworia checkout}
$PYTHON -u glworia_comparison/check_time_reference.py
