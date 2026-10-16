#!/bin/bash
# Slurm resources below are those used on Harvard FASRC (112-core sapphire nodes); adjust for your cluster.
#SBATCH -n 112
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH -p sapphire,dvorkin
#SBATCH --mem=400000
#SBATCH -o outputs/logs/%x_%j.out
#SBATCH -e outputs/logs/%x_%j.err
# Builds + caches the production F table for ULA or PBH and validates it against direct FIONA
# (scripts/3_table/validate_table_ula_pbh.py). Usage: sbatch slurm/validate_table_ula_pbh.sh --lens ULA|PBH [args]
# Run from the repository root (or set INFER_WITH_FIONA_ROOT); PYTHON selects the interpreter.
cd "${INFER_WITH_FIONA_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON=${PYTHON:-python}
$PYTHON -u scripts/3_table/validate_table_ula_pbh.py "$@"
