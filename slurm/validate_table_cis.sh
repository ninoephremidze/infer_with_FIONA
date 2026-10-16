#!/bin/bash
# Slurm resources below are those used on Harvard FASRC (112-core sapphire nodes); adjust for your cluster.
#SBATCH -n 112
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -p dvorkin,shared,sapphire
#SBATCH --mem=64000
#SBATCH -o outputs/logs/%x_%j.out
#SBATCH -e outputs/logs/%x_%j.err
# Validates table.py (F(w,x_c,y) table) against direct FIONA over the 200-combo
# v3 grid. Stage B's 201x201 table is ~3.5 GiB, so 64 GB is ample.
# Usage: sbatch slurm/validate_table_cis.sh [extra args for scripts/3_table/validate_table_cis.py]
# Smoke test: sbatch -p test -t 01:00:00 slurm/validate_table_cis.sh --quick
# Run from the repository root (or set INFER_WITH_FIONA_ROOT); PYTHON selects the interpreter.
cd "${INFER_WITH_FIONA_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON=${PYTHON:-python}
$PYTHON -u scripts/3_table/validate_table_cis.py "$@"
