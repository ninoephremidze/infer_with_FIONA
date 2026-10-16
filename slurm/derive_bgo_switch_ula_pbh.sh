#!/bin/bash
# Slurm resources below are those used on Harvard FASRC (112-core sapphire nodes); adjust for your cluster.
#SBATCH -n 112
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -p sapphire,dvorkin
#SBATCH --mem=200000
#SBATCH -o outputs/logs/%x_%j.out
#SBATCH -e outputs/logs/%x_%j.err
# bGO switch + GL v4 check for the ULA soliton or point lens (scripts/2_convergence/derive_bgo_switch_ula_pbh.py).
# Usage: sbatch slurm/derive_bgo_switch_ula_pbh.sh --lens ULA|PointLens [--quick]
# Smoke test: sbatch -p test -t 01:00:00 slurm/derive_bgo_switch_ula_pbh.sh --lens ULA --quick
# Run from the repository root (or set INFER_WITH_FIONA_ROOT); PYTHON selects the interpreter.
cd "${INFER_WITH_FIONA_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON=${PYTHON:-python}
$PYTHON -u scripts/2_convergence/derive_bgo_switch_ula_pbh.py "$@"
