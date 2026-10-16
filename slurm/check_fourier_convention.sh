#!/bin/bash
# Slurm resources below are those used on Harvard FASRC (112-core sapphire nodes); adjust for your cluster.
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -p test
#SBATCH --mem=64000
#SBATCH -o outputs/logs/%x_%j.out
#SBATCH -e outputs/logs/%x_%j.err
# scripts/3_table/check_fourier_convention.py --full: the Fourier-convention test with the production HybridF
# (cached F table, ~5 GB), plus the PE self-check of HybridF against direct FIONA in the new convention.
# Run from the repository root (or set INFER_WITH_FIONA_ROOT); PYTHON selects the interpreter.
cd "${INFER_WITH_FIONA_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON=${PYTHON:-python}
$PYTHON scripts/3_table/check_fourier_convention.py --full
$PYTHON - <<'PYEOF'
import numpy as np, finufft
from infer_with_fiona import amplification, paths
GL = str(paths.GL_RULES_DIR)
F = amplification.HybridF(0.0, 1.0, 0.1, 5.0, 201, 0.1, 2000.0, 1024.0, GL, 8)
G = 8.0 * np.pi * 6.6743e-11 / 299792458.0 ** 3 * 1.9884099e+30
for xc, y in [(0.02, 0.3), (0.2, 1.0)]:
    amplification.self_check(F, 100.0, xc, y, np.arange(20.0, 1024.0, 0.25), G, GL, 8)
PYEOF
