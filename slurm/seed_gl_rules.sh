#!/bin/bash
# Slurm resources below are those used on Harvard FASRC (112-core sapphire nodes); adjust for your cluster.
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -p shared,sapphire
#SBATCH --mem=16000
#SBATCH -o outputs/logs/%x_%j.out
#SBATCH -e outputs/logs/%x_%j.err
# Pre-generates one large Gauss-Legendre rule into data/gl_rules/ with
# scipy.special.roots_legendre. Measured cost is O(n^2), not O(n): 64k 107 s, 128k 425 s,
# so 256k ~0.5 h, 512k ~2 h, 1024k ~8 h (estimates). One-time: cached afterwards.
# Usage: sbatch slurm/seed_gl_rules.sh <n>
# Run from the repository root (or set INFER_WITH_FIONA_ROOT); PYTHON selects the interpreter.
cd "${INFER_WITH_FIONA_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON=${PYTHON:-python}
$PYTHON -u -c "
import finufft
from infer_with_fiona import paths
from infer_with_fiona.quadrature.gl_cache import seed_gl_cache
seed_gl_cache(str(paths.GL_RULES_DIR), n_values=(${1:?Usage: sbatch slurm/seed_gl_rules.sh <n>},))
"
