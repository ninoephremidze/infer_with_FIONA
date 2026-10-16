#!/bin/bash
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -p test
#SBATCH --mem=64000
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
# check_fourier_convention.py --full: the Fourier-convention test with the production HybridF
# (cached F table, ~5 GB), plus the PE self-check of HybridF against direct FIONA in the new convention.
cd /n/netscratch/dvorkin_lab/Lab/nephremidze/2-LISA/0-dark_matter_inference
PY=/n/home02/nephremidze/.conda/envs/glow311/bin/python
$PY check_fourier_convention.py --full
$PY - <<'PYEOF'
import os, numpy as np, finufft, fiona_pe_F
R = os.getcwd()
F = fiona_pe_F.HybridF(0.0, 1.0, 0.1, 5.0, 201, 0.1, 2000.0, 1024.0, os.path.join(R, 'fiona_gl_cache'), 8, root=R)
G = 8.0 * np.pi * 6.6743e-11 / 299792458.0 ** 3 * 1.9884099e+30
for xc, y in [(0.02, 0.3), (0.2, 1.0)]:
    fiona_pe_F.self_check(F, 100.0, xc, y, np.arange(20.0, 1024.0, 0.25), G, os.path.join(R, 'fiona_gl_cache'), 8)
PYEOF
