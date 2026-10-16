#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -p shared,sapphire
#SBATCH --mem=16000
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
# Pre-generates one large Gauss-Legendre rule into the PE GL cache (fiona_gl_cache) with
# scipy.special.roots_legendre. Measured cost is O(n^2), not O(n): 64k 107 s, 128k 425 s,
# so 256k ~0.5 h, 512k ~2 h, 1024k ~8 h (estimates). One-time: cached afterwards.
# Usage: sbatch submit_seed_gl.sh <n>
cd /n/netscratch/dvorkin_lab/Lab/nephremidze/2-LISA/0-dark_matter_inference
/n/home02/nephremidze/.conda/envs/glow311/bin/python -u -c "
import finufft, os, fiona_adaptive_gl_v3 as g
g.seed_gl_cache(os.path.abspath('fiona_gl_cache'), n_values=($1,))
"
