#!/bin/bash
#SBATCH -n 112
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -p dvorkin,shared,sapphire
#SBATCH --mem=128000
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
# Re-derives the FIONA GL schedule by convergence against a high-n reference, plus bGO
# accuracy (derive_gl_schedule_v4.py).
# Usage: sbatch submit_derive_gl_schedule_v4.sh [extra args]
# Smoke test: sbatch -p test -t 01:00:00 submit_derive_gl_schedule_v4.sh --quick
cd /n/netscratch/dvorkin_lab/Lab/nephremidze/2-LISA/0-dark_matter_inference
/n/home02/nephremidze/.conda/envs/glow311/bin/python -u derive_gl_schedule_v4.py "$@"
