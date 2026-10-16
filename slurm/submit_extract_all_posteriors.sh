#!/bin/bash
#SBATCH -n 24
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -p test
#SBATCH --mem=400000
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
# Cache posteriors of all FIONA lal runs (final or checkpoint) for the 0-all_*_results notebooks.
cd /n/netscratch/dvorkin_lab/Lab/nephremidze/2-LISA/0-dark_matter_inference
/n/home02/nephremidze/.conda/envs/glow311/bin/python -u extract_all_posteriors.py "$@"
