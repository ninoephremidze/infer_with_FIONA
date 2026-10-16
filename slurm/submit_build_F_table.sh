#!/bin/bash
#SBATCH -n 112
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -p sapphire,dvorkin
#SBATCH --mem=400000
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
# Builds the cached F table for a given M_Lz prior (see build_F_table.py).
# Usage: sbatch submit_build_F_table.sh <m_lz_prior_min> <m_lz_prior_max> [grid_res]
cd /n/netscratch/dvorkin_lab/Lab/nephremidze/2-LISA/0-dark_matter_inference
/n/home02/nephremidze/.conda/envs/glow311/bin/python -u build_F_table.py \
    --m-lz-prior-min "${1:?min}" --m-lz-prior-max "${2:?max}" --grid-res "${3:-high}"
