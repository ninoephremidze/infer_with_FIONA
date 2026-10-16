#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -p test
#SBATCH --mem=64000
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
# FIONA F (min-image time reference) vs glworia F, complex (check_time_reference.py)
cd /n/netscratch/dvorkin_lab/Lab/nephremidze/2-LISA/0-dark_matter_inference
export GLWORIA_PREFIX=$(pwd)/glworia_repo
/n/home02/nephremidze/.conda/envs/glow311/bin/python -u check_time_reference.py
