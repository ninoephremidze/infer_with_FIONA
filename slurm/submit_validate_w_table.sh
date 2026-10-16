#!/bin/bash
#SBATCH -n 112
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -p dvorkin,shared,sapphire
#SBATCH --mem=64000
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
# Validates fiona_w_table.py (F(w,x_c,y) table) against direct FIONA over the 200-combo
# v3 grid. Stage B's 201x201 table is ~3.5 GiB, so 64 GB is ample.
# Usage: sbatch submit_validate_w_table.sh [extra args for validate_w_table.py]
# Smoke test: sbatch -p test -t 01:00:00 submit_validate_w_table.sh --quick
cd /n/netscratch/dvorkin_lab/Lab/nephremidze/2-LISA/0-dark_matter_inference
/n/home02/nephremidze/.conda/envs/glow311/bin/python -u validate_w_table.py "$@"
