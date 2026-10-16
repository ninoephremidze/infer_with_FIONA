#!/bin/bash
#SBATCH -n 112
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH -p sapphire,dvorkin
#SBATCH --mem=400000
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
# Builds + caches the production F table for ULA or PBH and validates it against direct FIONA
# (validate_w_table_lens.py). Usage: sbatch submit_validate_w_table_lens.sh --lens ULA|PBH [args]
cd /n/netscratch/dvorkin_lab/Lab/nephremidze/2-LISA/0-dark_matter_inference
/n/home02/nephremidze/.conda/envs/glow311/bin/python -u validate_w_table_lens.py "$@"
