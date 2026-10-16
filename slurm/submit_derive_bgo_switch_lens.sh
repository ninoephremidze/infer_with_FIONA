#!/bin/bash
#SBATCH -n 112
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -p sapphire,dvorkin
#SBATCH --mem=200000
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
# bGO switch + GL v4 check for the ULA soliton or point lens (derive_bgo_switch_lens.py).
# Usage: sbatch submit_derive_bgo_switch_lens.sh --lens ULA|PointLens [--quick]
# Smoke test: sbatch -p test -t 01:00:00 submit_derive_bgo_switch_lens.sh --lens ULA --quick
cd /n/netscratch/dvorkin_lab/Lab/nephremidze/2-LISA/0-dark_matter_inference
/n/home02/nephremidze/.conda/envs/glow311/bin/python -u derive_bgo_switch_lens.py "$@"
