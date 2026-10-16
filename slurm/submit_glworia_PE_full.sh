#!/bin/bash
#SBATCH -n 112        # Number of cores (single node) -- matches FIONA side for a like-for-like
                      # wall-clock comparison, even though glworia's own F(w) interpolation
                      # lookup itself is not internally parallelized (only the bilby/dynesty
                      # sampler pool, via npool=nproc in PE_full.py, benefits from more cores).
#SBATCH -N 1
#SBATCH -t 3-00:00    # Runtime in D-HH:MM -- matches FIONA side's own production cap.
#SBATCH -p dvorkin,shared,sapphire
#SBATCH --mem=150000  # MB. Generous worker-process floor, matching the FIONA side's own
                      # ~110-120 GB overhead finding (112 workers each independently loading
                      # their own stack) plus headroom -- glworia's own interpolation table
                      # is already built and loaded once from disk (not rebuilt here), so
                      # this is not sized around table-build memory the way the FIONA-side
                      # submit scripts are.
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nino_ephremidze@college.harvard.edu

# Submission script for glworia's OWN released PE pipeline (scripts/PE_full.py), run against
# their own already-built CIS interpolation table (interpolation/CIS_amp_y_..._N_grid_101_
# N_grid_strong_201_..., built earlier via do_interpolation.py -- NOT rebuilt by this script).
# This is the reference side of the FIONA-vs-glworia production comparison; see
# submit_PE_CIS_glworia_reproduction.sh for the FIONA side of the same benchmark cases.
#
# Usage: sbatch submit_glworia_PE_full.sh <config_path>
#   config_path: path to a glworia PE_submission_settings/CIS/*.json config, RELATIVE TO
#                glworia_repo/ (e.g. PE_submission_settings/CIS/MLz_100.0_dL_500.0_y_0.3000_xc_0.0200.json)
#
# Uses `glow311` (this project's main env, which already has bilby). PE_full.py only LOADS
# an already-built interpolation table via glworia.amp.load_interp (plain numpy/scipy/pickle,
# no JAX at all) -- the older `lensing` env (JAX 0.4.30, jax.experimental.host_callback) is
# only needed for glworia's from-scratch contour-integration code (do_interpolation.py,
# amplification_factor.compute_F), which this script never calls. Confirmed 2026-09-21: the
# `lensing` env doesn't even have bilby installed, so this is not just a preference -- it's
# required for PE_full.py to run at all.
#
# If this job is killed by walltime/preemption/node failure, just resubmit the SAME command
# -- bilby's own dynesty wrapper checkpoints and resumes automatically (matching the FIONA
# side's own behavior).

CONFIG_PATH=${1:?"Usage: sbatch submit_glworia_PE_full.sh <config_path (relative to glworia_repo/)>"}

cd /n/netscratch/dvorkin_lab/Lab/nephremidze/2-LISA/0-dark_matter_inference/glworia_repo
export GLWORIA_PREFIX=$(pwd)

echo "config_path=${CONFIG_PATH}  GLWORIA_PREFIX=${GLWORIA_PREFIX}"

/n/home02/nephremidze/.conda/envs/glow311/bin/python scripts/PE_full.py "${CONFIG_PATH}"
