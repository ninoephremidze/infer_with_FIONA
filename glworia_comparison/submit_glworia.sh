#!/bin/bash
# Slurm resources below are those used on Harvard FASRC (112-core sapphire nodes); adjust for your cluster.
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
#SBATCH -o glworia_%j.out
#SBATCH -e glworia_%j.err

# Submission script for glworia's OWN released PE pipeline (scripts/PE_full.py), run against
# their own already-built CIS interpolation table (interpolation/CIS_amp_y_..._N_grid_101_
# N_grid_strong_201_..., built earlier via do_interpolation.py -- NOT rebuilt by this script).
# This is the reference side of the FIONA-vs-glworia production comparison; see
# slurm/run_inference_cis.sh for the FIONA side of the same benchmark cases.
#
# Usage: GLWORIA_PREFIX=/path/to/glworia sbatch glworia_comparison/submit_glworia.sh <config_path>
#   config_path: a PE_submission_settings/CIS/*.json config RELATIVE TO GLWORIA_PREFIX, e.g. one of the
#                *_lal_loguniform.json files copied there from glworia_comparison/configs/.
#
# PE_full.py only loads the already-built interpolation table (plain numpy/scipy/pickle), so it runs in
# the same Python environment as this repository. Building the table (do_interpolation.py) needs
# glworia's own JAX environment.
#
# If this job is killed by walltime/preemption/node failure, just resubmit the SAME command
# -- bilby's own dynesty wrapper checkpoints and resumes automatically (matching the FIONA
# side's own behavior).

CONFIG_PATH=${1:?"Usage: GLWORIA_PREFIX=/path/to/glworia sbatch glworia_comparison/submit_glworia.sh <config_path (relative to GLWORIA_PREFIX)>"}

cd "${GLWORIA_PREFIX:?set GLWORIA_PREFIX to the root of a patched glworia checkout}"
export GLWORIA_PREFIX=$(pwd)
PYTHON=${PYTHON:-python}

echo "config_path=${CONFIG_PATH}  GLWORIA_PREFIX=${GLWORIA_PREFIX}"

$PYTHON scripts/PE_full.py "${CONFIG_PATH}"
