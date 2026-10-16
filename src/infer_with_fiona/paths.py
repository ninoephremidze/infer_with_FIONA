"""Locations of the repository's data files, caches and outputs.

Everything is resolved from the repository root, so scripts work from any working directory.
The root defaults to the directory that contains src/ (an editable install, pip install -e .)
and can be overridden with the environment variable INFER_WITH_FIONA_ROOT.
"""
import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get('INFER_WITH_FIONA_ROOT',
                                Path(__file__).resolve().parents[2])).resolve()

# shipped inputs
DATA_DIR = REPO_ROOT / 'data'
GL_RULES_DIR = DATA_DIR / 'gl_rules'                    # cached Gauss-Legendre rules
BGO_SWITCH_DIR = DATA_DIR / 'bgo_switch'
BGO_SWITCH = {'CIS': BGO_SWITCH_DIR / 'cis.npz',        # w_s(x_c, y) maps per lens model
              'ULA': BGO_SWITCH_DIR / 'ula.npz',
              'PointLens': BGO_SWITCH_DIR / 'pbh.npz'}
BGO_SWITCH['PBH'] = BGO_SWITCH['PointLens']
CAUSTIC_PKL = DATA_DIR / 'caustic_cis_ycrit.pkl'        # glworia's tabulated y_crit(x_c)
TABLE_VALIDATION_POINTS = DATA_DIR / 'table_validation_points.json'
FISHER_COVARIANCES = DATA_DIR / 'fisher_covariances.npz'

# shipped results
RESULTS_DIR = REPO_ROOT / 'results'
POSTERIORS_DIR = RESULTS_DIR / 'posteriors'

# written by the pipeline (git-ignored)
OUTPUTS_DIR = REPO_ROOT / 'outputs'
TABLE_CACHE_DIR = OUTPUTS_DIR / 'table_cache'           # F tables, keyed by configuration hash
RUNS_DIR = OUTPUTS_DIR / 'runs'                         # bilby/dynesty inference runs
FIGURES_DIR = OUTPUTS_DIR / 'figures'
CHECKS_DIR = OUTPUTS_DIR / 'checks'                     # convergence and validation outputs
