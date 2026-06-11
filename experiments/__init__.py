"""Training scripts reproducing the experiments in the Neural FIM paper.

Each module is runnable as ``python -m experiments.<name>`` from the repository
root (or ``python experiments/<name>.py``). See ``experiments/README.md`` for the
mapping between scripts and paper figures/tables.
"""
import os
import sys

# Make ``src`` importable when scripts are run directly from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
