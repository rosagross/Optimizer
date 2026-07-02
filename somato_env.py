"""
somato_env.py
-------------
Environment + import bootstrap for driving the SomatoModel
(/data/p_02989/.../SomatosensoryLaminarModel) from this Optimizer repo.

The SomatoModel module reads several environment variables *at import time*
(SIMDIR, WDDIR, DATADIR, SUBJECTS_DIR) and imports `parameters` / `helper_functions`
from directories that are not on the path by default. This module sets sensible
defaults for the current machine (the /data/p_02989 tree) and puts the required
directories on sys.path, then exposes `import_somato_model()` which returns the
`SomatoModel` class.

Override any path by exporting the corresponding environment variable before running.
"""

import os
import sys

# --- default paths on the /data/p_02989 machine (override via env if needed) ---------
_DEFAULTS = {
    "WDDIR":        "/data/p_02989/Modelling/grossmannr_wd/SomatosensoryLaminarModel",
    "SIMDIR":       "/data/p_02989/Modelling/output_grossmannr",
    "RESDIR":       "/data/p_02989/shared_workspace/results_grossmannr",
    "DATADIR":      "/data/p_02989/shared_workspace/datadir",
    "SUBJECTS_DIR": "/data/p_02989/shared_workspace/freesurfer",
}


def bootstrap():
    """Set env-var defaults (only if unset) and add the model dirs to sys.path.

    Must be called *before* importing somato_model. Also forces a non-interactive
    matplotlib backend so the optimizer's live-plot calls don't require a display.
    """
    for k, v in _DEFAULTS.items():
        os.environ.setdefault(k, v)

    # headless-safe plotting (optimizer.py uses plt.ion()/show()/pause())
    os.environ.setdefault("MPLBACKEND", "Agg")

    wddir = os.environ["WDDIR"]
    sim_dir = os.path.join(wddir, "Simulations")
    model_dir = os.path.join(sim_dir, "model")
    # parameters.py lives in Simulations/, somato_model.py in Simulations/model/
    for p in (sim_dir, model_dir):
        if p not in sys.path:
            sys.path.insert(0, p)


def import_somato_model():
    """Bootstrap then import and return the SomatoModel class."""
    bootstrap()
    from somato_model import SomatoModel  # noqa: E402  (import after path setup)
    return SomatoModel


# Fixed (non-optimised) model parameters — mirrors run_optimization.py:68-75.
BASE_PARAMS = {
    "g_thal":          2,
    "sI_thal":         0.5,
    "extI_cellcounts": 1000,
    "bI_cellcounts":   100,
    "thal_cellcounts": 500,
    "area":            "all",
}

# Electrical-modality subjects whose forward models are averaged in compute_dipoles
# (identical list to run_optimization.py:31).
SUBID_ELEC = [15, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 34, 35, 36, 37, 38, 39, 40,
              42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
