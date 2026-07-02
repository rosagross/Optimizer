"""
run_somato_fit.py
-----------------
Driver: fit SomatoModel parameters with the hybrid GA + Gauss-Newton optimizer in
optimizer.py.

Edit the CONFIG block, then run:

    python run_somato_fit.py

What it does
    1. Builds the `ref` dict (parameter bounds, subjects, chosen observable).
    2. Builds the target vector `ref['y0']`:
         - synthetic  : from a saved ground-truth dipole trace (parameter recovery)
         - measured   : from the group ROI target CSVs
       ...processed identically to how `predicted` is built (same flatten helpers),
       so `residual = predicted - y0` is meaningful.
    3. Smoke-checks that predicted and target have the same shape.
    4. Runs `_run_and_save(ref, RESULT_PATH)` and reports the fitted parameters.

Note on generations: optimizer.py hard-codes `tg = 1` (a single generation) and
`N1 = 60`. For a real fit raise `tg` (e.g. 20-50) in optimizer.py; for a quick
smoke test the defaults are fine.
"""

import os
import numpy as np

import somato_env
somato_env.bootstrap()  # set env vars + sys.path BEFORE importing model / optimizer

from model2optimize import (
    get_model, simulate_dipoles, predict,
    flatten_timecourse, flatten_prestim, flatten_timefreq,
    _TF_ANALYSIS, ROIS,
)
from optimizer import _run_and_save

# ===========================================================================
# CONFIG
# ===========================================================================
# Which observable forms the residual the optimizer minimises.
#   'timecourse' | 'prestim_spectrum' | 'timefreq' | 'combined'
OBSERVABLE = "prestim_spectrum"
COMBINED_OBSERVABLES = ("timecourse", "prestim_spectrum")  # used only if OBSERVABLE == 'combined'


# 'synthetic' -> parameter-recovery: fit a ground-truth trace made from TRUE_PARAMS
#                (validates the wiring — the optimum should recover TRUE_PARAMS)
# 'measured'  -> fit the measured group ROI target CSVs
TARGET_MODE = "measured"

# Synthetic ground-truth parameters. By default the target trace is generated
# in-process with the current model + forward models (self-consistent, so recovery
# is exact). Set SYNTHETIC_TARGET_PATH to load a pre-saved trace instead — but note a
# trace saved under different forward models / model code will NOT be reproducible
# (the dipole projection sign/scale differs), so in-process generation is recommended.
TRUE_PARAMS = {
    "coupling_strength": 10, "strength_I": 0.68, "g_intercortical": 1.0,
    "Ib_strength": 6, "Iext_strength": 40, "Iext_duration": 0.016,
}
SYNTHETIC_TARGET_PATH = None  # e.g. ".../optimization/synthetic_target.hdf5"; None = generate in-process

# Where to save the optimizer's HDF5 result.
RESULT_PATH = os.path.join(os.environ["SIMDIR"], "optimization", "somato_gafit.h5")

# Parameters to fit and their [lower, upper] bounds (mirrors run_optimization.py,
# minus 'scaling_factor' which is not a model attribute).
PARAM_NAMES = [
    "coupling_strength",
    "strength_I",
    "g_intercortical",
    "Ib_strength",
    "Iext_strength",
    "Iext_duration",
]
BOUNDARY = np.array([
    [1,     30.0],   # coupling_strength
    [0.5,    0.8],   # strength_I
    [0.5,    3.0],   # g_intercortical
    [2,     10.0],   # Ib_strength
    [1,    100.0],   # Iext_strength
    [0.001,  0.1],   # Iext_duration
])

# Measured-target CSVs (under RESDIR), same paths as run_optimization.py.
_ROI_DIR = os.path.join(
    os.environ["RESDIR"], "Figures", "Main", "eeg_results",
    "source_reconstruction", "group", "_preprestim_corrected", "roi_epochswise",
)
TF_CSV = os.path.join(_ROI_DIR, "group_roi_tf_morlet_ses-elec_preprestim_corrected.csv")
TC_CSV = os.path.join(_ROI_DIR, "group_roi_timecourse_pooled_ses-elec_preprestim_corrected.csv")
PS_CSV = os.path.join(_ROI_DIR, "group_roi_prestim_spectrum_ses-elec_preprestim_corrected.csv")


# ===========================================================================
# Target construction
# ===========================================================================
def _observables():
    return list(COMBINED_OBSERVABLES) if OBSERVABLE == "combined" else [OBSERVABLE]


def build_target(ref, model, sim_dip0, target_dip):
    """Build ref['y0'] for the selected observable(s), storing any per-observable
    selection state the predictor needs. `sim_dip0` is one baseline simulation used
    only to fix the simulated frequency grid / TF window length. `target_dip` is the
    ground-truth dipole trace in synthetic mode, else None (measured)."""
    pieces = []
    for obs in _observables():
        if obs == "timecourse":
            tgt = (model.compute_timecourse(target_dip) if target_dip is not None
                   else model.load_target_timecourse(TC_CSV))
            pieces.append(flatten_timecourse(tgt))

        elif obs == "prestim_spectrum":
            fmin, fmax = 1.0, 40.0
            f_sim, _ = model.compute_prestim_spectrum(sim_dip0, fmin, fmax)
            if target_dip is not None:
                f_tgt, spec_tgt = model.compute_prestim_spectrum(target_dip, fmin, fmax)
            else:
                f_tgt, spec_tgt = model.load_target_prestim_spectrum(PS_CSV)
            tmask = (f_tgt >= fmin) & (f_tgt <= fmax)
            common = np.intersect1d(np.round(f_sim, 6), np.round(f_tgt[tmask], 6))
            ref["_prestim_sim_sel"] = np.isin(np.round(f_sim, 6), common)
            ref["_prestim_freqs"] = common          # per-ROI freq axis (for plotting)
            tgt_sel = np.isin(np.round(f_tgt, 6), common)
            pieces.append(flatten_prestim(spec_tgt, tgt_sel))

        elif obs == "timefreq":
            tf_sim = model.compute_timefreq(sim_dip0)
            tf_tgt = (model.compute_timefreq(target_dip) if target_dip is not None
                      else model.load_target_timefreq(TF_CSV))
            sim_cols = tf_sim[ROIS[0]][:, _TF_ANALYSIS].shape[1]
            tgt_cols = tf_tgt[ROIS[0]][:, _TF_ANALYSIS].shape[1]
            ref["_tf_ncols"] = min(sim_cols, tgt_cols)
            pieces.append(flatten_timefreq(tf_tgt, ncols=ref["_tf_ncols"]))

        else:
            raise ValueError(f"unknown observable: {obs!r}")

    # record section boundaries so plot_somato_fit can split the residual vector
    # per observable (needed for 'combined'); stored as "name:length" strings.
    ref["_obs_sections"] = [f"{obs}:{p.size}" for obs, p in zip(_observables(), pieces)]

    return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]


# ===========================================================================
# Run
# ===========================================================================
def main():
    ref = {
        "model": {"boundary": BOUNDARY},
        "param_names": PARAM_NAMES,
        "subjects": somato_env.SUBID_ELEC,
        "base_params": somato_env.BASE_PARAMS,
        "observable": OBSERVABLE,
        "combined_observables": COMBINED_OBSERVABLES,
    }

    model = get_model(ref)

    # one baseline simulation at mid-bounds: fixes the simulated freq grid / TF window,
    # and doubles as the smoke-test prediction.
    p0 = BOUNDARY.mean(axis=1)
    sim_dip0 = simulate_dipoles(model, p0, ref)

    # ground-truth trace (synthetic) or None (measured)
    target_dip = None
    if TARGET_MODE == "synthetic":
        if SYNTHETIC_TARGET_PATH:
            target_dip = model.load_dipole_trace(SYNTHETIC_TARGET_PATH)
        else:  # generate in-process from TRUE_PARAMS (self-consistent recovery)
            p_true = np.array([TRUE_PARAMS[n] for n in PARAM_NAMES])
            target_dip = simulate_dipoles(model, p_true, ref)
    elif TARGET_MODE != "measured":
        raise ValueError(f"TARGET_MODE must be 'synthetic' or 'measured', got {TARGET_MODE!r}")

    ref["y0"] = build_target(ref, model, sim_dip0, target_dip)

    # smoke check: predicted and target must share length (constant across all evals)
    predicted0 = predict(model, sim_dip0, ref)
    assert predicted0.shape == ref["y0"].shape, (
        f"predicted {predicted0.shape} != target {ref['y0'].shape} — "
        f"observable/ROI processing mismatch"
    )
    print(f"[ok] observable={OBSERVABLE!r} target_mode={TARGET_MODE!r} "
          f"residual length={ref['y0'].size}")

    p_post = _run_and_save(ref, RESULT_PATH)

    print("\n== fitted parameters ==")
    for name, val in zip(PARAM_NAMES, p_post):
        if TARGET_MODE == "synthetic" and name in TRUE_PARAMS:
            print(f"  {name:18s} = {val:10.4f}   (true {TRUE_PARAMS[name]})")
        else:
            print(f"  {name:18s} = {val:10.4f}")
    print(f"\nresult saved to: {RESULT_PATH}")

    # diagnostic figures (convergence, optimized params, best-fit vs target)
    from plot_somato_fit import plot_result
    true = TRUE_PARAMS if TARGET_MODE == "synthetic" else None
    plot_result(RESULT_PATH, true_params=true)


if __name__ == "__main__":
    main()
