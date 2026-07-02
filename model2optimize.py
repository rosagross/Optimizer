"""
model2optimize.py  (USER-IMPLEMENTED adapter for optimizer.py)
--------------------------------------------------------------
Runs one forward simulation of the SomatoModel for a candidate parameter vector
`p` and returns the model prediction as a flat 1-D vector, ready for the GA +
Gauss-Newton optimizer in optimizer.py.

The optimizer minimises a residual *vector* (`predicted - ref['y0']`) whose
sum-of-squares is the cost (see GA/gradient_toolbox/evaluation.py). We therefore
turn the model output into a flat, fixed-length vector, normalised exactly the way
the SomatoModel's own error functions normalise, so that
    sum((predicted - y0)**2)
is (proportional to) the model's built-in MSE for the chosen observable.

Which observable forms the residual is chosen by `ref['observable']`:
    'timecourse'        - per-ROI ERP time course      (shared-peak normalised)
    'prestim_spectrum'  - pre-stimulus power spectrum   (unit-sum relative power)
    'timefreq'          - Morlet TF power               (log10 baseline-normalised)
    'combined'          - concatenation of the above (see ref['combined_observables'])

`ref` must also contain:
    ref['param_names'] : ordered list matching the boundary rows / p
    ref['subjects']    : subject-id list for compute_dipoles
Optional:
    ref['base_params'] : fixed model params (defaults to somato_env.BASE_PARAMS)

The predicted vector is also stashed at ref['sim']['simMEP2'] so optimizer.py's
live-plot (which hardcodes houtput['sim']['simMEP2']) works unchanged.
"""

import numpy as np

import somato_env

ROIS = ("A3b", "A1", "S2")
_EPS = 1e-10

# Windows/normalisation constants — copied from somato_model's error functions so
# the residual matches the model's own metric exactly.
_TC_ANALYSIS = slice(250, None)   # compute_error_timecourse: stimulus onset onward
_TF_BASELINE = slice(150, 200)    # compute_error_timefreq: -500..-430 ms baseline
_TF_ANALYSIS = slice(200, 400)    # compute_error_timefreq: -200 ms onward


# ---------------------------------------------------------------------------
# Model handling (one cached instance, reused across evaluations)
# The model is cached at module scope — NOT inside `ref` — because optimizer.py
# serialises the whole `ref` dict to HDF5 at the end of a run, and a SomatoModel
# object doesn't belong in there.
# ---------------------------------------------------------------------------
_MODEL = None
_MODEL_KEY = None


def get_model(ref):
    """Return a cached SomatoModel, building it on first use (or if base_params change)."""
    global _MODEL, _MODEL_KEY
    base_params = ref.get("base_params", somato_env.BASE_PARAMS)
    key = repr(sorted(base_params.items()))
    if _MODEL is None or key != _MODEL_KEY:
        SomatoModel = somato_env.import_somato_model()
        _MODEL = SomatoModel(base_params)
        _MODEL_KEY = key
    return _MODEL


def simulate_dipoles(model, p, ref):
    """Run one full simulation for parameter vector p and return the ROI dipoles."""
    params = dict(zip(ref["param_names"], np.asarray(p, dtype=float)))
    model.apply_params(params)
    model.initialize_state()
    model.simulate()
    return model.compute_dipoles(ref["subjects"])


# ---------------------------------------------------------------------------
# Observable -> flat vector helpers.
# Each takes a *dict of ROI arrays* (or spectra) and returns a 1-D vector.
# The SAME helper is applied to the simulated output and to the target, so
# `predicted` and `y0` are guaranteed to be built identically.
# ---------------------------------------------------------------------------
def flatten_timecourse(tc_dict):
    """Shared-peak-normalised, concatenated ROI ERP traces (mirrors
    compute_error_timecourse: one peak across A3b/A1/S2 preserves cross-area ratios)."""
    peak = max(np.max(np.abs(np.asarray(tc_dict[r])[_TC_ANALYSIS])) for r in ROIS) + _EPS
    return np.concatenate([np.asarray(tc_dict[r])[_TC_ANALYSIS] / peak for r in ROIS])


def flatten_prestim(spec_dict, sel):
    """Unit-sum (relative power) per ROI over the selected common bins, concatenated
    (mirrors compute_error_prestim_spectrum)."""
    out = []
    for r in ROIS:
        s = np.asarray(spec_dict[r])[sel]
        out.append(s / (s.sum() + _EPS))
    return np.concatenate(out)


def flatten_timefreq(tf_dict, ncols=None):
    """log10 baseline-normalised TF power per ROI, flattened and concatenated
    (mirrors compute_error_timefreq). If ncols is given, the analysis window is
    truncated so simulated and target vectors share a length."""
    out = []
    for r in ROIS:
        P = np.asarray(tf_dict[r])
        bl = P[:, _TF_BASELINE].mean(axis=1, keepdims=True)
        an = P[:, _TF_ANALYSIS]
        if ncols is not None:
            an = an[:, :ncols]
        out.append(np.log10(an / (bl + _EPS) + _EPS).ravel())
    return np.concatenate(out)


def _observable_list(ref):
    obs = ref.get("observable", "timecourse")
    if obs == "combined":
        return list(ref.get("combined_observables", ("timecourse", "prestim_spectrum")))
    return [obs]


def _predict_one(model, sim_dip, obs, ref):
    """Flat predicted vector for a single observable from a simulated dipole set."""
    if obs == "timecourse":
        return flatten_timecourse(model.compute_timecourse(sim_dip))
    if obs == "prestim_spectrum":
        _f, spec = model.compute_prestim_spectrum(sim_dip)
        return flatten_prestim(spec, ref["_prestim_sim_sel"])
    if obs == "timefreq":
        return flatten_timefreq(model.compute_timefreq(sim_dip), ncols=ref.get("_tf_ncols"))
    raise ValueError(f"unknown observable: {obs!r}")


def predict(model, sim_dip, ref):
    """Flat predicted vector across all selected observables."""
    parts = [_predict_one(model, sim_dip, obs, ref) for obs in _observable_list(ref)]
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


# ---------------------------------------------------------------------------
# The interface optimizer.py expects
# ---------------------------------------------------------------------------
def model2optimize(p, ref):
    """Run the model with parameters p; return (predicted_vector, ref).

    Also stores the prediction at ref['sim']['simMEP2'] for the optimizer's live plot.
    """
    model = get_model(ref)
    sim_dip = simulate_dipoles(model, p, ref)
    predicted = predict(model, sim_dip, ref)

    ref.setdefault("sim", {})["simMEP2"] = predicted
    return predicted, ref
