"""
plot_somato_fit.py
------------------
Diagnostic plots for a SomatoModel fit produced by run_somato_fit.py / optimizer.py.

Self-contained: reads ONLY the result HDF5 (numpy + h5py + matplotlib). It needs no
model, no env vars, no mne — because the optimizer's final save stores both the target
(ref['y0']) and the best-fit prediction (ref['sim']['simMEP2']) in the file.

Three figures:
    <stem>_convergence.pdf    - best cost per generation
    <stem>_parameters.pdf     - parameter evolution + final values within bounds
    <stem>_fit_comparison.pdf - best fit vs target, per ROI (adapts to the observable)

Standalone:
    python plot_somato_fit.py /path/to/result.h5
Or import plot_result(path, true_params=...) from the driver.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe
import matplotlib.pyplot as plt

from h5_helpers import load_h5_to_dict
import h5py

# ROI order must match model2optimize.ROIS (the order predicted/target are concatenated).
ROIS = ("A3b", "A1", "S2")

# Okabe-Ito colorblind-safe qualitative palette (fixed order, never cycled).
_OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00",
              "#F0E442", "#000000"]
_TARGET_COLOR = "#000000"     # target: neutral ink (reference)
_PREDICT_COLOR = "#D55E00"    # best fit: vermillion (distinct, CVD-safe)

_INK = "#222222"
_MUTED = "#888888"


# ---------------------------------------------------------------------------
# loading / decoding
# ---------------------------------------------------------------------------
def _decode(x):
    """bytes -> str; array/list of bytes -> list[str]; else unchanged."""
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    if isinstance(x, np.ndarray) and x.dtype.kind == "S":
        return [b.decode("utf-8") for b in x]
    return x


def load_result(path):
    """Load the optimizer result HDF5 into a plain dict (strings decoded)."""
    with h5py.File(path, "r") as f:
        data = load_h5_to_dict(f)
    ref = data["ref"]
    ref["observable"] = _decode(ref.get("observable", b"timecourse"))
    ref["param_names"] = _decode(ref["param_names"])
    if "_obs_sections" in ref:
        # stored as an array of "name:length" byte strings
        ref["_obs_sections"] = [s.split(":") for s in _decode(ref["_obs_sections"])]
        ref["_obs_sections"] = [(n, int(l)) for n, l in ref["_obs_sections"]]
    return data


def _style_axes(ax):
    """Recessive spines + light grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_MUTED)
    ax.tick_params(colors=_INK, labelsize=9)
    ax.grid(True, color=_MUTED, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 1. convergence
# ---------------------------------------------------------------------------
def plot_convergence(KS, outpath):
    """Best cost per generation (log-y where possible). One series -> no legend."""
    KS = np.atleast_1d(np.asarray(KS, dtype=float))
    gens = np.arange(1, len(KS) + 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(gens, KS, "-o", color=_OKABE_ITO[0], lw=2.0, markersize=7,
            markerfacecolor=_OKABE_ITO[0], markeredgecolor="white", markeredgewidth=0.8)
    if np.all(KS > 0):
        ax.set_yscale("log")
    ax.set_xlabel("Generation", color=_INK)
    ax.set_ylabel("Best cost  (sum of squared residuals)", color=_INK)
    ax.set_title("Error convergence", color=_INK, fontsize=12, fontweight="bold")
    if len(gens) == 1:
        ax.set_xticks([1])
        ax.annotate("single generation (tg=1)", xy=(1, KS[0]),
                    xytext=(0.5, 0.9), textcoords="axes fraction",
                    color=_MUTED, fontsize=9)
    # direct-label the final value only
    ax.annotate(f"{KS[-1]:.4g}", xy=(gens[-1], KS[-1]),
                xytext=(4, 4), textcoords="offset points", color=_INK, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


# ---------------------------------------------------------------------------
# 2. parameters
# ---------------------------------------------------------------------------
def plot_parameters(KP, p_post, boundary, param_names, outpath, true_params=None):
    """Left: parameter evolution normalised to bounds. Right: final values within bounds."""
    KP = np.atleast_2d(np.asarray(KP, dtype=float))       # (n_gen, n_param)
    p_post = np.asarray(p_post, dtype=float)
    bounds = np.asarray(boundary, dtype=float)            # (n_param, 2)
    names = list(param_names)
    n = len(names)
    span = (bounds[:, 1] - bounds[:, 0]).copy()
    span[span == 0] = 1.0

    fig, (ax_evo, ax_bar) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) evolution, normalised to [0,1] within each param's bounds
    gens = np.arange(1, KP.shape[0] + 1)
    evo_norm = (KP - bounds[:, 0]) / span
    for j, name in enumerate(names):
        c = _OKABE_ITO[j % len(_OKABE_ITO)]
        ax_evo.plot(gens, evo_norm[:, j], "-o", color=c, lw=1.8, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.6, label=name)
    ax_evo.set_ylim(-0.05, 1.05)
    ax_evo.set_xlabel("Generation", color=_INK)
    ax_evo.set_ylabel("Value  (normalised to bounds)", color=_INK)
    ax_evo.set_title("Parameter evolution", color=_INK, fontsize=12, fontweight="bold")
    if KP.shape[0] == 1:
        ax_evo.set_xticks([1])
    ax_evo.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5),
                  frameon=False)
    _style_axes(ax_evo)

    # (b) final params as position within their [lo, hi] bound range
    y = np.arange(n)[::-1]     # top-to-bottom in listed order
    frac = (p_post - bounds[:, 0]) / span
    ax_bar.hlines(y, 0, 1, color=_MUTED, alpha=0.35, linewidth=6, zorder=1)
    ax_bar.scatter(frac, y, s=90, color=_PREDICT_COLOR, zorder=3,
                   edgecolor="white", linewidth=0.8, label="fitted")
    if true_params is not None:
        tvals = np.array([true_params.get(nm, np.nan) for nm in names], dtype=float)
        tfrac = (tvals - bounds[:, 0]) / span
        ax_bar.scatter(tfrac, y, s=90, marker="|", color=_INK, zorder=4,
                       linewidth=2.0, label="true")
    for yi, nm, val in zip(y, names, p_post):
        ax_bar.annotate(f"{val:.4g}", xy=(1.02, yi), va="center",
                        color=_INK, fontsize=8)
    ax_bar.set_xlim(-0.03, 1.18)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(names, fontsize=9, color=_INK)
    ax_bar.set_xlabel("Position within bounds  (lo → hi)", color=_INK)
    ax_bar.set_title("Optimized parameters", color=_INK, fontsize=12, fontweight="bold",
                     pad=22)
    if true_params is not None:
        # above the panel (2 entries) so it never collides with the value labels
        ax_bar.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                      ncol=2, frameon=False)
    _style_axes(ax_bar)
    ax_bar.grid(True, axis="x", color=_MUTED, alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


# ---------------------------------------------------------------------------
# 3. fit vs target
# ---------------------------------------------------------------------------
def _sections(ref):
    """Return [(observable_name, length), ...] for the concatenated residual vector."""
    if "_obs_sections" in ref:
        return ref["_obs_sections"]
    return [(ref["observable"], int(np.size(ref["y0"])))]


def plot_fit_comparison(ref, outpath):
    """Best fit vs target, split per ROI, adapting to the observable(s)."""
    predicted = np.asarray(ref["sim"]["simMEP2"], dtype=float)
    target = np.asarray(ref["y0"], dtype=float)
    sections = _sections(ref)

    nrows = len(sections)
    fig, axes = plt.subplots(nrows, 3, figsize=(13, 3.4 * nrows), squeeze=False)

    off = 0
    for row, (obs, seclen) in enumerate(sections):
        pred = predicted[off:off + seclen]
        tgt = target[off:off + seclen]
        off += seclen
        per_roi = seclen // len(ROIS)

        for col, roi in enumerate(ROIS):
            ax = axes[row][col]
            p = pred[col * per_roi:(col + 1) * per_roi]
            t = tgt[col * per_roi:(col + 1) * per_roi]

            if obs == "timecourse":
                x = np.linspace(0, 250, per_roi)   # onset .. +250 ms (analysis window)
                ax.plot(x, t, color=_TARGET_COLOR, lw=1.8, label="target")
                ax.plot(x, p, color=_PREDICT_COLOR, lw=1.6, label="best fit")
                ax.axhline(0, color=_MUTED, lw=0.6, alpha=0.5)
                if row == nrows - 1:
                    ax.set_xlabel("Time (ms)", color=_INK)
                if col == 0:
                    ax.set_ylabel("timecourse\n(shared-peak norm.)", color=_INK)

            elif obs == "prestim_spectrum":
                freqs = ref.get("_prestim_freqs")
                x = np.asarray(freqs, dtype=float) if freqs is not None and np.size(freqs) == per_roi \
                    else np.arange(per_roi)
                ax.plot(x, t, "-o", color=_TARGET_COLOR, lw=1.8, markersize=4, label="target")
                ax.plot(x, p, "-o", color=_PREDICT_COLOR, lw=1.6, markersize=4, label="best fit")
                if row == nrows - 1:
                    ax.set_xlabel("Frequency (Hz)", color=_INK)
                if col == 0:
                    ax.set_ylabel("prestim spectrum\n(unit-sum norm.)", color=_INK)

            elif obs == "timefreq":
                ncols = int(ref.get("_tf_ncols", per_roi // 40))
                P = p.reshape(40, ncols)
                T = t.reshape(40, ncols)
                vmax = float(np.percentile(np.abs(np.concatenate([P, T])), 99)) or 1.0
                # two stacked panels within the cell would need gridspec; instead show
                # the difference (fit - target) so one panel per ROI suffices.
                im = ax.imshow(P - T, aspect="auto", origin="lower", cmap="RdBu_r",
                               vmin=-vmax, vmax=vmax,
                               extent=[-200, 250, 1, 40])
                if row == nrows - 1:
                    ax.set_xlabel("Time (ms)", color=_INK)
                if col == 0:
                    ax.set_ylabel("timefreq  Δlog10\n(fit − target)", color=_INK)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if row == 0:
                ax.set_title(roi, color=_INK, fontsize=11)
            _style_axes(ax)

        # one legend per row (line observables), drawn on the last column
        if obs in ("timecourse", "prestim_spectrum"):
            axes[row][-1].legend(fontsize=8, loc="upper right", frameon=False)

    fig.suptitle("Best fit vs target", color=_INK, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def plot_result(path, true_params=None, outdir=None):
    """Produce all three figures for a result HDF5. Returns the list of PNG paths."""
    data = load_result(path)
    ref = data["ref"]
    boundary = ref.get("boundary", ref["model"]["boundary"])

    stem = os.path.splitext(os.path.basename(path))[0]
    outdir = outdir or os.path.join(os.path.dirname(os.path.abspath(path)), "figures")
    os.makedirs(outdir, exist_ok=True)
    p = lambda suffix: os.path.join(outdir, f"{stem}_{suffix}.pdf")

    outs = [
        plot_convergence(data["KS"], p("convergence")),
        plot_parameters(data["KP"], data["p_post"], boundary, ref["param_names"],
                        p("parameters"), true_params=true_params),
        plot_fit_comparison(ref, p("fit_comparison")),
    ]
    print("saved figures:")
    for o in outs:
        print(" ", o)
    return outs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python plot_somato_fit.py <result.h5>")
    plot_result(sys.argv[1])
