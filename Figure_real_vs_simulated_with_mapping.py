# -*- coding: utf-8 -*-
"""
Figure_real_vs_simulated_with_mapping.py

Two-part publication figure:
  LEFT (2x3 grid): a real N=1 g2(tau) trace (top row) and a background-
  fraction-matched simulated N=1 trace (bottom row), shown at three
  matched integration times, sparse to clean.
  RIGHT (single panel): the iter_# -> real-integration-time correspondence
  curve, with the three simulated iterations used on the left marked and
  labeled to match the left panels' letters - tying both halves of the
  figure together visually.

Matching philosophy: the three real integration times are looked up
against the mapping curve (Compute_iter_to_integration_time_mapping.py)
to find the simulated iteration with an EQUIVALENT baseline count rate,
rather than picking a simulated iteration arbitrarily - this is what
makes the comparison meaningful. The simulated file itself was chosen to
have an injected background fraction close to the real file's calibrated
value (see training_labels.csv / experimental_calibration.csv), so any
visual difference between top and bottom rows reflects sparsity, not an
unrelated difference in noise level.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def require_file(filepath):
    """Fails early with a clear, actionable message (listing what IS in
    the target folder) instead of letting a bare FileNotFoundError from
    deep inside pandas be the only clue."""
    if not os.path.exists(filepath):
        folder = os.path.dirname(filepath) or "."
        if os.path.isdir(folder):
            available = sorted(os.listdir(folder))
            listing = "\n    ".join(available) if available else "(folder is empty)"
        else:
            listing = f"(folder '{folder}' does not exist at all - check your working directory)"
        raise FileNotFoundError(
            f"\n\nExpected file not found:\n    {os.path.abspath(filepath)}\n\n"
            f"Contents of '{folder}':\n    {listing}\n\n"
            f"Check: (1) this script's working directory matches where your "
            f"Experimental_Data/Simulation_Results folders actually are, and "
            f"(2) the filename in SETTINGS matches exactly what's listed above."
        )
    return filepath

# =========================================================
# CONFIGURATION
# =========================================================
SETTINGS = {
    # Folders the required files are read from - update these if your
    # local folder names differ. "real_file" is expected to be Step_00's
    # OUTPUT format: a plain "tau,iter_1,...,iter_9" CSV, already
    # timing-corrected and already normalized to g2(tau) - NOT the raw,
    # pre-Step_00 format (APD1/APD2 header rows + raw counts). If you
    # instead want to plot directly from a raw, unprocessed file, the
    # real-data loading section below would need to do its own shift
    # detection and normalization instead (see Step_00 for that logic).
    # NOTE ON FILENAME: Step_00 only rewrites the "_n<N>.csv" suffix of a
    # file if its classification changed - anything before that (including
    # a prefix like "exp_g2_raw_") is left exactly as in the original
    # filename, so check your own Experimental_Data folder's actual
    # listing rather than assuming the prefix was stripped.
    "experimental_data_dir": "Experimental_Data",
    "simulation_data_dir": "Simulation_Results",

    "real_file": "exp_g2_raw_rep_1_n1.csv",
    # Any simulated N=1 file whose injected background fraction is close to
    # the calibrated value of the real file (see training_labels.csv and
    # experimental_calibration.csv).
    "sim_file": "g2_raw_rep_28_n1.csv",
    # Mapping lookup written by Compute_iter_to_integration_time_mapping.py;
    # looked for in the current working directory (it is a derived file, not
    # raw simulation output) - update this path if you keep it elsewhere.
    "lookup_csv": "iter_to_time_mapping_lookup.csv",

    # Real iterations to show (indices into the 9-point real acquisition
    # schedule, i.e. 1, 2, 5, 10, 20, 50, 100, 200, 500 s). The matched
    # simulated iterations are NOT typed in by hand: they are looked up
    # from the mapping curve (see sim_iterations_for_times below), so the
    # bottom row always corresponds to the same integration times as the
    # top row. Real times must lie within the range covered by the
    # mapping (the 300 simulated iterations correspond to about 79 s, so
    # the 100 s trace (iter_7) cannot be matched: use iter_6 = 50 s).
    # Real iter_1 (1 s) is not used: at that sparsity ANY N=1 file's far-tail
    # bins hold only 0-2 raw counts (a structural property of the thinning
    # schedule, not specific to one file), which discretizes g2(tau) into a
    # handful of levels rather than looking like continuous noise. iter_2
    # (2 s) avoids this while remaining clearly in the sparse regime.
    "real_iterations_to_plot": [2, 4, 6],          # 2, 10, 50 s
    "sim_iterations_to_plot": None,                # None = look up from the mapping curve
    "real_integration_times_s": [1, 2, 5, 10, 20, 50, 100, 200, 500],

    "far_tail_ns": 30.0,       # far-field region used for sim normalization
    "plot_window_ns": 50.0,    # tau axis window shown in every left panel

    "real_color": "#1f4e8c",
    "sim_color": "#c0392b",

    "save_path": "figure_real_vs_simulated_with_mapping.png",
    "dpi": 300,
    "show_plot": True,         # only has an effect if matplotlib uses an interactive backend
}


# =========================================================
# REAL DATA LOADING
# =========================================================
def load_real_file(filepath):
    """
    Loads a file already processed by Step_00: a plain CSV with header
    row "tau,iter_1,...,iter_9", where every iter_ column is ALREADY
    normalized to g2(tau) and the tau axis is ALREADY timing-corrected
    (re-centered so the dip sits at tau=0) - no further processing is
    needed here, unlike the raw, pre-Step_00 format.
    """
    df = pd.read_csv(filepath)
    tau = df["tau"].values
    g2_df = df.drop(columns=["tau"])
    return tau, g2_df


# =========================================================
# SIMULATED DATA LOADING / NORMALIZATION
# =========================================================
def load_sim_file(filepath):
    df = pd.read_csv(filepath)
    tau = df["tau"].values
    counts_df = df.drop(columns=["tau"])
    return tau, counts_df


def normalize_g2_sim(counts, tau, far_tail_ns):
    far_m = np.abs(tau) >= far_tail_ns
    baseline = counts[far_m].mean()
    if baseline <= 0:
        return np.full_like(counts, np.nan)
    return counts / baseline


# =========================================================
# MAPPING: real integration time -> simulated iteration
# =========================================================
def sim_iterations_for_times(lookup, real_times_s):
    """For each real integration time, return the simulated iteration whose
    mapped real time (representative_real_time_s in the lookup table) is
    closest to it. The mapping covers a limited range (the last simulated
    iteration corresponds to about 79 s), so a real time beyond it cannot
    be matched and raises an error rather than being silently clipped."""
    it = lookup["iteration"].values.astype(int)
    t = np.maximum.accumulate(lookup["representative_real_time_s"].values.astype(float))
    extrap = lookup["fraction_extrapolated"].values.astype(float) if "fraction_extrapolated" in lookup else np.zeros(len(t))
    out = []
    for target in real_times_s:
        if target > t[-1]:
            raise ValueError(
                f"Real time {target:g} s is beyond the range covered by the mapping "
                f"(maximum {t[-1]:.1f} s at iteration {it[-1]}). Choose a shorter real "
                f"iteration (e.g. 50 s) or extend the simulation.")
        idx = int(np.searchsorted(t, target, side="left"))
        cands = [j for j in (idx - 1, idx) if 0 <= j < len(t)]
        j = min(cands, key=lambda k: abs(t[k] - target))
        note = "  (mapping flagged as extrapolated in the lookup table)" if extrap[j] > 0 else ""
        print(f"  real {target:g} s  ->  simulated iteration {it[j]}  (mapped time {t[j]:.2f} s){note}")
        out.append(int(it[j]))
    return out


# =========================================================
# FIGURE
# =========================================================
def make_figure(settings):
    # --- Real data (already timing-corrected and normalized by Step_00) ---
    real_filepath = require_file(os.path.join(settings["experimental_data_dir"], settings["real_file"]))
    tau_r, g2_r = load_real_file(real_filepath)
    window_r = np.abs(tau_r) <= settings["plot_window_ns"]

    # --- Simulated data (no timing offset - simulation has none by construction) ---
    sim_filepath = require_file(os.path.join(settings["simulation_data_dir"], settings["sim_file"]))
    tau_s, counts_s = load_sim_file(sim_filepath)
    window_s = np.abs(tau_s) <= settings["plot_window_ns"]

    # --- Mapping lookup, for the right panel ---
    lookup = pd.read_csv(require_file(settings["lookup_csv"]))

    real_iters = settings["real_iterations_to_plot"]
    times_s = settings["real_integration_times_s"]
    sim_iters = settings["sim_iterations_to_plot"]
    if sim_iters is None:
        print("Matching real integration times to simulated iterations:")
        sim_iters = sim_iterations_for_times(lookup, [times_s[it - 1] for it in real_iters])
    missing = [it for it in sim_iters if f"iter_{it}" not in counts_s.columns]
    if missing:
        raise KeyError(f"Simulated file has no column(s) iter_{missing}")
    real_color, sim_color = settings["real_color"], settings["sim_color"]

    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.linewidth": 0.9,
        "mathtext.fontset": "cm",
    })

    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(nrows=2, ncols=4, width_ratios=[1, 1, 1, 1.3],
                           wspace=0.38, hspace=0.45)

    # --- Left: real data, top row ---
    for i, it in enumerate(real_iters):
        ax = fig.add_subplot(gs[0, i])
        col_idx = it - 1
        g2 = g2_r[f"iter_{it}"].values
        ax.plot(tau_r[window_r], g2[window_r], "-", color=real_color, linewidth=0.9)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)
        ax.set_title(rf"$t_{{\mathrm{{int}}}}$ = {times_s[col_idx]:g} s", fontsize=11)
        ax.set_xlim(-settings["plot_window_ns"], settings["plot_window_ns"])
        ax.tick_params(direction="in", top=True, right=True, labelbottom=False)
        ax.grid(alpha=0.2, linewidth=0.6)
        if i == 0:
            ax.set_ylabel(r"$g^{(2)}(\tau)$")

    # --- Left: simulated data, bottom row ---
    for i, it in enumerate(sim_iters):
        ax = fig.add_subplot(gs[1, i])
        g2 = normalize_g2_sim(counts_s[f"iter_{it}"].values, tau_s, settings["far_tail_ns"])
        ax.plot(tau_s[window_s], g2[window_s], "-", color=sim_color, linewidth=0.9)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)
        ax.set_title(f"iter_{it}", fontsize=11)
        ax.set_xlabel(r"Delay time $\tau$ (ns)")
        ax.set_xlim(-settings["plot_window_ns"], settings["plot_window_ns"])
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(alpha=0.2, linewidth=0.6)
        if i == 0:
            ax.set_ylabel(r"$g^{(2)}(\tau)$")

    # --- Right: iter_# <-> real time mapping, spanning both rows ---
    ax_map = fig.add_subplot(gs[:, 3])
    ax_map.plot(lookup["iteration"], lookup["representative_real_time_s"], "-",
                color="#333333", linewidth=1.6)
    ax_map.fill_between(lookup["iteration"], lookup["representative_real_time_s_iqr_low"],
                         lookup["representative_real_time_s_iqr_high"], alpha=0.2,
                         color="#333333", label="IQR across files")
    ax_map.set_yscale("log")
    ax_map.set_xlabel("Simulated iteration")
    ax_map.set_ylabel("Equivalent real integration time (s)")
    ax_map.set_title("iter_# $\\leftrightarrow$ real time", fontsize=11)
    ax_map.grid(alpha=0.25, which="both")
    ax_map.tick_params(direction="in", top=True, right=True)

    # Mark the three simulated iterations used on the left with a vertical
    # line and a marker, so the two halves of the figure are visually tied
    # together even without text labels (add your own labels manually).
    for i, it in enumerate(sim_iters):
        t_mapped = lookup.loc[lookup.iteration == it, "representative_real_time_s"].values[0]
        ax_map.axvline(it, color=sim_color, linestyle=":", linewidth=1.1, alpha=0.8)
        ax_map.plot(it, t_mapped, "o", color=sim_color, markersize=5, zorder=5)
    ax_map.legend(loc="lower right", fontsize=8)

    plt.savefig(settings["save_path"], dpi=settings["dpi"], bbox_inches="tight")
    print(f"Saved: {settings['save_path']}")
    # plt.show() only works with an interactive backend; with a non-interactive
    # one (e.g. Agg, when running headless or from some IDEs) it just emits
    # "FigureCanvasAgg is non-interactive" - the figure is already saved above.
    backend = matplotlib.get_backend().lower()
    non_interactive = ("agg", "pdf", "ps", "svg", "cairo", "template")
    if settings.get("show_plot", True) and not any(b in backend for b in non_interactive):
        plt.show()
    plt.close(fig)


# =========================================================
# EXECUTION
# =========================================================
if __name__ == "__main__":
    make_figure(SETTINGS)