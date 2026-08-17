# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:43:42 2026

@author: carlo
"""
"""
Plot_g2_at_iterations_column.py

Same figure as Plot_g2_at_iterations.py, but panels are stacked in ONE
COLUMN (vertically) rather than in one row (side by side) - useful for a
narrow single-column figure placement.

Publication-quality figure: g2(tau) for ONE real experimental file, shown
at several specified iter_# columns (integration times) - the same
"sparse to clean" panel convention used in prior literature (e.g.
Kudyshev et al. 2020, Fig. 1b).

Two things this does that a naive "just plot the raw column" script would
miss:
  1. NORMALIZATION: exp_g2_raw_*.csv files contain RAW coincidence counts,
     not g2(tau) - each panel is normalized separately using the
     accidental-coincidence baseline formula (APD1_rate * APD2_rate *
     bin_width * integration_time), matching Step_0's validated approach
     (checked there against the instrument's own normalization, agreement
     within 1-3%).
  2. TIMING OFFSET CORRECTION: real files have a fixed, small hardware
     timing offset between the two detector channels (typically ~18ns),
     so the antibunching dip does NOT sit at tau=0 in the raw data. This
     is detected once (from the file's best-available, most-converged
     iteration) and the SAME shift is applied to every panel, since it's
     a fixed hardware property, not something that changes with
     integration time - matches Step_0's approach exactly.

LAYOUT NOTE: with panels stacked vertically instead of side by side, the
axis-label convention flips relative to the row version: here, every
panel gets its OWN y-label (since each panel's y-range can differ
substantially - a sparse, noisy panel and a clean, converged one do not
share a sensible common scale), while only the BOTTOM-most panel gets an
x-axis label and visible tick labels (since all panels share the same
delay-time axis, repeating it at every panel would be redundant).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION
# =========================================================
SETTINGS = {
    "filepath": "exp_g2_raw_rep_1_n1.csv",
    "iterations_to_plot": [1, 4, 9],

    # Known real acquisition schedule (seconds), index-matched to
    # iter_1..iter_9 - must match Step_0/Step_01's schedule.
    "real_integration_times_s": [1, 2, 5, 10, 20, 50, 100, 200, 500],

    "bin_width_ns": 0.1,

    # Timing-shift detection (mirrors Step_0): searched using the LAST
    # (best signal-to-noise) iteration, in this window.
    "shift_search_window_ns": (2.0, 50.0),
    "shift_smoothing_bins": 15,

    # Plot window after re-centering - wide enough to show the dip plus a
    # clear flat baseline for context, without the full noisy tail
    # dominating the figure.
    "plot_window_ns": 50.0,

    "save_path": "g2_at_iterations_column.png",
    "dpi": 300,
    # Per-panel size (width, height) - width stays fixed as panels stack
    # downward, unlike the row version where height stays fixed as panels
    # extend sideways.
    #"figsize_per_panel": (4.6, 2.6),
    "figsize_per_panel": (5, 3),
}


# =========================================================
# LOADING
# =========================================================
def load_real_file(filepath):
    """Returns (tau, counts_df, apd1_rates, apd2_rates). counts_df has one
    column per iter_#, raw (unnormalized) coincidence counts."""
    raw = pd.read_csv(filepath, header=None)
    apd1_rates = raw.iloc[0, 1:].astype(float).values
    apd2_rates = raw.iloc[1, 1:].astype(float).values
    header_row = raw.iloc[2, :].values

    data = raw.iloc[3:, :].reset_index(drop=True)
    data.columns = header_row
    data = data.astype(float)

    tau = data["tau"].values
    counts_df = data.drop(columns=["tau"])
    return tau, counts_df, apd1_rates, apd2_rates


# =========================================================
# TIMING OFFSET DETECTION (mirrors Step_0)
# =========================================================
def detect_timing_shift(tau, counts_last_iter, settings):
    """
    Detects the dip location using the LAST (best S/N) iteration: tail-
    normalizes, smooths, and finds the minimum within the expected search
    window. Returns the shift in ns (subtract this from tau to re-center).
    """
    far_m = np.abs(tau) >= 10.0
    baseline = counts_last_iter[far_m].mean()
    normalized = counts_last_iter / baseline if baseline > 0 else counts_last_iter

    smoothed = pd.Series(normalized).rolling(
        settings["shift_smoothing_bins"], center=True, min_periods=1
    ).mean().values

    lo, hi = settings["shift_search_window_ns"]
    search_m = (tau >= lo) & (tau <= hi)
    if not search_m.any():
        return 0.0
    idx_in_window = np.argmin(smoothed[search_m])
    shift = tau[search_m][idx_in_window]
    return shift


# =========================================================
# NORMALIZATION (accidental-coincidence formula, matches Step_0)
# =========================================================
def normalize_g2(counts, apd1_rate, apd2_rate, bin_width_ns, integration_time_s):
    bin_width_s = bin_width_ns * 1e-9
    baseline = apd1_rate * apd2_rate * bin_width_s * integration_time_s
    if baseline <= 0:
        return np.full_like(counts, np.nan)
    return counts / baseline


# =========================================================
# FIGURE
# =========================================================
def make_figure(settings):
    tau, counts_df, apd1_rates, apd2_rates = load_real_file(settings["filepath"])
    iters = settings["iterations_to_plot"]
    times_s = settings["real_integration_times_s"]

    last_iter = counts_df.shape[1]
    shift_ns = detect_timing_shift(tau, counts_df[f"iter_{last_iter}"].values, settings)
    tau_corrected = tau - shift_ns
    print(f"Detected timing shift: {shift_ns:+.2f} ns (from iter_{last_iter}, applied to all panels)")

    window_m = np.abs(tau_corrected) <= settings["plot_window_ns"]

    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.linewidth": 0.9,
        "mathtext.fontset": "cm",
    })

    n_panels = len(iters)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(settings["figsize_per_panel"][0], settings["figsize_per_panel"][1] * n_panels),
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    panel_labels = "abcdefgh"

    for i, (ax, it, label) in enumerate(zip(axes, iters, panel_labels)):
        col_idx = it - 1
        counts = counts_df[f"iter_{it}"].values
        g2 = normalize_g2(counts, apd1_rates[col_idx], apd2_rates[col_idx],
                           settings["bin_width_ns"], times_s[col_idx])

        ax.plot(tau_corrected[window_m], g2[window_m], "-", color="#1f4e8c",
                 linewidth=0.9, alpha=0.9)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)

        ax.set_ylabel(r"$g^{(2)}(\tau)$")
        is_bottom_panel = (i == n_panels - 1)
        if is_bottom_panel:
            ax.set_xlabel(r"Delay time $\tau$ (ns)")
        else:
            ax.tick_params(labelbottom=False)
        ax.set_title(rf"$t_{{\mathrm{{int}}}}$ = {times_s[col_idx]:g} s", fontsize=11)
        ax.set_xlim(-settings["plot_window_ns"], settings["plot_window_ns"])
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(alpha=0.2, linewidth=0.6)

    plt.tight_layout()
    plt.savefig(settings["save_path"], dpi=settings["dpi"], bbox_inches="tight")
    print(f"Saved: {settings['save_path']}")
    plt.show()
    plt.close()


# =========================================================
# EXECUTION
# =========================================================
if __name__ == "__main__":
    make_figure(SETTINGS)