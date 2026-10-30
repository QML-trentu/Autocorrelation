# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:21:34 2026

@author: carlo
"""

# -*- coding: utf-8 -*-
"""
Compute_iter_to_integration_time_mapping.py

Maps Step_02's simulated iter_# axis to an approximate equivalent REAL
integration time, so a reader can interpret "iteration 150" as "roughly
equivalent to N seconds of real acquisition."

WHY BASELINE COUNT RATE, NOT ELAPSED TIME DIRECTLY: iter_# is just a
Monte Carlo chunk counter with no inherent time unit, and different real
emitters accumulate counts at different rates (brighter emitters reach
the same statistics faster), so there's no single universal "iter_# =
X seconds" fact. What actually governs how noisy or clean a g2(tau)
measurement looks - in both real and simulated data - is the accumulated
BASELINE COUNT RATE (counts/bin in the flat far-delay region). This is
computable in both worlds on equal footing:
    - Real: at each of the instrument's known integration times, via the
      accidental-coincidence formula APD1_rate * APD2_rate * bin_width *
      T (the same formula Step_00 uses, validated there against the
      instrument's own normalization).
    - Simulated: the actual far-tail region mean at each iter_#, read
      directly from the raw counts.
Matching these two curves gives the lookup.

HONEST LIMITATION: this gives a REPRESENTATIVE correspondence, not a
universal exact one. Real emitters vary in brightness, so a dim emitter
reaches a given baseline level slower (in real seconds) than a bright
one, even though both would map to the same simulated iter_#. This script
reports the mapping from EACH real file individually AND a pooled
median, so the spread across real emitters is visible rather than
hidden.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION
# =========================================================
SETTINGS = {
    "real_data_dir": "Experimental_Data_Raw",   # original raw files (with the APD1/APD2 rate rows)
    "simulated_data_dir": "Simulation_Results",  # Step_02 output (g2_raw_rep_*.csv)

    # Real filenames are matched by this prefix; simulated files are
    # anything NOT matching this prefix but matching rep_<id>_n<N>.csv.
    "real_file_prefix": "exp_",

    # Known integration-time schedule for the real acquisition (seconds) -
    # must match Step_00/Step_01's hardcoded schedule. Update here if your
    # acquisition schedule ever changes.
    "real_integration_times_s": [1, 2, 5, 10, 20, 50, 100, 200, 500],

    "bin_width_ns": 0.1,

    # Far-tail region used to estimate the baseline count rate - kept
    # conservative (well clear of the dip AND any uncorrected ~18ns real
    # timing offset) so this works directly on RAW real files without
    # needing Step_00's full timing correction first.
    "far_tail_ns": 30.0,
}

FILENAME_PATTERN = re.compile(r"rep_(\d+)_n(\d+)\.csv$")


# =========================================================
# REAL DATA: baseline count rate vs. known integration time
# =========================================================
def compute_real_baseline_curve(filepath):
    """
    Returns (times_s, baseline_rate_per_bin) for one real raw file, using
    the accidental-coincidence formula (APD1_rate * APD2_rate * bin_width
    * T) - matches Step_00's validated baseline derivation, and only needs
    the APD header rows, not the actual counts.
    """
    raw = pd.read_csv(filepath, header=None)
    apd1_rates = raw.iloc[0, 1:].astype(float).values
    apd2_rates = raw.iloc[1, 1:].astype(float).values

    times_s = np.array(SETTINGS["real_integration_times_s"], dtype=float)
    if len(times_s) != len(apd1_rates):
        raise ValueError(
            f"{filepath}: found {len(apd1_rates)} APD rate columns but "
            f"SETTINGS['real_integration_times_s'] has {len(times_s)} entries - "
            "these must match 1:1."
        )

    bin_w_s = SETTINGS["bin_width_ns"] * 1e-9
    baseline_rate = apd1_rates * apd2_rates * bin_w_s * times_s
    return times_s, baseline_rate


# =========================================================
# SIMULATED DATA: baseline count rate vs. iter_#
# =========================================================
def get_iter_columns(df):
    cols = [c for c in df.columns if c.startswith("iter_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def compute_sim_baseline_curve(filepath, settings):
    """Returns (iters, baseline_rate_per_bin) for one simulated raw file -
    the actual far-tail region mean at each iter_#, read directly from the
    (cumulative) raw counts."""
    df = pd.read_csv(filepath)
    tau = df["tau"].values
    far_m = np.abs(tau) >= settings["far_tail_ns"]

    iter_cols = get_iter_columns(df)
    iters = np.array([int(c.split("_")[1]) for c in iter_cols])
    baseline = df[iter_cols].values[far_m, :].mean(axis=0)
    return iters, baseline


# =========================================================
# FILE DISCOVERY
# =========================================================
def discover_files(settings):
    real_dir, sim_dir = settings["real_data_dir"], settings["simulated_data_dir"]
    prefix = settings["real_file_prefix"]

    real_files = [
        os.path.join(real_dir, f) for f in sorted(os.listdir(real_dir))
        if f.startswith(prefix) and FILENAME_PATTERN.search(f) and f.endswith(".csv")
        and "_norm" not in f and "_rawcounts" not in f
    ]
    sim_files = [
        os.path.join(sim_dir, f) for f in sorted(os.listdir(sim_dir))
        if f.startswith("g2_raw_") and FILENAME_PATTERN.search(f)
    ]
    return real_files, sim_files


# =========================================================
# MAPPING
# =========================================================
def map_iter_to_time(sim_iters, sim_baseline, real_times_s, real_baseline):
    """
    For each sim_iter, find the real integration time giving the SAME
    baseline count rate, via log-log linear interpolation of the real
    curve (baseline count rates and integration times both span orders
    of magnitude, so interpolating in log-space is far more appropriate
    than linear space).

    Sim iterations whose baseline falls outside the real curve's observed
    range are extrapolated (log-log linear) rather than dropped, but
    flagged in the returned mask so this can be reported honestly.
    """
    log_real_baseline = np.log10(real_baseline)
    log_real_times = np.log10(real_times_s)
    order = np.argsort(log_real_baseline)

    log_sim_baseline = np.log10(np.maximum(sim_baseline, 1e-12))
    log_time_mapped = np.interp(
        log_sim_baseline, log_real_baseline[order], log_real_times[order]
    )
    time_mapped_s = 10 ** log_time_mapped

    in_range = (sim_baseline >= real_baseline.min()) & (sim_baseline <= real_baseline.max())
    return time_mapped_s, in_range


def build_full_mapping(settings):
    real_files, sim_files = discover_files(settings)
    if not real_files:
        raise FileNotFoundError(f"No real files found (prefix '{settings['real_file_prefix']}') in {settings['real_data_dir']}")
    if not sim_files:
        raise FileNotFoundError(f"No simulated g2_raw_rep_*.csv files found in {settings['simulated_data_dir']}")

    print(f"Found {len(real_files)} real file(s), {len(sim_files)} simulated file(s).")

    real_curves = {}
    for f in real_files:
        times_s, baseline = compute_real_baseline_curve(f)
        real_curves[os.path.basename(f)] = (times_s, baseline)
        print(f"  Real {os.path.basename(f)}: baseline rate spans "
              f"{baseline.min():.3g} - {baseline.max():.3g} counts/bin "
              f"over {times_s.min():.0f}-{times_s.max():.0f}s")

    # Use the FIRST simulated file to define the reference iter_# axis
    # (Step_02 files should all share the same axis/length).
    sim_iters, _ = compute_sim_baseline_curve(sim_files[0], settings)

    per_file_mappings = {}
    for f in sim_files:
        iters, sim_baseline = compute_sim_baseline_curve(f, settings)
        per_real_mapping = {}
        for real_name, (times_s, real_baseline) in real_curves.items():
            time_mapped_s, in_range = map_iter_to_time(iters, sim_baseline, times_s, real_baseline)
            per_real_mapping[real_name] = time_mapped_s
        per_file_mappings[os.path.basename(f)] = (iters, sim_baseline, per_real_mapping)
        n_out = sum(1 for real_name in real_curves
                    if not map_iter_to_time(iters, sim_baseline, *real_curves[real_name])[1].all())
        print(f"  Simulated {os.path.basename(f)}: baseline rate spans "
              f"{sim_baseline.min():.3g} - {sim_baseline.max():.3g} counts/bin")

    return real_curves, per_file_mappings, sim_iters


def build_summary_table(real_curves, per_file_mappings, sim_iters):
    """
    One row per (simulated file, iter_#), with the mapped time from EACH
    real file plus the across-real-files median - so the spread across
    real emitters is visible, not hidden behind a single number. Also
    flags, per real file, whether that iteration's mapping required
    extrapolating beyond the real data's own observed baseline range
    (honest, not swept under the rug - see module docstring).
    """
    real_names = list(real_curves.keys())
    rows = []
    for sim_name, (iters, sim_baseline, per_real_mapping) in per_file_mappings.items():
        stacked = np.stack([per_real_mapping[name] for name in real_names], axis=1)
        median_time_s = np.median(stacked, axis=1)

        in_range_flags = {}
        for name in real_names:
            times_s, real_baseline = real_curves[name]
            _, in_range = map_iter_to_time(iters, sim_baseline, times_s, real_baseline)
            in_range_flags[name] = in_range

        for i, it in enumerate(iters):
            row = {"sim_file": sim_name, "iteration": it, "sim_baseline_counts_per_bin": sim_baseline[i]}
            for j, name in enumerate(real_names):
                row[f"mapped_time_s__{name}"] = stacked[i, j]
                row[f"extrapolated__{name}"] = not in_range_flags[name][i]
            row["mapped_time_s__median_across_real_files"] = median_time_s[i]
            row["any_extrapolated"] = any(not in_range_flags[name][i] for name in real_names)
            rows.append(row)
    return pd.DataFrame(rows)


def build_pooled_lookup(summary_df, sim_iters):
    """
    Collapses the per-(sim_file, iteration) table into ONE representative
    iter_# -> real-time lookup, averaged across all simulated files too
    (not just across real files), for a single reference table/figure.
    Includes the fraction of (sim_file, real_file) combinations that
    needed extrapolation at each iteration, so this isn't hidden.
    """
    grouped = summary_df.groupby("iteration")["mapped_time_s__median_across_real_files"]
    lookup = grouped.median().reset_index()
    lookup.columns = ["iteration", "representative_real_time_s"]
    lookup["representative_real_time_s_iqr_low"] = grouped.quantile(0.25).values
    lookup["representative_real_time_s_iqr_high"] = grouped.quantile(0.75).values
    lookup["fraction_extrapolated"] = summary_df.groupby("iteration")["any_extrapolated"].mean().values
    return lookup


# =========================================================
# PLOT
# =========================================================
def plot_mapping(real_curves, per_file_mappings, lookup, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    for name, (times_s, baseline) in real_curves.items():
        ax.plot(times_s, baseline, "o-", label=f"real: {name}", alpha=0.8)
    for name, (iters, sim_baseline, _) in per_file_mappings.items():
        ax.plot(lookup["representative_real_time_s"], sim_baseline, ".", alpha=0.3,
                color="gray", markersize=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Integration time (s)")
    ax.set_ylabel("Baseline count rate (counts/bin)")
    ax.set_title("Baseline count rate vs. real integration time\n(gray: simulated files, mapped)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(lookup["iteration"], lookup["representative_real_time_s"], "-", color="tab:blue")
    ax.fill_between(lookup["iteration"], lookup["representative_real_time_s_iqr_low"],
                     lookup["representative_real_time_s_iqr_high"], alpha=0.25, color="tab:blue",
                     label="IQR across real files/sim files")
    ax.set_yscale("log")
    ax.set_xlabel("Simulated iteration (iter_#)")
    ax.set_ylabel("Representative real integration time (s)")
    ax.set_title("iter_# -> approximate real integration time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


# =========================================================
# EXECUTION
# =========================================================
if __name__ == "__main__":
    real_curves, per_file_mappings, sim_iters = build_full_mapping(SETTINGS)

    summary_df = build_summary_table(real_curves, per_file_mappings, sim_iters)
    summary_df.to_csv("iter_to_time_mapping_detailed.csv", index=False)
    print("Saved: iter_to_time_mapping_detailed.csv")

    lookup = build_pooled_lookup(summary_df, sim_iters)
    lookup.to_csv("iter_to_time_mapping_lookup.csv", index=False)
    print("Saved: iter_to_time_mapping_lookup.csv")

    print("\nRepresentative lookup (subset):")
    print(lookup.iloc[:: max(1, len(lookup) // 15)].to_string(index=False))

    n_any_extrap = (lookup["fraction_extrapolated"] > 0).sum()
    print(f"\n{n_any_extrap} of {len(lookup)} iterations had at least one (sim file, real file) "
          f"combination requiring extrapolation beyond the real data's observed baseline range "
          f"(see 'fraction_extrapolated' column and 'extrapolated__<file>' flags in the detailed CSV) - "
          f"typically the very lowest iterations, against the brightest real emitters.")

    plot_mapping(real_curves, per_file_mappings, lookup, save_path="iter_to_time_mapping.png")

    print("\nDone.")