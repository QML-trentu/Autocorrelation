"""
Step_00_Correct_and_relabel_experimental_data.py

Corrects the timing offset between the two detector channels (re-centers
each file's antibunching dip to tau=0) and re-derives each file's true
emitter number N (1, 2, or >=3) from the corrected data.

Raw data files carry two header rows (APD1, APD2 - the per-iteration
single-channel count rates in Hz) before the tau/iter_ column header.
These rates are used to obtain a reliable normalization baseline:

  1. A validated baseline, rather than one re-derived from the measurement
     window itself. If a matching *_norm file is present, raw/norm gives
     the instrument's own applied baseline exactly (confirmed constant to
     6 decimal places across all bins in a file). If only the raw file is
     available, the accidental-coincidence formula
     APD1 * APD2 * bin_width * integration_time reproduces that same
     baseline to within 1-3%, and (unlike re-deriving a baseline from a
     narrow window of the measurement itself) is not biased by any
     residual bunching present in that window.
  2. Proper Poisson-weighted least-squares fitting, using genuine raw
     integer counts (sigma = sqrt(raw_counts)/baseline), which correctly
     down-weights noisier, lower-count bins.

For each file (matched as *_raw.csv) in OLD_DIR:
  1. Detects and applies the timing-offset correction, cropping to
     +/-TAU_LIMIT_NS, with a cross-file median-based outlier safeguard on
     the detected shift itself (see the "cross-file consistency check"
     below for why).
  2. Computes g2(0) two ways on the corrected data's last iteration (best
     signal-to-noise): a weighted fit (3-parameter model - see g2_model's
     docstring for why there is no tau2 term) and a direct window average.
     Classifies via the classical ideal-floor thresholds, preferring the
     fit when it converges.
  3. Excludes files whose classification-relevant value falls within
     BORDERLINE_MARGIN of a threshold boundary, since these are genuinely
     ambiguous rather than confidently one class or the other.
  4. Saves survivors under a filename reflecting their newly determined N
     (standardized to "_n3" for the whole N>=3 bucket, since the threshold
     scheme cannot distinguish 3 from 4 from 5 emitters). Two files are
     written per survivor:
       - "..._n<N>.csv" - tau + iter_ columns of properly-normalized g2
         values, at all 9 iterations.
       - "..._n<N>_rawcounts.csv" - a companion file with the same
         shifted/cropped data as raw (unnormalized) counts, so downstream
         scripts can reconstruct a proper Poisson sigma for weighted
         fitting.

Filenames preserve the same rep_<X> identifier. Reads from OLD_DIR, writes
to NEW_DIR, and prints a full audit log of every decision.
"""
import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# =========================================================
# CONFIG
# =========================================================
OLD_DIR = "Experimental_Data_Raw"
NEW_DIR = "Experimental_Data"

TAU_LIMIT_NS = 100.0
SEARCH_WINDOW_NS = (2.0, 50.0)
SMOOTHING_WINDOW = 15
OUTLIER_THRESHOLD_NS = 5.0

DIP_WINDOW_NS = 1.0     # direct g2(0) estimate: mean within |tau| < this
BASE_WINDOW_NS = 10.0   # used only to normalize the signal for LOCATING the
                         # dip (detect_shift) - the baseline used to compute
                         # g2 itself comes from get_baseline_per_iter below

THRESHOLD_1_2 = 0.5
THRESHOLD_2_3 = 2.0 / 3.0
BORDERLINE_MARGIN = 0.05

# Integration times (seconds) for iter_1..iter_9, used as a fallback to
# estimate the baseline via APD1*APD2*bin_width*T when no matching *_norm
# file is available (reproduces the instrument's own normalization to
# within 1-3% - see module docstring). Update this if the acquisition
# schedule ever changes.

ITERATION_INTEGRATION_TIMES_S = {1: 1, 2: 2, 3: 5, 4: 10, 5: 20, 6: 50, 7: 100, 8: 200, 9: 500}
BIN_WIDTH_S = 0.1e-9  # 0.1 ns, matching tau's units


def get_iter_columns(df):
    cols = [c for c in df.columns if c.startswith("iter_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def load_experimental_file(raw_filepath):
    """
    Parses the file format: 2 header rows (APD1, APD2 rates per iteration),
    then the real 'tau, iter_1, ...' column header, then the data.
    Falls back to plain single-header reading if the APD rows aren't
    present.

    Returns (tau, raw_df, iter_cols, apd1_rates, apd2_rates), where the
    APD rate dicts are None if this file didn't have them.
    """
    probe = pd.read_csv(raw_filepath, header=None, nrows=3)
    looks_new_style = str(probe.iloc[2, 0]).strip().lower() == "tau"

    if not looks_new_style:
        df = pd.read_csv(raw_filepath)
        iter_cols = get_iter_columns(df)
        return df["tau"].values, df, iter_cols, None, None

    iter_cols = list(probe.iloc[2].values[1:])
    apd1_rates = {col: float(v) for col, v in zip(iter_cols, probe.iloc[0].values[1:])}
    apd2_rates = {col: float(v) for col, v in zip(iter_cols, probe.iloc[1].values[1:])}

    df = pd.read_csv(raw_filepath, header=None, skiprows=3)
    df.columns = ["tau"] + iter_cols
    df = df.astype(float)

    return df["tau"].values, df, iter_cols, apd1_rates, apd2_rates


def get_baseline_per_iter(raw_filepath, iter_cols, apd1_rates, apd2_rates, raw_df):
    """
    Baseline priority, per iteration column:
      1. Matching *_norm file present -> raw/norm ratio (exact match to
         whatever the instrument actually applied).
      2. APD1/APD2 rates present -> accidental-coincidence formula
         (validated to ~1-3% - see module docstring).
      3. Neither available -> old narrow-window fallback, printed as a
         warning since it's known to be biased when bunching is present.

    Returns (baseline_per_iter dict, source_per_iter dict).
    """
    baseline = {}
    source = {}

    norm_path = re.sub(r"_raw\.csv$", "_norm.csv", raw_filepath)
    norm_df = None
    if os.path.exists(norm_path) and norm_path != raw_filepath:
        _, norm_df, _, _, _ = load_experimental_file(norm_path)

    tau = raw_df["tau"].values
    base_m = np.abs(tau) >= BASE_WINDOW_NS

    for col in iter_cols:
        if norm_df is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = raw_df[col].values / norm_df[col].values
            valid = np.isfinite(ratio) & (norm_df[col].values != 0)
            if np.any(valid):
                baseline[col] = float(np.median(ratio[valid]))
                source[col] = "norm_file"
                continue

        if apd1_rates is not None:
            it_num = int(col.split("_")[1])
            T = ITERATION_INTEGRATION_TIMES_S.get(it_num)
            if T is not None:
                baseline[col] = apd1_rates[col] * apd2_rates[col] * BIN_WIDTH_S * T
                source[col] = "apd_formula"
                continue

        print(f"  [warn] {os.path.basename(raw_filepath)}/{col}: no norm file or APD rates - "
              f"falling back to narrow-window baseline (may be biased if bunching is present)")
        baseline[col] = float(raw_df[col].values[base_m].mean())
        source[col] = "fallback_window (less reliable)"

    return baseline, source


def detect_shift(tau, counts, base_m, search_window_ns, smoothing_window):
    baseline = counts[base_m].mean()
    if baseline <= 0:
        return None, None
    norm = counts / baseline
    smoothed = pd.Series(norm).rolling(smoothing_window, center=True, min_periods=1).mean().values
    search_mask = (tau >= search_window_ns[0]) & (tau <= search_window_ns[1])
    if not np.any(search_mask):
        return None, None
    search_indices = np.where(search_mask)[0]
    local_argmin = np.argmin(smoothed[search_indices])
    dip_index = search_indices[local_argmin]
    return tau[dip_index], dip_index


def shift_and_crop_file(tau, columns_dict, dip_index, tau_limit_ns, bin_w):
    n_half = int(round(tau_limit_ns / bin_w))
    start = dip_index - n_half
    end = dip_index + n_half + 1
    if start < 0 or end > len(tau):
        return None, None
    new_tau = tau[start:end] - tau[dip_index]
    new_columns = {col: arr[start:end] for col, arr in columns_dict.items()}
    return new_tau, new_columns


def g2_model(t, A1, tau1, A2):
    """
    3-parameter model (no tau2, offset fixed at 1.0) - see the module
    docstring in Step_01_Extract_Params_from_experimental_data.py for the
    full reasoning: bunching only decays at the microsecond scale, so
    within our +/-100ns window it's an effectively constant elevation (A2),
    not a resolvable decay, and the instrument's own normalization already
    establishes the true baseline (offset=1) at proper microsecond-scale
    reference delays.
    """
    t = np.abs(t)
    return 1.0 - A1 * np.exp(-t / tau1) + A2


def fit_g2(tau, g2, sigma=None):
    """
    If sigma is given, performs a Poisson-weighted least-squares fit:
    bins with fewer real counts are noisier and are correctly given less
    influence on the fit. Falls back to an unweighted fit if no sigma is
    available.
    """
    mask = np.isfinite(g2)
    tau_fit, g2_fit_data = tau[mask], g2[mask]
    sigma_fit = sigma[mask] if sigma is not None else None

    if len(g2_fit_data) < 10:
        return None

    A1_guess = 1 - np.min(g2_fit_data)
    p0 = [A1_guess, 0.5, max(0, np.max(g2_fit_data) - 1)]
    bounds = ([0, 0.01, 0], [2, 15, 2])

    try:
        if sigma_fit is not None:
            popt, _ = curve_fit(g2_model, tau_fit, g2_fit_data, p0=p0, bounds=bounds,
                                  sigma=sigma_fit, absolute_sigma=True, maxfev=20000)
        else:
            popt, _ = curve_fit(g2_model, tau_fit, g2_fit_data, p0=p0, bounds=bounds, maxfev=20000)
        return popt
    except Exception:
        return None


def classify(g2_zero):
    if g2_zero < THRESHOLD_1_2:
        return 1
    elif g2_zero < THRESHOLD_2_3:
        return 2
    else:
        return 3  # standardized label for the whole N>=3 bucket


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    os.makedirs(NEW_DIR, exist_ok=True)

    pattern = re.compile(r"rep_(\d+)_n(\d+)(?:_raw)?\.csv$")
    files = [f for f in sorted(os.listdir(OLD_DIR))
             if pattern.search(f) and not f.endswith("_norm.csv")]
    if not files:
        raise ValueError(f"No raw files matching rep_<N>_n<M>[_raw].csv found in {OLD_DIR}")

    # ---- Pass 1: load + detect timing shift for every file ----
    detections = {}
    for f in files:
        filepath = os.path.join(OLD_DIR, f)
        tau, raw_df, iter_cols, apd1_rates, apd2_rates = load_experimental_file(filepath)
        bin_w = round(tau[1] - tau[0], 6)
        base_m = np.abs(tau) >= BASE_WINDOW_NS

        baseline_per_iter, baseline_source = get_baseline_per_iter(
            filepath, iter_cols, apd1_rates, apd2_rates, raw_df
        )

        last_counts = raw_df[iter_cols[-1]].values.astype(float)
        shift_ns, dip_index = detect_shift(tau, last_counts, base_m, SEARCH_WINDOW_NS, SMOOTHING_WINDOW)

        columns_dict = {col: raw_df[col].values.astype(float) for col in iter_cols}
        detections[f] = [shift_ns, dip_index, tau, columns_dict, bin_w, iter_cols,
                          baseline_per_iter, baseline_source]

    # ---- Cross-file consistency check for the shift itself ----
    valid_shifts = [v[0] for v in detections.values() if v[0] is not None]
    median_shift = np.median(valid_shifts) if valid_shifts else None
    if median_shift is not None:
        print(f"Median detected timing shift across {len(valid_shifts)} files: {median_shift:.3f} ns\n")
        for f, rec in detections.items():
            shift_ns = rec[0]
            if shift_ns is None:
                continue
            if abs(shift_ns - median_shift) > OUTLIER_THRESHOLD_NS:
                tau = rec[2]
                print(f"  [shift override] {f}: individual detection {shift_ns:+.2f} ns is "
                      f">{OUTLIER_THRESHOLD_NS} ns from the group median - using median instead")
                new_dip_index = int(np.argmin(np.abs(tau - median_shift)))
                detections[f][0] = tau[new_dip_index]
                detections[f][1] = new_dip_index

    # ---- Pass 2: shift+crop, re-fit (weighted), reclassify, and save ----
    summary_rows = []
    for f in files:
        (shift_ns, dip_index, tau, columns_dict, bin_w, iter_cols,
         baseline_per_iter, baseline_source) = detections[f]
        orig_match = pattern.search(f)
        orig_n = int(orig_match.group(2))

        if dip_index is None:
            print(f"  [excluded] {f}: could not detect a timing shift")
            summary_rows.append([f, None, orig_n, None, None, None, None, "", "excluded: no shift detected"])
            continue

        new_tau, new_columns = shift_and_crop_file(tau, columns_dict, dip_index, TAU_LIMIT_NS, bin_w)
        if new_tau is None:
            print(f"  [excluded] {f}: shift ({shift_ns:.2f} ns) too large for +/-{TAU_LIMIT_NS} ns output")
            summary_rows.append([f, shift_ns, orig_n, None, None, None, None, "", "excluded: shift too large"])
            continue

        last_col = iter_cols[-1]
        raw_counts_new = new_columns[last_col]
        baseline = baseline_per_iter[last_col]

        dip_m_new = np.abs(new_tau) <= DIP_WINDOW_NS
        g2_new = raw_counts_new / baseline
        sigma_new = np.sqrt(np.maximum(raw_counts_new, 1)) / baseline

        direct_g2_0 = float(g2_new[dip_m_new].mean())

        popt = fit_g2(new_tau, g2_new, sigma=sigma_new)
        if popt is not None:
            A1, tau1, A2 = popt
            fit_g2_0 = float(1.0 - A1 + A2)
            bound_note = f"A1_bound={A1 > 1.98}, baseline_source={baseline_source[last_col]}"
        else:
            fit_g2_0 = None
            bound_note = f"fit failed, baseline_source={baseline_source[last_col]}"

        direct_N = classify(direct_g2_0)
        fit_N = classify(fit_g2_0) if fit_g2_0 is not None else None

        value_for_classification = fit_g2_0 if fit_g2_0 is not None else direct_g2_0
        near_boundary = (
            abs(value_for_classification - THRESHOLD_1_2) < BORDERLINE_MARGIN
            or abs(value_for_classification - THRESHOLD_2_3) < BORDERLINE_MARGIN
        )

        if near_boundary:
            print(f"  [excluded - borderline] {f}: g2(0)={value_for_classification:.3f} is within "
                  f"{BORDERLINE_MARGIN} of a threshold boundary (0.5 or {THRESHOLD_2_3:.3f})")
            summary_rows.append([f, shift_ns, orig_n, fit_g2_0, direct_g2_0, fit_N, direct_N,
                                  bound_note, "excluded: too close to threshold boundary"])
            continue

        final_N = classify(value_for_classification)
        new_filename = re.sub(r"_n\d+(?:_raw)?\.csv$", f"_n{final_N}.csv", f)

        # Primary output: tau + iter_ columns of properly-normalized g2
        # values, at all 9 iterations - the standard format every
        # downstream script in this pipeline expects.
        out_df = pd.DataFrame({"tau": new_tau})
        for col in iter_cols:
            out_df[col] = new_columns[col] / baseline_per_iter[col]
        out_df.to_csv(os.path.join(NEW_DIR, new_filename), index=False)

        # Companion file: the same shifted/cropped data as raw
        # (unnormalized) counts - lets downstream scripts reconstruct a
        # proper Poisson sigma (sqrt(raw)/baseline) for weighted fitting.
        # The "_rawcounts" suffix deliberately does not match the plain
        # rep_<N>_n<M>.csv pattern other scripts search for, so it is
        # ignored by anything that doesn't explicitly look for it.
        rawcounts_filename = re.sub(r"\.csv$", "_rawcounts.csv", new_filename)
        raw_out_df = pd.DataFrame({"tau": new_tau})
        for col in iter_cols:
            raw_out_df[col] = new_columns[col]
        raw_out_df.to_csv(os.path.join(NEW_DIR, rawcounts_filename), index=False)

        relabel_note = "N unchanged" if final_N == orig_n else f"N changed {orig_n}->{final_N}"
        status = f"OK -> {new_filename} + {rawcounts_filename} ({relabel_note})"
        fit_str = f"{fit_g2_0:.3f}" if fit_g2_0 is not None else "n/a"
        print(f"  {f}: shift={shift_ns:+.2f}ns, direct_g2(0)={direct_g2_0:.3f}, fit_g2(0)={fit_str} "
              f"-> N={final_N}  [{relabel_note}]  ({bound_note})")

        summary_rows.append([f, shift_ns, orig_n, fit_g2_0, direct_g2_0, fit_N, direct_N, bound_note, status])

    print("\n=== Full audit log ===")
    summary_df = pd.DataFrame(summary_rows, columns=[
        "original_file", "shift_ns", "original_N", "fit_g2_0", "direct_g2_0",
        "fit_N", "direct_N", "bound_flags", "status"
    ])
    print(summary_df.to_string(index=False))

    n_excluded = summary_df["status"].str.startswith("excluded").sum()
    n_kept = len(summary_df) - n_excluded
    n_relabeled = summary_df["status"].str.contains("N changed", na=False).sum()
    print(f"\n{n_kept} files kept, {n_excluded} excluded, {n_relabeled} relabeled to a different N than their "
          f"original filename.")