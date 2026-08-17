# -*- coding: utf-8 -*-
"""
Bayesian_analysis_simulation_data_iterspecific.py

Sequential Bayesian classifier for binary emitter classification (N=1
"Single" vs N>=2 "Multi") from simulated g2(tau) HBT data, evaluated with
k-fold cross-validation and the same three reporting outputs used by the
LM and Encoder scripts in this project, so all three are directly
comparable.

CUMULATIVE DATA AND CHAINING: Step_02's raw histogram columns are
cumulative (iter_t = sum of chunks 1..t), not independent repeats -
confirmed both from Step_02's generation code and empirically (total
counts per iteration grow monotonically with iteration, consistent with
accumulation rather than independent regeneration). Chaining a Bayesian
posterior directly on these cumulative values would double-count
overlapping data and manufacture false confidence. We thus take the difference
between consecutive raw columns to recover the counts added by each chunk alone
- genuinely new, conditionally-independent evidence - and chain the
posterior update on those increments instead (see compute_chunk_increments
and sequential_posterior).

METHODOLOGY:
  For each of N_FOLDS stratified folds:
    - TRAIN files: (1) estimate a single representative tau1 from each
      file's well-converged last-iteration fit; (2) using that fixed tau1,
      compute every train file's per-super-chunk g2 estimate via a
      closed-form (no iterative optimizer) weighted linear fit for the
      amplitude alone, and pool by (true label, chunk index) into a
      Gaussian per class per chain position - since the reliability of a
      chunk's evidence varies systematically with how much data has
      accumulated by that point, the likelihood model is calibrated
      separately per chain position rather than pooled across the whole
      chain.
    - TEST files (held out): run the full sequential (chained) posterior
      update across all super-chunks using only this fold's learned
      tau1/hypotheses, then record predictions at the evaluated
      iterations.
  Every file is used as a held-out test file exactly once.

WHY A CLOSED-FORM ESTIMATOR, NOT A FULL NONLINEAR FIT: a full 4-parameter
fit at each super-chunk is unstable on this sparse, per-chunk data - many
fits pin at the same parameter bound regardless of true class, leaving the
two classes' learned Gaussians nearly indistinguishable. Fixing tau1
(calibrated once, from well-converged data) and solving for amplitude
alone removes the extra degrees of freedom causing that instability,
while still using the data's actual shape (rather than, say, a simple
windowed count ratio).

OUTPUT SCHEMA (identical to the LM and Encoder scripts' long_df, so the
plotting functions below produce directly comparable figures):
    file, true_label, fold, iteration, g2_zero, pred_label
(here g2_zero holds the posterior probability of "Multi" - the Bayesian
analog of the LM script's fitted g2(0): both indicate how far toward
Multi the evidence points, just on different scales.)
"""
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =========================================================
# CONFIGURATION (matches LM_analysis_Simulation_data.py's settings)
# =========================================================
SETTINGS = {
    "base_window_ns": 10.0,
    "min_base_counts": 5,

    "simulation_dir": "Simulation_Results",

    "n_folds": 5,
    "random_state": 42,

    # Reported/plotted iterations match the LM script's stride exactly, for
    # a literal side-by-side comparison. The chain itself is still computed
    # over every super-chunk internally (it must be, sequentially), only
    # the RECORDED points are subsampled (though here they coincide, since
    # super-chunks are already grouped at this same stride - see
    # aggregate_increments).
    "iteration_stride": 10,

    "default_confusion_matrix_iteration": None,  # None -> last available iteration
}


# =========================================================
# 1. CORE BAYESIAN MATHEMATICS
# =========================================================
def gaussian_log_likelihood(obs_g2, mu, sigma):
    sigma = max(sigma, 1e-6)
    return -0.5 * ((obs_g2 - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))


def get_iter_columns(df):
    cols = [c for c in df.columns if c.startswith("iter_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def compute_chunk_increments(raw_df, iter_cols):
    """
    Raw histogram columns are CUMULATIVE (iter_t = sum of chunks 1..t) -
    see module docstring. Differencing consecutive RAW columns recovers
    exactly the counts added by chunk t alone, which IS genuinely
    independent evidence (each chunk is a fresh simulation draw).
    Must be done on the raw file, never the normalized one: each
    normalized column is divided by a different baseline, so subtracting
    normalized columns does not recover a well-defined "new counts" value.
    """
    raw_counts = raw_df[iter_cols].values.astype(float)
    increments = np.empty_like(raw_counts)
    increments[:, 0] = raw_counts[:, 0]
    increments[:, 1:] = np.diff(raw_counts, axis=1)
    return increments


def aggregate_increments(increments, group_size):
    """
    Sums every group_size consecutive per-chunk increments into one
    'super-chunk' before extracting a g2 estimate from it.

    A single raw chunk (1/300th of the total run) is far too sparse for a
    stable per-chunk estimate of any kind. Aggregating into groups of 10
    (matching the reporting stride, so no information anyone would look
    at anyway is discarded) gives each update step a workable amount of
    data, while remaining fully sequential (30 chained update steps
    instead of 300, still each conditionally independent of the others).
    """
    n_total = increments.shape[1]
    n_groups = n_total // group_size
    trimmed = increments[:, : n_groups * group_size]
    grouped = trimmed.reshape(increments.shape[0], n_groups, group_size).sum(axis=2)
    return grouped


# =========================================================
# 1b. PER-SUPER-CHUNK g2(0) ESTIMATOR - CLOSED-FORM, LOW-PARAMETER
# =========================================================
# A full 4-parameter nonlinear fit at each super-chunk is unstable given
# how sparse the per-chunk data is: many fits pin at the same parameter
# bound regardless of true class, leaving the two classes' learned
# Gaussians nearly indistinguishable (mean difference smaller than either
# standard deviation). This section uses a much simpler, closed-form
# estimator instead:
#
#   1. ONCE per fold, estimate a single REPRESENTATIVE tau1 by fitting the
#      full model on each TRAIN file's LAST (most-converged, high-
#      statistics) iteration - the same well-behaved regime the LM script
#      always operates in. This is the only place the full nonlinear fit
#      is still used.
#   2. For every per-super-chunk estimate (both calibration and
#      evaluation), FIX tau1 at that representative value and solve for
#      the amplitude A1 alone. With tau1 fixed, g2(t)-1 = -A1*exp(-|t|/tau1)
#      is LINEAR in A1 - solvable by closed-form weighted least squares,
#      with no iterative optimizer and therefore no bound-pinning or
#      non-convergence failure mode at all.
def g2_biexp(tau, rho1_sq, tau1, rho2_sq, tau2):
    """g2(tau) = 1 - rho1^2*exp(-|tau|/tau1) + rho2^2*exp(-|tau|/tau2) -
    used ONLY for the one-time representative-tau1 calibration below."""
    t = np.abs(tau)
    return 1.0 - (rho1_sq * np.exp(-t / tau1)) + (rho2_sq * np.exp(-t / tau2))


def fit_g2_full(tau_axis, g2_signal, sigma=None):
    """Full 4-parameter fit - identical model/bounds to the LM script,
    used only once per fold (on well-converged data) to calibrate a
    representative tau1, not for per-super-chunk evidence."""
    p0 = [0.5, 0.5, 0.0, 3.0]
    bounds = ([0.0, 0.1, 0.0, 2.0], [1.5, 1.5, 1.0, 6.0])
    try:
        if sigma is not None:
            popt, _ = curve_fit(g2_biexp, tau_axis, g2_signal, p0=p0, bounds=bounds,
                                  sigma=sigma, absolute_sigma=True, maxfev=20000)
        else:
            popt, _ = curve_fit(g2_biexp, tau_axis, g2_signal, p0=p0, bounds=bounds, maxfev=20000)
        return popt
    except Exception:
        return None


def group_tail_normalize(counts, base_m, min_base_counts):
    """
    Same idea as LM's tail_normalize, but applied to a SINGLE aggregated
    super-chunk's increment (not a cumulative column): tail-normalizes to
    that super-chunk's own far-field baseline and propagates Poisson sigma.
    """
    base_sum = counts[base_m].sum()
    if base_sum < min_base_counts:
        return None, None, False
    baseline = counts[base_m].mean()
    norm_signal = counts / baseline
    sigma = np.sqrt(np.maximum(counts, 1)) / baseline
    return norm_signal, sigma, True


def estimate_representative_tau1(train_files, directory, settings, fallback=1.0):
    """
    Fits the full model at the LAST (most-converged) iteration of every
    TRAIN file and returns the median fitted tau1 - the single decay
    timescale used to fix tau1 in the closed-form per-super-chunk
    estimator below. Falls back to `fallback` (the midpoint of Step_02's
    tau1 sampling range, 0.5-1.5) if no train file gives a usable fit.
    """
    tau1_values = []
    for f, _ in train_files:
        raw_df = pd.read_csv(os.path.join(directory, f))
        iter_cols = get_iter_columns(raw_df)
        tau = raw_df["tau"].values
        base_m = np.abs(tau) >= settings["base_window_ns"]
        counts = raw_df[iter_cols[-1]].values.astype(float)
        norm_signal, sigma, ok = group_tail_normalize(counts, base_m, settings["min_base_counts"])
        if not ok:
            continue
        popt = fit_g2_full(tau, norm_signal, sigma)
        if popt is not None:
            tau1_values.append(popt[1])
    if len(tau1_values) == 0:
        return fallback
    return float(np.median(tau1_values))


def chunk_g2_estimates(grouped_increments, tau, base_m, tau1_fixed, min_base_counts):
    """
    Closed-form per-super-chunk g2(0): with tau1 FIXED at this fold's
    representative value, g2(t)-1 = -A1*exp(-|t|/tau1) is linear in A1,
    solved by weighted least squares (weights = 1/sigma^2, the same
    Poisson weighting used everywhere else in this project) with no
    iterative optimizer - so no bound-pinning or non-convergence is even
    possible here, unlike the full nonlinear fit.
    """
    n_groups = grouped_increments.shape[1]
    g2_out = np.full(n_groups, np.nan)
    x_shape = np.exp(-np.abs(tau) / tau1_fixed)  # fixed regressor, same every group

    for g in range(n_groups):
        counts = grouped_increments[:, g]
        norm_signal, sigma, ok = group_tail_normalize(counts, base_m, min_base_counts)
        if not ok:
            continue
        y = 1.0 - norm_signal
        w = 1.0 / np.maximum(sigma, 1e-6) ** 2
        denom = np.sum(w * x_shape ** 2)
        if denom <= 0:
            continue
        A1 = np.sum(w * x_shape * y) / denom
        g2_out[g] = 1.0 - A1
    return g2_out


def sequential_posterior(g2_chunks, hypotheses_by_index):
    """
    Proper sequential Bayesian update over BINARY hypotheses {0: Single,
    1: Multi}. Each super-chunk's g2 estimate is genuinely new,
    conditionally-independent evidence (given the increment construction
    above), so chaining posterior_{t-1} -> prior_t is statistically valid
    here.

    hypotheses_by_index is a list, one (mu, sigma) pair per class per
    chunk index, rather than a single fixed pair used for every step -
    since the reliability of a super-chunk's evidence changes
    systematically over the course of the chain (an early, noisy chunk
    and a late, well-converged chunk are not equally informative draws
    from the same distribution), the likelihood model is calibrated
    separately per chain position (see calibrate_hypotheses).

    Super-chunks flagged unreliable (NaN - insufficient baseline counts or
    non-convergent fit) are skipped: the posterior carries forward
    unchanged for that step.

    Returns an array of shape (n_chunks, 2): posterior probability of
    [Single, Multi] at each chunk.
    """
    log_prior = np.log(np.array([0.5, 0.5]))
    history = []
    for idx, g2 in enumerate(g2_chunks):
        hyp = hypotheses_by_index[idx]
        if not np.isfinite(g2):
            history.append(np.exp(log_prior))
            continue
        log_lik = np.array([
            gaussian_log_likelihood(g2, hyp[0]["mu"], hyp[0]["sigma"]),
            gaussian_log_likelihood(g2, hyp[1]["mu"], hyp[1]["sigma"]),
        ])
        log_post = log_lik + log_prior
        post = np.exp(log_post - np.max(log_post))
        post /= post.sum()
        history.append(post.copy())
        log_prior = np.log(np.clip(post, 1e-4, 1.0))
    return np.array(history)


# =========================================================
# 2. FILE DISCOVERY AND LABELING (identical to the LM script)
# =========================================================
FILENAME_PATTERN = re.compile(r"rep_(\d+)_n(\d+)\.csv$")


def list_raw_files_with_labels(directory):
    files, labels = [], []
    for f in sorted(os.listdir(directory)):
        if not f.startswith("g2_raw_"):
            continue
        m = FILENAME_PATTERN.search(f)
        if not m:
            continue
        n = int(m.group(2))
        files.append(f)
        labels.append(0 if n == 1 else 1)
    if not files:
        raise ValueError(f"No g2_raw_rep_<id>_n<N>.csv files found in {directory}")
    return files, np.array(labels)


def get_all_iterations(directory, files):
    sample_df = pd.read_csv(os.path.join(directory, files[0]))
    iter_cols = get_iter_columns(sample_df)
    return sorted(int(c.split("_")[1]) for c in iter_cols)


def file_chunk_g2(filepath, settings, tau1_fixed):
    """
    Loads one raw file and returns (iters, g2_chunks) for its full
    SUPER-CHUNK sequence (raw chunks aggregated in groups of
    settings["iteration_stride"] - see aggregate_increments for why).
    iters gives the LAST raw iteration number covered by each super-chunk,
    i.e. exactly the set of iterations reported/plotted elsewhere.
    """
    raw_df = pd.read_csv(filepath)
    iter_cols = get_iter_columns(raw_df)
    raw_iters = [int(c.split("_")[1]) for c in iter_cols]
    tau = raw_df["tau"].values

    base_m = np.abs(tau) >= settings["base_window_ns"]

    increments = compute_chunk_increments(raw_df, iter_cols)
    group_size = settings["iteration_stride"]
    grouped = aggregate_increments(increments, group_size)
    g2_chunks = chunk_g2_estimates(grouped, tau, base_m, tau1_fixed, settings["min_base_counts"])

    n_groups = grouped.shape[1]
    # the LAST raw iteration number in each group (e.g. group_size=10 -> 10,20,30,...)
    iters = [raw_iters[(g + 1) * group_size - 1] for g in range(n_groups)]
    return iters, g2_chunks


# =========================================================
# 3. HYPOTHESIS CALIBRATION (from this fold's TRAIN files only)
# =========================================================
def calibrate_hypotheses(train_files, directory, settings):
    """
    'Determine the distribution of g2 values for N=1, N>=2', PER CHUNK
    INDEX (per chain position) rather than pooled across the whole chain:
      1. Estimate this fold's representative tau1 from the train files'
         well-converged last-iteration fits (see estimate_representative_
         tau1).
      2. Using that fixed tau1, compute every train file's per-super-chunk
         g2 estimate (closed-form, see chunk_g2_estimates), and group by
         (true label, chunk index) rather than by true label alone - so
         the likelihood the sequential update uses at chain position i
         reflects what train files' g2 estimates actually looked like AT
         that same position, not an average over the whole noisy-to-clean
         range.

    FALLBACK: if a specific (class, index) combination has fewer than
    MIN_SAMPLES_PER_INDEX valid values (can happen at the sparsest early
    indices, or simply with few train files), that index falls back to
    the POOLED (all-index) estimate for that class - never left
    uncalibrated, just less specific for that one step.

    Returns (hypotheses_by_index, tau1_fixed, n_single_chunks_pooled,
    n_multi_chunks_pooled).
    """
    MIN_SAMPLES_PER_INDEX = 10

    tau1_fixed = estimate_representative_tau1(train_files, directory, settings)

    class_stats_by_index = None
    pooled_stats = {0: [], 1: []}

    for f, label in train_files:
        filepath = os.path.join(directory, f)
        _, g2_chunks = file_chunk_g2(filepath, settings, tau1_fixed)
        if class_stats_by_index is None:
            class_stats_by_index = [{0: [], 1: []} for _ in range(len(g2_chunks))]
        for idx, g2 in enumerate(g2_chunks):
            if np.isfinite(g2):
                class_stats_by_index[idx][label].append(g2)
                pooled_stats[label].append(g2)

    if len(pooled_stats[0]) == 0 or len(pooled_stats[1]) == 0:
        raise ValueError(
            "One class had zero valid per-super-chunk estimates in this fold's training "
            "data - check min_base_counts, or increase fold size."
        )

    pooled_hyp = {
        k: {"mu": float(np.mean(v)), "sigma": float(np.std(v))} for k, v in pooled_stats.items()
    }

    hypotheses_by_index = []
    n_fallback_used = 0
    for idx, class_stats in enumerate(class_stats_by_index):
        hyp = {}
        for k in (0, 1):
            values = class_stats[k]
            if len(values) >= MIN_SAMPLES_PER_INDEX:
                hyp[k] = {"mu": float(np.mean(values)), "sigma": float(np.std(values))}
            else:
                hyp[k] = pooled_hyp[k]
                n_fallback_used += 1
        hypotheses_by_index.append(hyp)

    if n_fallback_used > 0:
        print(f"    (fell back to pooled hypothesis for {n_fallback_used} of "
              f"{2 * len(hypotheses_by_index)} class/index combinations - too few train "
              f"samples at that specific chain position)")

    return hypotheses_by_index, tau1_fixed, len(pooled_stats[0]), len(pooled_stats[1])


# =========================================================
# 4. K-FOLD CROSS-VALIDATED EVALUATION
# =========================================================
def run_kfold_cv(settings):
    """
    Returns (long_df, fold_info_df) with the SAME schema as the LM script's
    run_kfold_cv, so every downstream plotting/metrics function is reused
    unchanged.
    """
    directory = settings["simulation_dir"]
    files, labels = list_raw_files_with_labels(directory)
    all_iterations = get_all_iterations(directory, files)
    last_iter = max(all_iterations)
    stride = settings["iteration_stride"]
    # Multiples of stride (10, 20, ..., last_iter) - matches the natural
    # super-chunk boundaries from aggregate_increments exactly, and this
    # SAME set is used in the LM script so both report/plot identically.
    eval_iterations = set(range(stride, last_iter + 1, stride))
    eval_iterations.add(last_iter)

    print(f"Found {len(files)} files ({np.sum(labels == 0)} Single, {np.sum(labels == 1)} Multi).")
    print(f"Recording predictions at {len(eval_iterations)} of {len(all_iterations)} iterations "
          f"(stride={settings['iteration_stride']}), matching the LM script exactly.")

    skf = StratifiedKFold(n_splits=settings["n_folds"], shuffle=True,
                           random_state=settings["random_state"])

    all_records = []
    fold_info = []
    per_index_calibration_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(files, labels)):
        train_files = [(files[i], labels[i]) for i in train_idx]
        test_files = [(files[i], labels[i]) for i in test_idx]

        hypotheses_by_index, tau1_fixed, n0, n1 = calibrate_hypotheses(train_files, directory, settings)
        fold_info.append([fold_idx, tau1_fixed, n0, n1, len(test_files)])
        print(f"  Fold {fold_idx}: tau1_fixed={tau1_fixed:.4f}, "
              f"{n0} Single / {n1} Multi pooled train chunks across "
              f"{len(hypotheses_by_index)} chain positions -> {len(test_files)} test files")

        for idx, hyp in enumerate(hypotheses_by_index):
            per_index_calibration_records.append([
                fold_idx, idx, hyp[0]["mu"], hyp[0]["sigma"], hyp[1]["mu"], hyp[1]["sigma"]
            ])

        for f, true_label in test_files:
            filepath = os.path.join(directory, f)
            iters, g2_chunks = file_chunk_g2(filepath, settings, tau1_fixed)
            posteriors = sequential_posterior(g2_chunks, hypotheses_by_index)  # (n_chunks, 2): [P(Single), P(Multi)]

            for it, post in zip(iters, posteriors):
                if it not in eval_iterations:
                    continue
                p_multi = post[1]
                pred_label = int(np.argmax(post))
                all_records.append([f, true_label, fold_idx, it, p_multi, pred_label])

    long_df = pd.DataFrame(all_records, columns=[
        "file", "true_label", "fold", "iteration", "g2_zero", "pred_label"
    ])
    fold_info_df = pd.DataFrame(fold_info, columns=[
        "fold", "tau1_fixed", "n_train_single_chunks_pooled", "n_train_multi_chunks_pooled", "n_test_files"
    ])
    # Per-index calibration detail (single_mu/sigma, multi_mu/sigma at each
    # chain position, per fold) - not part of the shared long_df/fold_info_df
    # schema, saved separately for transparency/auditing: shows directly
    # how the likelihood model's noise level shrinks (or doesn't) across
    # the chain.
    per_index_calibration_df = pd.DataFrame(per_index_calibration_records, columns=[
        "fold", "chunk_index", "single_mu", "single_sigma", "multi_mu", "multi_sigma"
    ])
    per_index_calibration_df.to_csv(
        os.path.join(directory, "Bayesian_iterspecific_per_index_calibration.csv"), index=False
    )
    return long_df, fold_info_df


# =========================================================
# 5. TIME-RESOLVED METRICS (identical logic to the LM script)
# =========================================================
def compute_time_resolved_metrics(long_df):
    metrics_rows = []
    for it in sorted(long_df["iteration"].unique()):
        sub = long_df[long_df["iteration"] == it].dropna(subset=["pred_label"])
        y_true = sub["true_label"].values.astype(int)
        preds = sub["pred_label"].values.astype(int)

        single_mask = y_true == 0
        recall = np.sum(preds[single_mask] == 0) / np.sum(single_mask) if np.sum(single_mask) > 0 else np.nan

        pred_single_mask = preds == 0
        precision = (
            np.sum(y_true[pred_single_mask] == 0) / np.sum(pred_single_mask)
            if np.sum(pred_single_mask) > 0 else np.nan
        )

        multi_mask = y_true == 1
        leakage = np.sum(preds[multi_mask] == 0) / np.sum(multi_mask) if np.sum(multi_mask) > 0 else np.nan

        metrics_rows.append([it, recall, precision, leakage, len(sub)])

    return pd.DataFrame(metrics_rows, columns=["Iteration", "Recall", "Precision", "Leakage", "n_valid_files"])


def plot_metrics_vs_iteration(metrics_df, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(metrics_df["Iteration"], metrics_df["Recall"], "s--", color="blue", label="Recall (Single)")
    ax.plot(metrics_df["Iteration"], metrics_df["Precision"], "o-", color="purple", label="Precision (Single)")
    ax.plot(metrics_df["Iteration"], metrics_df["Leakage"], "x:", color="red", label="Leakage (Multi->Single)")
    ax.set_xlabel("Iteration (integration time index)")
    ax.set_ylabel("Score")
    ax.set_title("Bayesian: Recall / Precision / Leakage vs. Integration Time\n(k-fold cross-validated, out-of-fold predictions)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


# =========================================================
# 6. CONFUSION MATRIX AT A CHOSEN ITERATION (identical to the LM script)
# =========================================================
def plot_confusion_matrix_at_iteration(long_df, iteration=None, settings=SETTINGS, save_path=None):
    if iteration is None:
        iteration = settings["default_confusion_matrix_iteration"] or long_df["iteration"].max()

    target_names = ["Single (N=1)", "Multi (N>=2)"]
    sub = long_df[long_df["iteration"] == iteration].dropna(subset=["pred_label"])

    if sub.empty:
        print(f"No valid classifications at iteration {iteration} to plot "
              f"(was it included in the evaluated iterations? see SETTINGS['iteration_stride']).")
        return

    y_true = sub["true_label"].astype(int)
    y_pred = sub["pred_label"].astype(int)

    print(f"\nCLASSIFICATION REPORT (iteration {iteration}, {len(sub)} out-of-fold files):")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, normalize="true", labels=[0, 1])
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt=".1%", cmap="Greens",
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Bayesian: Normalized Confusion Matrix (iteration {iteration})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


# =========================================================
# 7. CORRECTNESS HEATMAP (identical to the LM script)
# =========================================================
def plot_correctness_heatmap(long_df, save_path=None):
    df = long_df.copy()
    df["correct"] = np.where(
        df["pred_label"].isna(), np.nan,
        (df["pred_label"] == df["true_label"]).astype(float)
    )
    pivot = df.pivot(index="file", columns="iteration", values="correct")

    label_map = df.drop_duplicates("file").set_index("file")["true_label"]
    pivot = pivot.loc[label_map.sort_values().index]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.15 * len(pivot))))
    sns.heatmap(pivot, cmap="RdYlGn", cbar=False, linewidths=0.5, linecolor="lightgray",
                mask=pivot.isna(), ax=ax)
    ax.set_title("Bayesian: Correct (green) / Incorrect (red) per file, per iteration\n"
                  "(out-of-fold predictions; files grouped Single then Multi)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("File")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()
    return pivot


# =========================================================
# 8. EXECUTION
# =========================================================
if __name__ == "__main__":
    out_dir = SETTINGS["simulation_dir"]

    long_df, fold_info_df = run_kfold_cv(SETTINGS)

    print("\n" + "=" * 70)
    print("PER-FOLD CALIBRATION SUMMARY")
    print("=" * 70)
    print(fold_info_df.to_string(index=False))
    fold_info_df.to_csv(os.path.join(out_dir, "Bayesian_fold_calibration.csv"), index=False)

    metrics_df = compute_time_resolved_metrics(long_df)
    print("\nAGGREGATED PERFORMANCE vs. INTEGRATION TIME (out-of-fold):")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(os.path.join(out_dir, "Bayesian_metrics_vs_iteration.csv"), index=False)

    long_df.to_csv(os.path.join(out_dir, "Bayesian_predictions_long.csv"), index=False)

    plot_metrics_vs_iteration(metrics_df, save_path=os.path.join(out_dir, "Bayesian_metrics_vs_iteration.png"))

    plot_confusion_matrix_at_iteration(
        long_df, save_path=os.path.join(out_dir, "Bayesian_confusion_matrix_last_iter.png")
    )

    correctness_pivot = plot_correctness_heatmap(
        long_df, save_path=os.path.join(out_dir, "Bayesian_correctness_heatmap.png")
    )
    correctness_pivot.to_csv(os.path.join(out_dir, "Bayesian_correctness_matrix.csv"))

    print("\nDone. All outputs saved to:", out_dir)