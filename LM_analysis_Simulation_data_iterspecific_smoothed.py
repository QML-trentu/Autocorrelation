# -*- coding: utf-8 -*-
"""
LM_analysis_Simulation_data_iterspecific_smoothed.py

Levenberg-Marquardt (traditional nonlinear least-squares) baseline
classifier for binary emitter classification (N=1 "Single" vs N>=2
"Multi"), evaluated with k-fold cross-validation and the same three
reporting outputs used by the Bayesian and Encoder scripts in this
project, so all three are directly comparable:

  1. plot_confusion_matrix_at_iteration(..., iteration=X)
  2. plot_correctness_heatmap(...) - per-file, per-iteration correct/incorrect
  3. compute_time_resolved_metrics(...) + plot_metrics_vs_iteration(...) -
     Recall / Precision / Leakage vs iteration

METHODOLOGY:
  For each of N_FOLDS stratified folds:
    - TRAIN files: fit g2(0) at each evaluated iteration, grouped by true
      label, giving a separate decision threshold per iteration (the
      midpoint between the two classes' mean fitted g2(0) at that
      iteration) rather than a single threshold calibrated once from the
      best-converged data and reused everywhere - since fit behavior
      changes systematically with how much data is available, a
      threshold tuned for mature, high-statistics data can be the wrong
      threshold for sparse early-iteration data.
    - TEST files (held out): fit g2(0) at each evaluated iteration and
      classify using only that iteration's threshold from this fold
      (never calibrated on these files).
  Every file is used as a held-out test file exactly once, with a
  threshold that never saw it - avoiding the optimistic bias of
  calibrating and testing on the same data.

POST-HOC SMOOTHING: predictions are further stabilized with exponential
moving average (EMA) smoothing of each file's own per-iteration margin
(fitted g2(0) minus that iteration's calibrated threshold) - see
smooth_predictions_ema. The margin, not the raw g2(0) value, is smoothed:
because the calibrated threshold itself shifts substantially across the
chain (lower at sparse early iterations, higher at well-converged late
iterations), smoothing raw g2(0) directly would blend evidence computed
under very different decision boundaries before comparing it to only the
current iteration's threshold - an inconsistent comparison. Smoothing the
margin instead expresses each iteration's evidence relative to its own
decision boundary before blending, so combining evidence across
iterations happens on a consistent scale throughout. Same underlying
rationale as the Encoder's smoothing: a file's true label doesn't change
across iterations, only the statistics available about it do, so each
iteration's independent fit is a noisy read-out of a persistent,
unchanging truth, and averaging multiple such read-outs is a standard
variance-reduction technique - not a claim that consecutive iterations
are statistically independent (they are not, since the underlying data
is cumulative).

Definitions (N=1 = "positive" class, matching the goal of finding single
emitters):
    Recall    = P(predict Single | true Single)
    Precision = P(true Single | predict Single)
    Leakage   = P(predict Single | true Multi)   <- false verification rate,
                the standard single-photon-source-verification meaning of
                "leakage": how often a multi-emitter source passes as
                verified-single.
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
# CONFIGURATION
# =========================================================
SETTINGS = {
    "base_window_ns": 10.0,
    "min_base_counts": 5,

    # Step_02's output folder - used for BOTH calibration and evaluation,
    # via k-fold CV (no more separate real/synthetic split).
    "simulation_dir": "Simulation_Results",

    "n_folds": 5,
    "random_state": 42,

    # 300 iterations x ~100 files x an LM fit each is a lot of fitting -
    # the trend plot / correctness heatmap use every ITERATION_STRIDE'th
    # iteration for tractability. The confusion matrix at a single chosen
    # iteration is unaffected by this and can use ANY iteration on request.
    "iteration_stride": 10,

    # Default iteration for the confusion matrix plot if none is given.
    "default_confusion_matrix_iteration": None,

    # EMA smoothing (see smooth_predictions_ema below) - 0.4 matches the
    # value used in Encoder_analysis_Simulation_data_smoothed.py, for a
    # like-for-like comparison of the same smoothing strength across
    # methods. Not independently tuned for LM specifically.
    "ema_alpha": 0.4,
}


# =========================================================
# 1. PHYSICAL MODEL
# =========================================================
def g2_biexp(tau, rho1_sq, tau1, rho2_sq, tau2):
    """g2(tau) = 1 - rho1^2*exp(-|tau|/tau1) + rho2^2*exp(-|tau|/tau2)"""
    t = np.abs(tau)
    return 1.0 - (rho1_sq * np.exp(-t / tau1)) + (rho2_sq * np.exp(-t / tau2))


def fit_g2_zero(tau_axis, g2_signal, sigma=None):
    """
    Fits the bi-exponential model. Returns (g2_zero, success).

    Weighted (Poisson sigma) when sigma is given - the statistically
    correct approach for count data this sparse (see tail_normalize).
    """
    p0 = [0.5, 0.5, 0.0, 3.0]
    bounds = ([0.0, 0.1, 0.0, 2.0], [1.5, 1.5, 1.0, 6.0])
    try:
        if sigma is not None:
            popt, _ = curve_fit(g2_biexp, tau_axis, g2_signal, p0=p0, bounds=bounds,
                                  sigma=sigma, absolute_sigma=True, maxfev=20000)
        else:
            popt, _ = curve_fit(g2_biexp, tau_axis, g2_signal, p0=p0, bounds=bounds, maxfev=20000)
        rho1_sq, tau1, rho2_sq, tau2 = popt
        return 1.0 - rho1_sq + rho2_sq, True
    except Exception:
        return np.nan, False


def classify_g2_zero(g2_zero, threshold):
    """Binary: 0 = Single (N=1), 1 = Multi (N>=2)."""
    if not np.isfinite(g2_zero):
        return None
    return 0 if g2_zero < threshold else 1


def get_iter_columns(df):
    cols = [c for c in df.columns if c.startswith("iter_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def tail_normalize(counts, base_m, min_base_counts):
    """
    Returns (normalized_signal, sigma, ok). sigma is the propagated Poisson
    uncertainty per bin, passed to fit_g2_zero for weighted fitting.
    """
    base_sum = counts[base_m].sum()
    if base_sum < min_base_counts:
        return None, None, False
    baseline = counts[base_m].mean()
    norm_signal = counts / baseline
    sigma = np.sqrt(np.maximum(counts, 1)) / baseline
    return norm_signal, sigma, True


# =========================================================
# 2. FILE DISCOVERY AND LABELING
# =========================================================
FILENAME_PATTERN = re.compile(r"rep_(\d+)_n(\d+)\.csv$")


def list_raw_files_with_labels(directory):
    """
    Finds every g2_raw_rep_<id>_n<N>.csv file (raw counts - needed for
    weighted fitting) and returns [(filename, true_label)], true_label
    0=Single (N=1), 1=Multi (N>=2).
    """
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


def fit_file_at_iterations(filepath, settings, iterations):
    """
    Fits g2(0) for one file at each requested iteration. Returns a dict
    {iteration: g2_zero}. iterations not present in the file are skipped.
    """
    df = pd.read_csv(filepath)
    tau = df["tau"].values
    base_m = np.abs(tau) >= settings["base_window_ns"]

    iter_cols = get_iter_columns(df)
    available = {int(c.split("_")[1]): c for c in iter_cols}

    results = {}
    for it in iterations:
        if it not in available:
            continue
        counts = df[available[it]].values.astype(float)
        norm_signal, sigma, ok = tail_normalize(counts, base_m, settings["min_base_counts"])
        if not ok:
            results[it] = np.nan
            continue
        g2_zero, fit_ok = fit_g2_zero(tau, norm_signal, sigma=sigma)
        results[it] = g2_zero if fit_ok else np.nan
    return results


def get_last_iteration(directory, files):
    sample_df = pd.read_csv(os.path.join(directory, files[0]))
    iter_cols = get_iter_columns(sample_df)
    return max(int(c.split("_")[1]) for c in iter_cols)


def get_all_iterations(directory, files):
    sample_df = pd.read_csv(os.path.join(directory, files[0]))
    iter_cols = get_iter_columns(sample_df)
    return sorted(int(c.split("_")[1]) for c in iter_cols)


# =========================================================
# 3. THRESHOLD CALIBRATION (from this fold's TRAIN files only)
# =========================================================
def calibrate_thresholds_per_iteration(train_files, directory, settings, eval_iterations):
    """
    Calibrates a separate threshold for each evaluated iteration, using
    train files' fits at that same iteration, instead of a single
    threshold learned only from the last (best-converged) iteration and
    reused everywhere (the Bayesian script's per-chain-position likelihood
    calibration follows the same underlying reasoning). A threshold tuned
    for what mature, high-statistics data looks like may simply be the
    wrong threshold for what sparse early-iteration data looks like, since
    the two regimes have systematically different fit behavior.

    Each train file is read and fit once, across all eval_iterations in a
    single pass (via fit_file_at_iterations' existing batching), then
    results are regrouped by iteration - so this costs ~len(eval_iterations)
    times more calibration fits than a single-threshold version, not
    len(eval_iterations) times more file reads.

    FALLBACK: if a specific (class, iteration) combination has fewer than
    MIN_SAMPLES_PER_ITERATION successfully-fit train files, that iteration
    falls back to the pooled (all-iteration) mean for that class (mirrors
    the Bayesian script's per-index fallback), so no iteration is ever
    left uncalibrated, just less specific for that one step.
    """
    MIN_SAMPLES_PER_ITERATION = 5

    class_g2_by_iter = {it: {0: [], 1: []} for it in eval_iterations}
    for f, label in train_files:
        filepath = os.path.join(directory, f)
        g2_by_iter = fit_file_at_iterations(filepath, settings, eval_iterations)
        for it, g2_zero in g2_by_iter.items():
            if np.isfinite(g2_zero):
                class_g2_by_iter[it][label].append(g2_zero)

    pooled_single = [g2 for it in eval_iterations for g2 in class_g2_by_iter[it][0]]
    pooled_multi = [g2 for it in eval_iterations for g2 in class_g2_by_iter[it][1]]
    if len(pooled_single) == 0 or len(pooled_multi) == 0:
        raise ValueError(
            "One class had zero successfully-fit calibration files across ALL iterations "
            "in this fold - increase n_folds's training size or check for pervasive fit failures."
        )
    pooled_mean_single = float(np.mean(pooled_single))
    pooled_mean_multi = float(np.mean(pooled_multi))

    thresholds, means_single, means_multi, n_single, n_multi = {}, {}, {}, {}, {}
    n_fallback = 0
    for it in eval_iterations:
        vals0, vals1 = class_g2_by_iter[it][0], class_g2_by_iter[it][1]
        if len(vals0) < MIN_SAMPLES_PER_ITERATION:
            mean0 = pooled_mean_single
            n_fallback += 1
        else:
            mean0 = float(np.mean(vals0))
        if len(vals1) < MIN_SAMPLES_PER_ITERATION:
            mean1 = pooled_mean_multi
            n_fallback += 1
        else:
            mean1 = float(np.mean(vals1))

        thresholds[it] = (mean0 + mean1) / 2
        means_single[it] = mean0
        means_multi[it] = mean1
        n_single[it] = len(vals0)
        n_multi[it] = len(vals1)

    if n_fallback > 0:
        print(f"    (fell back to pooled threshold for {n_fallback} of "
              f"{2 * len(eval_iterations)} class/iteration combinations - too few "
              f"successfully-fit train files at that specific iteration)")

    return thresholds, means_single, means_multi, n_single, n_multi


# =========================================================
# 4. K-FOLD CROSS-VALIDATED EVALUATION
# =========================================================
def run_kfold_cv(settings):
    """
    Returns (long_df, fold_info_df):
      long_df: one row per (file, true_label, fold, iteration) with
               g2_zero and pred_label, for every TEST (held-out) file at
               every evaluated iteration - i.e. out-of-fold predictions
               covering every file exactly once.
      fold_info_df: one row per fold with its calibrated threshold and
                    class means, for transparency/auditing.
    """
    directory = settings["simulation_dir"]
    files, labels = list_raw_files_with_labels(directory)
    all_iterations = get_all_iterations(directory, files)
    last_iter = max(all_iterations)
    # Multiples of stride (10, 20, ..., last_iter) - kept identical to the
    # Bayesian script's eval_iterations (there, this also matches the
    # natural per-chunk aggregation boundaries) so both report/plot at
    # exactly the same iterations for a fair, direct comparison.
    eval_iterations = list(range(settings["iteration_stride"], last_iter + 1, settings["iteration_stride"]))
    if last_iter not in eval_iterations:
        eval_iterations = eval_iterations + [last_iter]

    print(f"Found {len(files)} files ({np.sum(labels == 0)} Single, {np.sum(labels == 1)} Multi).")
    print(f"Evaluating {len(eval_iterations)} of {len(all_iterations)} iterations "
          f"(stride={settings['iteration_stride']}): {eval_iterations}")

    skf = StratifiedKFold(n_splits=settings["n_folds"], shuffle=True,
                           random_state=settings["random_state"])

    all_records = []
    fold_info = []
    per_iter_calibration_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(files, labels)):
        train_files = [(files[i], labels[i]) for i in train_idx]
        test_files = [(files[i], labels[i]) for i in test_idx]

        thresholds, means_single, means_multi, n_single, n_multi = calibrate_thresholds_per_iteration(
            train_files, directory, settings, eval_iterations
        )
        fold_info.append([fold_idx, len(train_files), len(test_files)])
        print(f"  Fold {fold_idx}: calibrated {len(eval_iterations)} per-iteration thresholds "
              f"(e.g. iter {eval_iterations[0]}: threshold={thresholds[eval_iterations[0]]:.4f}; "
              f"iter {eval_iterations[-1]}: threshold={thresholds[eval_iterations[-1]]:.4f}) "
              f"-> {len(test_files)} test files")

        for it in eval_iterations:
            per_iter_calibration_records.append([
                fold_idx, it, thresholds[it], means_single[it], means_multi[it], n_single[it], n_multi[it]
            ])

        for f, true_label in test_files:
            filepath = os.path.join(directory, f)
            g2_by_iter = fit_file_at_iterations(filepath, settings, eval_iterations)
            for it, g2_zero in g2_by_iter.items():
                pred_label = classify_g2_zero(g2_zero, thresholds[it])
                all_records.append([f, true_label, fold_idx, thresholds[it], it, g2_zero, pred_label])

    long_df = pd.DataFrame(all_records, columns=[
        "file", "true_label", "fold", "threshold", "iteration", "g2_zero", "pred_label"
    ])
    fold_info_df = pd.DataFrame(fold_info, columns=[
        "fold", "n_train_files", "n_test_files"
    ])
    # Per-iteration calibration detail (threshold, class means, sample
    # counts at each evaluated iteration, per fold) - not part of the
    # shared long_df/fold_info_df schema, saved separately for
    # transparency/auditing, mirroring the Bayesian iterspecific variant's
    # Bayesian_iterspecific_per_index_calibration.csv.
    per_iter_calibration_df = pd.DataFrame(per_iter_calibration_records, columns=[
        "fold", "iteration", "threshold", "single_mean_g2_0", "multi_mean_g2_0",
        "n_train_single", "n_train_multi"
    ])
    per_iter_calibration_df.to_csv(
        os.path.join(directory, "LM_iterspecific_per_iteration_calibration.csv"), index=False
    )
    return long_df, fold_info_df


# =========================================================
# 4b. POST-HOC SMOOTHING ACROSS ITERATIONS
# =========================================================
def smooth_predictions_ema(long_df, alpha):
    """
    Exponential moving average (EMA) smoothing of each file's own
    per-iteration margin (fitted g2(0) minus that iteration's calibrated
    threshold), then reclassifies from the sign of the smoothed margin.

    The margin, not the raw g2(0) value, is smoothed: the per-iteration
    threshold calibrated above moves substantially across the chain
    (lower at sparse early iterations, higher at well-converged late
    ones). Smoothing g2(0) in its own raw scale would blend evidence
    computed under very different "natural" thresholds, then compare that
    blend against only the current iteration's threshold - an
    inconsistent comparison. Smoothing the margin instead fixes this:
    g2(0)-threshold is already expressed relative to that iteration's own
    decision boundary before smoothing, so blending margins across
    iterations combines evidence on a consistent scale (positive =
    leaning Multi, negative = leaning Single, regardless of iteration) -
    the same role logit(P(Multi)) plays for the Encoder, adapted to LM's
    threshold-relative-margin representation instead of a probability.
    """
    df = long_df.sort_values(["file", "iteration"]).copy()
    margin = df["g2_zero"].values - df["threshold"].values
    files_arr = df["file"].values

    smoothed_margin = np.empty_like(margin)
    prev_file = None
    prev_smoothed = np.nan
    for i in range(len(df)):
        this_file = files_arr[i]
        new_file = this_file != prev_file
        if not np.isfinite(margin[i]):
            smoothed_margin[i] = np.nan if (new_file or not np.isfinite(prev_smoothed)) else prev_smoothed
        elif new_file or not np.isfinite(prev_smoothed):
            smoothed_margin[i] = margin[i]
        else:
            smoothed_margin[i] = alpha * margin[i] + (1 - alpha) * prev_smoothed
        prev_smoothed = smoothed_margin[i]
        prev_file = this_file

    # Recover a g2_zero-like value for reporting/plotting continuity
    # (threshold + smoothed margin), and classify directly from the
    # margin's sign (equivalent to classify_g2_zero, but avoids re-adding
    # then re-subtracting the threshold).
    df["g2_zero"] = df["threshold"].values + smoothed_margin
    df["pred_label"] = [
        (None if not np.isfinite(m) else (0 if m < 0 else 1)) for m in smoothed_margin
    ]
    return df


# =========================================================
# 5. TIME-RESOLVED METRICS (Recall / Precision / Leakage vs iteration)
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
    ax.set_title("LM Baseline: Recall / Precision / Leakage vs. Integration Time\n(k-fold cross-validated, out-of-fold predictions)")
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
# 6. CONFUSION MATRIX AT A CHOSEN ITERATION
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
    plt.title(f"LM Baseline: Normalized Confusion Matrix (iteration {iteration})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


# =========================================================
# 7. CORRECTNESS HEATMAP (per-file, per-iteration correct/incorrect)
# =========================================================
def plot_correctness_heatmap(long_df, save_path=None):
    """
    One row per file, one column per (evaluated) iteration: green if that
    iteration's out-of-fold classification was correct, red if not, blank
    if the fit didn't converge.
    """
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
    ax.set_title("LM Baseline: Correct (green) / Incorrect (red) per file, per iteration\n"
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
    fold_info_df.to_csv(os.path.join(out_dir, "LM_fold_calibration.csv"), index=False)

    long_df.to_csv(os.path.join(out_dir, "LM_predictions_long_raw.csv"), index=False)

    # Predictions are smoothed before computing/reporting metrics (see
    # smooth_predictions_ema) - the raw (pre-smoothing) predictions above
    # are kept as an intermediate artifact for reproducibility, but the
    # raw, unsmoothed metrics themselves are not part of the reported
    # analysis and are not computed here.
    smoothed_df = smooth_predictions_ema(long_df, alpha=SETTINGS["ema_alpha"])
    smoothed_df.to_csv(os.path.join(out_dir, "LM_predictions_long.csv"), index=False)

    metrics_df = compute_time_resolved_metrics(smoothed_df)
    print(f"\nPERFORMANCE vs. INTEGRATION TIME (out-of-fold, EMA alpha={SETTINGS['ema_alpha']}):")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(os.path.join(out_dir, "LM_metrics_vs_iteration.csv"), index=False)

    plot_metrics_vs_iteration(metrics_df, save_path=os.path.join(out_dir, "LM_metrics_vs_iteration.png"))

    plot_confusion_matrix_at_iteration(
        smoothed_df, save_path=os.path.join(out_dir, "LM_confusion_matrix_last_iter.png")
    )  # default: last evaluated iteration
    # Example override: plot_confusion_matrix_at_iteration(smoothed_df, iteration=51)

    correctness_pivot = plot_correctness_heatmap(
        smoothed_df, save_path=os.path.join(out_dir, "LM_correctness_heatmap.png")
    )
    correctness_pivot.to_csv(os.path.join(out_dir, "LM_correctness_matrix.csv"))

    print("\nDone. All outputs saved to:", out_dir)