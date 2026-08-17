# -*- coding: utf-8 -*-
"""
Encoder_analysis_simulation_data_smoothed.py

Binary (N=1 "Single" vs N>=2 "Multi") neural network classifier using
scikit-learn's MLPClassifier, evaluated with the same k-fold cross-
validation and the same three reporting outputs as the LM and Bayesian
scripts in this project, so all three are directly comparable.

FEATURE REPRESENTATION: rather than feeding the full ~2000-point raw g2
curve directly to the network, each curve is reduced to a compact feature
vector (see two_tier_features): full original resolution within
+/-NEAR_WINDOW_NS of the dip (where essentially all the discriminating
shape lives), plus a small number of coarse-averaged bins covering the
flat far-field region on each side (which only needs to establish the
baseline level, not fine detail). This keeps the feature count small
relative to the number of independent training files available, limiting
the model's capacity to memorize file-specific noise rather than learn
the general "dip clarifies with more integration time" pattern. A naive
uniform downsampling of the full curve to a similarly small number of
bins performs worse here: the bin width becomes comparable to or wider
than the narrowest antibunching lifetime in Step_02's range, washing out
the dip almost entirely (the same coarse-binning bias that affects the
LM fit at narrow tau1).

CLASS BALANCE: Step_02's default emitter-count distribution gives a
minority Single class (~18-20% of files). Since plain classification
accuracy - used both for gradient descent and for MLPClassifier's
internal early-stopping metric - barely penalizes missing minority-class
examples when they are this outnumbered, training rows are oversampled to
balance classes (see oversample_minority) before fitting. This is applied
to TRAINING data only; test data is always left at its true, natural
class balance.

EARLY STOPPING WITHOUT TEST-SET LEAKAGE: sklearn's
MLPClassifier(early_stopping=True) carves its own validation split
internally from whatever data is passed to .fit() - since only this
fold's TRAIN files are ever passed to .fit(), the held-out TEST fold is
never touched during training or model selection. (One residual
imperfection, accepted as a pragmatic trade-off: sklearn's internal
validation split is per-row, i.e. by (file, iteration) sample, not
per-file, so different iterations of the same train file could end up
split between its internal train/validation portions. This only affects
when training stops, not the final evaluation on the genuinely held-out
test fold.)

POST-HOC SMOOTHING: predictions are further stabilized with exponential
moving average (EMA) smoothing of each file's own per-iteration P(Multi)
sequence (see smooth_predictions_ema) - explicitly not a Bayesian chain,
and that distinction matters. The Bayesian script's sequential update is
valid because each of its evidence increments is genuinely independent
(differences between consecutive cumulative histograms). This network is
instead fed the cumulative curve directly at every iteration, so
consecutive predictions are not independent evidence; combining them the
way the Bayesian script does would manufacture false confidence from
heavily overlapping data. EMA smoothing makes no such independence
claim - it is a plain signal-processing technique that blends each new
observation with the running smoothed value, reducing point-to-point
jaggedness by construction without overclaiming statistical rigor it
doesn't have. Both raw (unsmoothed) and smoothed metrics/predictions are
saved, so the effect of this step can be inspected directly.

OUTPUT SCHEMA (identical to the LM and Bayesian scripts' long_df, so the
plotting functions below produce directly comparable figures):
    file, true_label, fold, iteration, g2_zero, pred_label
(here g2_zero holds the predicted probability of "Multi" - the network's
analog of LM's fitted g2(0) and Bayesian's posterior probability.)
"""
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# =========================================================
# CONFIGURATION (k-fold/iteration settings match LM/Bayesian exactly)
# =========================================================
SETTINGS = {
    "simulation_dir": "Simulation_Results",
    "n_folds": 5,
    "random_state": 42,
    "iteration_stride": 10,
    "default_confusion_matrix_iteration": None,

    # Feature representation (see module docstring point 1): full
    # resolution within +/- this many ns of the dip, coarse bins beyond.
    "near_window_ns": 10.0,
    "n_far_bins_per_side": 15,

    # MLP architecture/training - deliberately modest given likely-small
    # file counts (see docstring); alpha (L2) set fairly high to further
    # discourage memorization.
    "hidden_layer_sizes": (64, 32),
    "alpha": 1e-2,
    "max_iter": 2000,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 15,

    # EMA smoothing (see smooth_predictions_ema) - NOT the same "alpha" as
    # the MLP's L2 regularization above, just an unfortunately-overloaded
    # name in sklearn's own convention. 0.4 is a starting point, not
    # tuned - test directly against the unsmoothed output before trusting it.
    "ema_alpha": 0.4,
}


# =========================================================
# 1. FILE DISCOVERY AND LABELING (same convention as LM/Bayesian)
# =========================================================
FILENAME_PATTERN = re.compile(r"rep_(\d+)_n(\d+)\.csv$")


def list_norm_files_with_labels(directory):
    """
    Finds every g2_norm_rep_<id>_n<N>.csv file (pre-normalized - the NN
    doesn't need raw counts, it has no explicit statistical weighting)
    and returns [(filename, true_label)], true_label 0=Single, 1=Multi.
    """
    files, labels = [], []
    for f in sorted(os.listdir(directory)):
        if not f.startswith("g2_norm_"):
            continue
        m = FILENAME_PATTERN.search(f)
        if not m:
            continue
        n = int(m.group(2))
        files.append(f)
        labels.append(0 if n == 1 else 1)
    if not files:
        raise ValueError(f"No g2_norm_rep_<id>_n<N>.csv files found in {directory}")
    return files, np.array(labels)


def get_all_iterations(directory, files):
    sample_df = pd.read_csv(os.path.join(directory, files[0]))
    iter_cols = [c for c in sample_df.columns if c.startswith("iter_")]
    return sorted(int(c.split("_")[1]) for c in iter_cols)


# =========================================================
# 2. FEATURE REPRESENTATION (see docstring point 1)
# =========================================================
def _coarse_bin(values, n_bins):
    n = len(values)
    bin_size = max(n // n_bins, 1)
    trimmed = values[: bin_size * n_bins]
    return trimmed.reshape(n_bins, bin_size).mean(axis=1)


def two_tier_features(tau, x, near_window_ns, n_far_bins_per_side):
    """
    Full original resolution within +/-near_window_ns of tau=0 (where the
    dip lives), plus n_far_bins_per_side coarse-averaged bins on each side
    beyond that (the flat far-field region, which only needs to establish
    the baseline level). See module docstring for why this is used
    instead of naive uniform downsampling.
    """
    near_m = np.abs(tau) <= near_window_ns
    far_neg_m = tau < -near_window_ns
    far_pos_m = tau > near_window_ns

    near_features = x[near_m]
    far_neg_features = _coarse_bin(x[far_neg_m], n_far_bins_per_side)
    far_pos_features = _coarse_bin(x[far_pos_m], n_far_bins_per_side)

    return np.concatenate([far_neg_features, near_features, far_pos_features])


def oversample_minority(X, y, random_state):
    """
    Duplicates minority-class rows (with a small amount of Gaussian jitter
    on each duplicate, so the optimizer doesn't see literally identical
    rows repeated) until both classes have equal representation.

    Without this, a fold with ~18-20%/80-82% class balance (the natural
    result of Step_02's default n_emitters~Uniform(1,5)) trains a model
    that can achieve deceptively high overall accuracy by mostly
    predicting the majority class - plain accuracy, used both for
    gradient descent and MLPClassifier's internal early-stopping metric,
    barely penalizes missing minority-class examples when they are this
    outnumbered, so Recall for Single can stay poor even while reported
    validation accuracy looks excellent.

    Only ever applied to TRAINING data - test data is left at its true,
    natural class balance throughout this whole project's evaluation.
    """
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()

    X_parts, y_parts = [X], [y]
    for cls, count in zip(classes, counts):
        n_needed = majority_count - count
        if n_needed <= 0:
            continue
        cls_rows = X[y == cls]
        pick_idx = rng.randint(0, len(cls_rows), size=n_needed)
        duplicated = cls_rows[pick_idx]
        # Small jitter (1% of each feature's training std) so duplicates
        # aren't bit-for-bit identical rows.
        feature_std = X.std(axis=0)
        jitter = rng.normal(0, 1, size=duplicated.shape) * (0.01 * feature_std)
        X_parts.append(duplicated + jitter)
        y_parts.append(np.full(n_needed, cls))

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def build_feature_matrix(directory, files_labels, iterations, settings, tau_ref=None):
    """
    Returns (X, y, file_col, iter_col): one row per (file, iteration),
    X built via two_tier_features. tau_ref (the tau axis) is read once
    from the first file passed if not given - assumed identical across
    all files, which Step_02 guarantees.
    """
    X_list, y_list, file_list, iter_list = [], [], [], []
    for f, label in files_labels:
        df = pd.read_csv(os.path.join(directory, f))
        if tau_ref is None:
            tau_ref = df["tau"].values
        for it in iterations:
            col = f"iter_{it}"
            if col not in df.columns:
                continue
            x = df[col].values.astype(float)
            feat = two_tier_features(tau_ref, x, settings["near_window_ns"],
                                      settings["n_far_bins_per_side"])
            X_list.append(feat)
            y_list.append(label)
            file_list.append(f)
            iter_list.append(it)
    return np.array(X_list), np.array(y_list), np.array(file_list), np.array(iter_list)


# =========================================================
# 3. K-FOLD CROSS-VALIDATED EVALUATION
# =========================================================
def run_kfold_cv(settings):
    """
    Returns (long_df, fold_info_df) with the SAME schema as the LM/
    Bayesian scripts' run_kfold_cv, so every downstream plotting/metrics
    function is reused unchanged.
    """
    directory = settings["simulation_dir"]
    files, labels = list_norm_files_with_labels(directory)
    all_iterations = get_all_iterations(directory, files)
    last_iter = max(all_iterations)
    stride = settings["iteration_stride"]
    eval_iterations = sorted(set(range(stride, last_iter + 1, stride)) | {last_iter})

    tau_ref = pd.read_csv(os.path.join(directory, files[0]))["tau"].values
    n_features = len(two_tier_features(tau_ref, tau_ref, settings["near_window_ns"],
                                        settings["n_far_bins_per_side"]))

    print(f"Found {len(files)} files ({np.sum(labels == 0)} Single, {np.sum(labels == 1)} Multi).")
    print(f"Feature representation: {n_features} values per curve "
          f"(full resolution within +/-{settings['near_window_ns']}ns, "
          f"{settings['n_far_bins_per_side']} coarse bins per side beyond that).")
    print(f"Training and recording predictions at {len(eval_iterations)} of {len(all_iterations)} "
          f"iterations (stride={stride}), matching the LM/Bayesian scripts exactly.")

    skf = StratifiedKFold(n_splits=settings["n_folds"], shuffle=True,
                           random_state=settings["random_state"])

    all_records = []
    fold_info = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(files, labels)):
        train_files_labels = [(files[i], labels[i]) for i in train_idx]
        test_files_labels = [(files[i], labels[i]) for i in test_idx]

        X_train, y_train, _, _ = build_feature_matrix(
            directory, train_files_labels, eval_iterations, settings, tau_ref
        )
        X_test, y_test, file_test, iter_test = build_feature_matrix(
            directory, test_files_labels, eval_iterations, settings, tau_ref
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_bal, y_train_bal = oversample_minority(
            X_train_scaled, y_train, settings["random_state"] + fold_idx
        )

        clf = MLPClassifier(
            hidden_layer_sizes=settings["hidden_layer_sizes"],
            alpha=settings["alpha"],
            max_iter=settings["max_iter"],
            early_stopping=settings["early_stopping"],
            validation_fraction=settings["validation_fraction"],
            n_iter_no_change=settings["n_iter_no_change"],
            random_state=settings["random_state"] + fold_idx,
        )
        clf.fit(X_train_bal, y_train_bal)

        n0 = int(np.sum(y_train_bal == 0))
        n1 = int(np.sum(y_train_bal == 1))
        val_acc = clf.best_validation_score_ if hasattr(clf, "best_validation_score_") else np.nan
        fold_info.append([fold_idx, n0, n1, len(test_files_labels), clf.n_iter_, val_acc])
        print(f"  Fold {fold_idx}: trained on {n0} Single / {n1} Multi (file,iteration) rows "
              f"AFTER oversampling to balance classes "
              f"({clf.n_iter_} iterations to convergence) -> {len(test_files_labels)} test files")

        probs = clf.predict_proba(X_test_scaled)
        preds = clf.predict(X_test_scaled)
        multi_col = list(clf.classes_).index(1)
        p_multi = probs[:, multi_col]

        for f, true_label, it, pm, pred in zip(file_test, y_test, iter_test, p_multi, preds):
            all_records.append([f, int(true_label), fold_idx, int(it), float(pm), int(pred)])

    long_df = pd.DataFrame(all_records, columns=[
        "file", "true_label", "fold", "iteration", "g2_zero", "pred_label"
    ])
    fold_info_df = pd.DataFrame(fold_info, columns=[
        "fold", "n_train_single_rows", "n_train_multi_rows", "n_test_files",
        "n_iter_to_converge", "internal_val_accuracy"
    ])
    return long_df, fold_info_df


# =========================================================
# 3b. POST-HOC SMOOTHING ACROSS ITERATIONS
# =========================================================
def smooth_predictions_ema(long_df, alpha):
    """
    Exponential moving average (EMA) smoothing of each file's own
    per-iteration P(Multi) sequence, in logit space.

    IMPORTANT - why this is smoothing, not a Bayesian chain: the Bayesian
    script's sequential update is statistically valid because each
    super-chunk is built from a genuinely independent increment (raw
    count DIFFERENCES between consecutive cumulative iterations). The
    encoder, in contrast, is fed the cumulative curve directly at every
    iteration - iteration i's input already contains everything iteration
    i-1's input did. Treating consecutive encoder outputs as independent
    evidence and summing their log-likelihood-ratios (the Bayesian
    approach) would manufacture false confidence from heavily overlapping
    data - exactly the mistake the Bayesian design was built to avoid in
    the first place.

    EMA smoothing makes no such independence claim. It's a plain signal-
    processing technique: each point is a weighted blend of the new
    observation and the previous smoothed value, which directly reduces
    point-to-point jaggedness by construction, without claiming the
    underlying observations are independent.

    alpha (0 < alpha <= 1) controls the responsiveness/smoothness
    trade-off: alpha=1 reproduces the raw (unsmoothed) sequence exactly;
    smaller alpha smooths more but reacts more slowly to genuine
    improvement in later iterations.

    Operates per-file, independent of fold (this is purely a temporal
    post-processing step on already-computed out-of-fold predictions, not
    part of training/calibration).
    """
    df = long_df.sort_values(["file", "iteration"]).copy()
    p = np.clip(df["g2_zero"].values, 1e-4, 1 - 1e-4)
    logit = np.log(p / (1 - p))

    smoothed_logit = np.empty_like(logit)
    prev_file = None
    prev_smoothed = None
    for i in range(len(df)):
        this_file = df["file"].values[i]
        if this_file != prev_file:
            smoothed_logit[i] = logit[i]  # first observation for this file: no history yet
        else:
            smoothed_logit[i] = alpha * logit[i] + (1 - alpha) * prev_smoothed
        prev_smoothed = smoothed_logit[i]
        prev_file = this_file

    smoothed_p = 1 / (1 + np.exp(-smoothed_logit))
    df["g2_zero"] = smoothed_p
    df["pred_label"] = (smoothed_p > 0.5).astype(int)
    return df


# =========================================================
# 4. TIME-RESOLVED METRICS (identical logic to LM/Bayesian)
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
    ax.set_title("Encoder/NN (sklearn MLP): Recall / Precision / Leakage vs. Integration Time\n(k-fold cross-validated, out-of-fold predictions)")
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
# 5. CONFUSION MATRIX AT A CHOSEN ITERATION (identical to LM/Bayesian)
# =========================================================
def plot_confusion_matrix_at_iteration(long_df, iteration=None, settings=SETTINGS, save_path=None):
    if iteration is None:
        iteration = settings["default_confusion_matrix_iteration"] or long_df["iteration"].max()

    target_names = ["Single (N=1)", "Multi (N>=2)"]
    sub = long_df[long_df["iteration"] == iteration].dropna(subset=["pred_label"])

    if sub.empty:
        print(f"No valid classifications at iteration {iteration} to plot.")
        return

    y_true = sub["true_label"].astype(int)
    y_pred = sub["pred_label"].astype(int)

    print(f"\nCLASSIFICATION REPORT (iteration {iteration}, {len(sub)} out-of-fold files):")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, normalize="true", labels=[0, 1])
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt=".1%", cmap="Greens",
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Encoder/NN: Normalized Confusion Matrix (iteration {iteration})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


# =========================================================
# 6. CORRECTNESS HEATMAP (identical to LM/Bayesian)
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
    ax.set_title("Encoder/NN: Correct (green) / Incorrect (red) per file, per iteration\n"
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
# 7. EXECUTION
# =========================================================
if __name__ == "__main__":
    out_dir = SETTINGS["simulation_dir"]

    long_df, fold_info_df = run_kfold_cv(SETTINGS)

    print("\n" + "=" * 70)
    print("PER-FOLD TRAINING SUMMARY")
    print("=" * 70)
    print(fold_info_df.to_string(index=False))
    fold_info_df.to_csv(os.path.join(out_dir, "Encoder_fold_training.csv"), index=False)

    long_df.to_csv(os.path.join(out_dir, "Encoder_predictions_long_raw.csv"), index=False)

    # --- RAW (unsmoothed) outputs, kept for direct before/after comparison ---
    metrics_df_raw = compute_time_resolved_metrics(long_df)
    print("\nRAW (unsmoothed) PERFORMANCE vs. INTEGRATION TIME (out-of-fold):")
    print(metrics_df_raw.to_string(index=False))
    metrics_df_raw.to_csv(os.path.join(out_dir, "Encoder_metrics_vs_iteration_raw.csv"), index=False)
    plot_metrics_vs_iteration(metrics_df_raw, save_path=os.path.join(out_dir, "Encoder_metrics_vs_iteration_raw.png"))

    # --- SMOOTHED outputs (this variant's change - see smooth_predictions_ema) ---
    smoothed_df = smooth_predictions_ema(long_df, alpha=SETTINGS["ema_alpha"])
    smoothed_df.to_csv(os.path.join(out_dir, "Encoder_predictions_long.csv"), index=False)

    metrics_df = compute_time_resolved_metrics(smoothed_df)
    print(f"\nSMOOTHED (EMA alpha={SETTINGS['ema_alpha']}) PERFORMANCE vs. INTEGRATION TIME (out-of-fold):")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(os.path.join(out_dir, "Encoder_metrics_vs_iteration.csv"), index=False)

    plot_metrics_vs_iteration(metrics_df, save_path=os.path.join(out_dir, "Encoder_metrics_vs_iteration.png"))

    plot_confusion_matrix_at_iteration(
        smoothed_df, save_path=os.path.join(out_dir, "Encoder_confusion_matrix_last_iter.png")
    )

    correctness_pivot = plot_correctness_heatmap(
        smoothed_df, save_path=os.path.join(out_dir, "Encoder_correctness_heatmap.png")
    )
    correctness_pivot.to_csv(os.path.join(out_dir, "Encoder_correctness_matrix.csv"))

    print("\nDone. All outputs saved to:", out_dir)
    print("(Both raw and EMA-smoothed metrics/predictions saved - compare "
          "Encoder_metrics_vs_iteration_raw.csv vs Encoder_metrics_vs_iteration.csv directly.)")