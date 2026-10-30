# -*- coding: utf-8 -*-
"""
LM_analysis_fixed_tau1.py

Levenberg-Marquardt (LM) classification with the antibunching lifetime fixed
to a value calibrated beforehand, compared with the standard LM analysis
(all parameters free) on identical folds.

Everything except the fitted model is identical to
LM_analysis_Simulation_data_iterspecific_smoothed.py (same files, same
stratified 5-fold split with the same seed, same tail normalization and
Poisson weights, same per-iteration midpoint threshold calibrated on the
training files, same EMA smoothing of the margin). Three LM variants are
compared on exactly the same folds:

  LM_free        : the original LM fit (all parameters free, nonlinear LM).
  LM_tau1_A1     : tau1 FIXED to this fold's calibrated value, amplitude A1
                   as the ONLY fit parameter (pedestal A2 = 0). This is the
                   same per-histogram estimator that the Bayesian classifier
                   uses for each increment, but applied to the cumulative
                   histogram and followed by the usual LM decision rule.
  LM_tau1_A1_A2  : tau1 FIXED, amplitude A1 and pedestal A2 free (linear).

tau1 is calibrated exactly as in the Bayesian script: the median tau1 of the
full fit to the last iteration of the TRAIN files of each fold
(Bayesian_analysis_simulation_data_iterspecific.estimate_representative_tau1).
With tau1 fixed the model is linear in the amplitudes, so the fit is a closed-
form weighted least-squares solution (no iterative optimizer).

Optionally the Bayesian classifier is run on the same folds for reference.

Put this file in the same folder as the two scripts it imports and run it
from the folder that contains Simulation_Results/.
"""
import os
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import LM_analysis_Simulation_data_iterspecific_smoothed as lm
import Bayesian_analysis_simulation_data_iterspecific as bay

SETTINGS = dict(lm.SETTINGS)
SETTINGS.update({
    "simulation_dir": "Simulation_Results",
    "output_prefix": "LM_fixed_tau1",
    "run_bayesian_reference": True,
    "n_processes": 2,
})
VARIANTS = ["LM_free", "LM_tau1_A1", "LM_tau1_A1_A2"]


# ---------------------------------------------------------------------------
# per-file cache: normalized signal and Poisson sigma at the evaluated iterations
# ---------------------------------------------------------------------------
def _load_one(args):
    filepath, eval_iterations, settings = args
    df = pd.read_csv(filepath)
    tau = df["tau"].values
    base_m = np.abs(tau) >= settings["base_window_ns"]
    out = {}
    for it in eval_iterations:
        counts = df[f"iter_{it}"].values.astype(float)
        norm, sigma, ok = lm.tail_normalize(counts, base_m, settings["min_base_counts"])
        if not ok:
            out[it] = None
            continue
        g2_free, fit_ok = lm.fit_g2_zero(tau, norm, sigma=sigma)
        popt = None
        if it == max(eval_iterations):
            popt = bay.fit_g2_full(tau, norm, sigma)      # for the tau1 calibration
        out[it] = (norm, sigma, g2_free if fit_ok else np.nan, popt[1] if popt is not None else np.nan)
    return tau, out


def weighted_linear_g2zero(tau, norm, sigma, tau1, with_pedestal):
    """g2(0) from a weighted least-squares fit with tau1 fixed.
    y = 1 - g2 = A1*exp(-|t|/tau1) - A2  ->  g2(0) = 1 - A1 + A2."""
    x = np.exp(-np.abs(tau) / tau1)
    y = 1.0 - norm
    w = 1.0 / np.maximum(sigma, 1e-6) ** 2
    if not with_pedestal:
        A1 = np.sum(w * x * y) / np.sum(w * x * x)
        return 1.0 - A1
    X = np.column_stack([x, -np.ones_like(x)])
    XtW = X.T * w
    A1, A2 = np.linalg.solve(XtW @ X, XtW @ y)
    return 1.0 - A1 + A2


def _calibrate_and_classify(per_file, train_idx, test_idx, labels, files, eval_iterations, tau1, variant):
    """Returns long-format records for the test files of one fold."""
    def g2(f_idx, it):
        entry = per_file[f_idx][1][it]
        if entry is None:
            return np.nan
        tau, (norm, sigma, g2_free, _) = per_file[f_idx][0], entry
        if variant == "LM_free":
            return g2_free
        return weighted_linear_g2zero(tau, norm, sigma, tau1, with_pedestal=(variant == "LM_tau1_A1_A2"))

    # per-iteration threshold = midpoint of the class means in the training files
    thr = {}
    pooled = {0: [], 1: []}
    by_it = {it: {0: [], 1: []} for it in eval_iterations}
    for i in train_idx:
        for it in eval_iterations:
            v = g2(i, it)
            if np.isfinite(v):
                by_it[it][labels[i]].append(v)
                pooled[labels[i]].append(v)
    pm = {c: float(np.mean(pooled[c])) for c in (0, 1)}
    for it in eval_iterations:
        m0 = np.mean(by_it[it][0]) if len(by_it[it][0]) >= 5 else pm[0]
        m1 = np.mean(by_it[it][1]) if len(by_it[it][1]) >= 5 else pm[1]
        thr[it] = (m0 + m1) / 2
    rec = []
    for i in test_idx:
        for it in eval_iterations:
            v = g2(i, it)
            pred = None if not np.isfinite(v) else (0 if v < thr[it] else 1)
            rec.append([files[i], labels[i], it, thr[it], v, pred])
    return rec


def metrics_table(long_df):
    return lm.compute_time_resolved_metrics(long_df)


def main(settings=SETTINGS):
    directory = settings["simulation_dir"]
    files, labels = lm.list_raw_files_with_labels(directory)
    all_iters = lm.get_all_iterations(directory, files)
    last = max(all_iters)
    eval_iterations = list(range(settings["iteration_stride"], last + 1, settings["iteration_stride"]))
    if last not in eval_iterations:
        eval_iterations.append(last)
    print(f"{len(files)} files ({np.sum(labels == 0)} single, {np.sum(labels == 1)} multi); "
          f"{len(eval_iterations)} evaluated iterations")

    args = [(os.path.join(directory, f), eval_iterations, settings) for f in files]
    with Pool(settings["n_processes"]) as pool:
        per_file = pool.map(_load_one, args, chunksize=8)

    skf = StratifiedKFold(n_splits=settings["n_folds"], shuffle=True, random_state=settings["random_state"])
    records = {v: [] for v in VARIANTS}
    tau1_per_fold = []
    for fold, (tr, te) in enumerate(skf.split(files, labels)):
        # tau1 calibrated exactly as in the Bayesian script (median of the last-iteration fits, TRAIN files)
        t1 = [per_file[i][1][last][3] for i in tr if per_file[i][1][last] is not None]
        t1 = [v for v in t1 if np.isfinite(v)]
        tau1 = float(np.median(t1)) if t1 else 1.0
        tau1_per_fold.append(tau1)
        for v in VARIANTS:
            for r in _calibrate_and_classify(per_file, tr, te, labels, files, eval_iterations, tau1, v):
                records[v].append(r + [fold])
        print(f"  fold {fold}: calibrated tau1 = {tau1:.3f} ns")

    out = {}
    for v in VARIANTS:
        df = pd.DataFrame(records[v], columns=["file", "true_label", "iteration", "threshold", "g2_zero",
                                               "pred_label", "fold"])
        sm = lm.smooth_predictions_ema(df, alpha=settings["ema_alpha"])
        m = metrics_table(sm)
        m.insert(0, "variant", v)
        m_raw = metrics_table(df)
        m_raw.insert(0, "variant", v)
        out[v] = (sm, m, m_raw)
        sm.to_csv(f"{settings['output_prefix']}_{v}_predictions_long.csv", index=False)

    metrics = pd.concat([out[v][1] for v in VARIANTS])
    metrics.to_csv(f"{settings['output_prefix']}_metrics_vs_iteration.csv", index=False)
    pd.concat([out[v][2] for v in VARIANTS]).to_csv(
        f"{settings['output_prefix']}_metrics_vs_iteration_unsmoothed.csv", index=False)

    if settings["run_bayesian_reference"]:
        bay_long, _ = bay.run_kfold_cv(dict(settings))
        bm = lm.compute_time_resolved_metrics(bay_long)
        bm.insert(0, "variant", "Bayesian")
        bm.to_csv(f"{settings['output_prefix']}_Bayesian_reference_metrics.csv", index=False)
        bay_long.to_csv(f"{settings['output_prefix']}_Bayesian_predictions_long.csv", index=False)
        metrics = pd.concat([metrics, bm])

    show = metrics[metrics["Iteration"].isin([10, 30, 60, 100, 200, 300])]
    print("\nOut-of-fold metrics (LM variants EMA-smoothed as in the main text):")
    print(show.pivot_table(index=["Iteration", "variant"], values=["Recall", "Precision", "Leakage"])
          .round(4).to_string())
    print("\nCalibrated tau1 per fold (ns):", [round(t, 3) for t in tau1_per_fold])
    return metrics


if __name__ == "__main__":
    main()
