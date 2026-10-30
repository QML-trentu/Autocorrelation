# -*- coding: utf-8 -*-
"""
Benchmark_computational_cost.py

Wall-clock cost of the three classifiers (LM, sequential Bayesian, MLP),
measured on ONE fold of the same 5-fold stratified split (seed 42) used in
the paper, with the code of the repository (the three analysis scripts are
imported unchanged; nothing is re-implemented here except the timing).

Two kinds of cost are reported:

  (1) ONE-OFF cost per fold, paid once on the training datasets:
        LM        : calibration of the per-iteration thresholds (one LM fit of
                    every training dataset at each of the 30 evaluated iterations)
        Bayesian  : calibration of tau1 (one full fit per training dataset, last
                    iteration) and of the class- and position-specific Gaussian
                    likelihoods (closed-form estimate for every super-chunk)
        MLP       : feature construction, scaling, oversampling and training
                    (MLPClassifier with early stopping)

  (2) COST OF CLASSIFYING A NEW DATASET once calibrated / trained:
        - the whole 30-point trajectory (iterations 10, 20, ..., 300)
        - one new histogram (the cost of one more update during an acquisition)

All timings are in-memory compute times: the CSV files are read once before
the timed section (the cost of reading files is not a property of the
method), and single-threaded (BLAS/OpenMP threads set to 1) so that the
numbers are comparable between methods and between machines.

The training/calibration cost of LM and of the Bayesian classifier is
strictly proportional to the number of training datasets (it is a loop over
datasets), so it is measured on a random subset of the training datasets
(N_CALIB_SUBSET) and reported per training dataset and for the full training
set of a fold (extrapolated linearly). The MLP is trained on the complete
training set of the fold (its cost is not simply proportional to the number of
datasets), and, for reference, also on the subset.

Usage (from the folder containing Simulation_Results/ with both g2_raw_* and
g2_norm_* files, next to the three analysis scripts):
    python Benchmark_computational_cost.py
Output: Benchmark_computational_cost.csv and a printed summary.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import time
import platform
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import LM_analysis_Simulation_data_iterspecific_smoothed as lm
import Bayesian_analysis_simulation_data_iterspecific as bay
import Encoder_analysis_simulation_data_smoothed as enc

CFG = {
    "simulation_dir": "Simulation_Results",
    "n_folds": 5,
    "random_state": 42,
    "n_calib_subset": 240,     # training datasets used to time the LM and Bayesian calibration
    "n_train_paper": 4000,     # training datasets per fold in the full 5,000-dataset study (for the extrapolation)
    "n_test_datasets": 20,     # held-out datasets used to time the classification
    "n_repeats": 5,            # repetitions of each classification timing (median reported)
    "output_csv": "Benchmark_computational_cost.csv",
}

# ---------------------------------------------------------------------------
# in-memory cache of the CSV files, so that the timed sections do not include disk I/O
# ---------------------------------------------------------------------------
_orig_read_csv = pd.read_csv
_cache = {}


def _cached_read_csv(path, *a, **k):
    return _cache[path] if path in _cache else _orig_read_csv(path, *a, **k)


def _compact(df):
    """DataFrame with float64 tau and ONE float32 block for all histogram columns (low memory)."""
    cols = [c for c in df.columns if c != "tau"]
    out = pd.DataFrame(df[cols].to_numpy(dtype=np.float32), columns=cols)
    out.insert(0, "tau", df["tau"].to_numpy(dtype=np.float64))
    return out


def _normalise_like_step02(raw_df, t_max=100.0):
    """Same operation as normalize_final_g2 in Step_02: counts / mean(counts[|tau| > 0.8 t_max])."""
    tau = raw_df["tau"].to_numpy()
    m = np.abs(tau) > 0.8 * t_max
    cols = [c for c in raw_df.columns if c != "tau"]
    a = raw_df[cols].to_numpy(dtype=np.float64)
    b = a[m].mean(axis=0)
    a = np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    out = pd.DataFrame(a.astype(np.float32), columns=cols)
    out.insert(0, "tau", tau.astype(np.float64))
    return out


def preload(paths):
    for p in paths:
        if p in _cache:
            continue
        if os.path.exists(p):
            df = _compact(_orig_read_csv(p))
        else:
            # g2_norm_* file not on disk: derive it in memory from the g2_raw_* file (identical operation to Step_02)
            df = _normalise_like_step02(_orig_read_csv(p.replace("g2_norm_", "g2_raw_")))
        _cache[p] = df


def drop(paths):
    for p in paths:
        _cache.pop(p, None)


def timed(fn, repeats=1):
    ts = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def main(cfg=CFG):
    d = cfg["simulation_dir"]
    pd.read_csv = _cached_read_csv                      # patched globally: used by the three imported scripts
    raw_files, labels = lm.list_raw_files_with_labels(d)
    norm_files = [f.replace("g2_raw_", "g2_norm_") for f in raw_files]      # same datasets, normalised
    skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["random_state"])
    train_idx, test_idx = next(iter(skf.split(raw_files, labels)))      # fold 0
    rng = np.random.RandomState(0)
    sub_idx = rng.choice(train_idx, size=min(cfg["n_calib_subset"], len(train_idx)), replace=False)
    test_sel = rng.choice(test_idx, size=min(cfg["n_test_datasets"], len(test_idx)), replace=False)

    all_iters = lm.get_all_iterations(d, raw_files)
    stride = 10
    eval_its = sorted(set(range(stride, max(all_iters) + 1, stride)) | {max(all_iters)})
    S_lm = dict(lm.SETTINGS); S_bay = dict(bay.SETTINGS); S_enc = dict(enc.SETTINGS)
    for S in (S_lm, S_bay, S_enc):
        S["simulation_dir"] = d
    rows = []
    P = lambda f: os.path.join(d, f)
    print(f"{len(raw_files)} datasets; fold 0: {len(train_idx)} train ({int(np.sum(labels[train_idx] == 0))} single), "
          f"calibration subset {len(sub_idx)}, {len(test_sel)} test datasets timed.")

    # ============================ LM ============================
    sub_files = [(raw_files[i], labels[i]) for i in sub_idx]
    preload([P(f) for f, _ in sub_files] + [P(raw_files[i]) for i in test_sel])
    t_cal, _ = timed(lambda: lm.calibrate_thresholds_per_iteration(sub_files, d, S_lm, eval_its))
    n_fits_cal = len(sub_files) * len(eval_its)
    # classification of new datasets
    base_m = None
    def lm_traj(f):
        df = pd.read_csv(P(f)); tau = df["tau"].values
        bm = np.abs(tau) >= S_lm["base_window_ns"]
        for it in eval_its:
            counts = df[f"iter_{it}"].values.astype(float)
            norm, sigma, ok = lm.tail_normalize(counts, bm, S_lm["min_base_counts"])
            if ok:
                lm.fit_g2_zero(tau, norm, sigma=sigma)
    t_traj = np.median([timed(lambda f=raw_files[i]: lm_traj(f), cfg["n_repeats"])[0] for i in test_sel])
    t_one = t_traj / len(eval_its)        # mean time of one LM fit of a cumulative histogram
    rows.append(dict(method="LM", calib_n=len(sub_files), calib_total_s=t_cal, calib_per_dataset_ms=1e3 * t_cal / len(sub_files),
                     calib_paper_size_s_extrapolated=t_cal / len(sub_files) * cfg["n_train_paper"],
                     classify_trajectory_ms=1e3 * t_traj, classify_one_histogram_ms=1e3 * t_one,
                     note=f"{n_fits_cal} nonlinear fits in calibration; each histogram is fitted independently"))
    print(rows[-1])

    # ============================ Bayesian ============================
    t_cal, (hyps, tau1, _, _) = timed(lambda: bay.calibrate_hypotheses(sub_files, d, S_bay))
    def bay_traj(f):
        df = pd.read_csv(P(f)); tau = df["tau"].values
        cols = bay.get_iter_columns(df)
        bm = np.abs(tau) >= S_bay["base_window_ns"]
        inc = bay.compute_chunk_increments(df, cols)
        grouped = bay.aggregate_increments(inc, S_bay["iteration_stride"])
        g2 = bay.chunk_g2_estimates(grouped, tau, bm, tau1, S_bay["min_base_counts"])
        bay.sequential_posterior(g2, hyps)
    def bay_one(f):
        # one more update during an acquisition: new super-chunk of increments -> estimate -> posterior step
        df = pd.read_csv(P(f)); tau = df["tau"].values
        cols = bay.get_iter_columns(df)
        bm = np.abs(tau) >= S_bay["base_window_ns"]
        inc = bay.compute_chunk_increments(df, cols)
        grouped = bay.aggregate_increments(inc, S_bay["iteration_stride"])[:, 5:6]
        def go():
            g2 = bay.chunk_g2_estimates(grouped, tau, bm, tau1, S_bay["min_base_counts"])
            return bay.sequential_posterior(g2, [hyps[5]])
        return timed(go, cfg["n_repeats"])[0]
    t_traj = np.median([timed(lambda f=raw_files[i]: bay_traj(f), cfg["n_repeats"])[0] for i in test_sel])
    t_one = np.median([bay_one(raw_files[i]) for i in test_sel])
    rows.append(dict(method="Bayesian", calib_n=len(sub_files), calib_total_s=t_cal, calib_per_dataset_ms=1e3 * t_cal / len(sub_files),
                     calib_paper_size_s_extrapolated=t_cal / len(sub_files) * cfg["n_train_paper"],
                     classify_trajectory_ms=1e3 * t_traj, classify_one_histogram_ms=1e3 * t_one,
                     note=f"one nonlinear fit per training dataset (tau1) + closed-form estimates; tau1={tau1:.3f} ns"))
    print(rows[-1])
    drop([P(f) for f, _ in sub_files] + [P(raw_files[i]) for i in test_sel])

    # ============================ MLP ============================
    preload([P(norm_files[0])])
    tau_ref = _cache[P(norm_files[0])]["tau"].values
    def features_streaming(idx):
        """Feature matrix built file by file (file reading is NOT timed and nothing is kept in memory
        except the features), identical to enc.build_feature_matrix. Returns X, y, time spent on features."""
        X_list, y_list, t_feat = [], [], 0.0
        for i in idx:
            if os.path.exists(P(norm_files[i])):
                df = _orig_read_csv(P(norm_files[i]))
            else:                      # derive the normalised file from the raw one (same operation as Step_02)
                df = _normalise_like_step02(_orig_read_csv(P(raw_files[i])))
            cols = [df[f"iter_{it}"].values.astype(float) for it in eval_its]
            t0 = time.perf_counter()
            feats = [enc.two_tier_features(tau_ref, x, S_enc["near_window_ns"], S_enc["n_far_bins_per_side"]) for x in cols]
            t_feat += time.perf_counter() - t0
            X_list.extend(feats); y_list.extend([labels[i]] * len(feats))
        return np.array(X_list), np.array(y_list), t_feat

    def train_mlp(X, y):
        def go():
            sc = StandardScaler(); Xs = sc.fit_transform(X)
            Xb, yb = enc.oversample_minority(Xs, y, S_enc["random_state"])
            clf = MLPClassifier(hidden_layer_sizes=S_enc["hidden_layer_sizes"], alpha=S_enc["alpha"], max_iter=S_enc["max_iter"],
                                early_stopping=S_enc["early_stopping"], validation_fraction=S_enc["validation_fraction"],
                                n_iter_no_change=S_enc["n_iter_no_change"], random_state=S_enc["random_state"])
            clf.fit(Xb, yb)
            return clf, sc
        return go
    mlp_runs = {}
    for tag, idx in (("subset", sub_idx), ("full fold", train_idx)):
        X, y, t_feat = features_streaming(idx)
        t_fit, (clf, sc) = timed(train_mlp(X, y))          # scaling + oversampling + MLP training
        t_total = t_feat + t_fit
        mlp_runs[tag] = (len(idx), t_total, t_feat, clf, sc)
        print(f"MLP {tag}: n={len(idx)} total {t_total:.1f} s (features {t_feat:.1f} s, scaling+oversampling+training {t_fit:.1f} s), {clf.n_iter_} epochs")
    n_full, t_full, t_feat_full, clf, sc = mlp_runs["full fold"]
    test_paths = [P(norm_files[i]) for i in test_sel]; preload(test_paths)
    def mlp_traj(f):
        df = pd.read_csv(P(f))
        X = np.array([enc.two_tier_features(tau_ref, df[f"iter_{it}"].values.astype(float), S_enc["near_window_ns"], S_enc["n_far_bins_per_side"]) for it in eval_its])
        clf.predict_proba(sc.transform(X))
    def mlp_one(f):
        df = pd.read_csv(P(f)); x = df[f"iter_{max(eval_its)}"].values.astype(float)
        def go():
            X = enc.two_tier_features(tau_ref, x, S_enc["near_window_ns"], S_enc["n_far_bins_per_side"])[None, :]
            return clf.predict_proba(sc.transform(X))
        return timed(go, cfg["n_repeats"])[0]
    t_traj = np.median([timed(lambda f=norm_files[i]: mlp_traj(f), cfg["n_repeats"])[0] for i in test_sel])
    t_one = np.median([mlp_one(norm_files[i]) for i in test_sel])
    ns, ts = mlp_runs["subset"][0], mlp_runs["subset"][1]
    rows.append(dict(method="MLP", calib_n=n_full, calib_total_s=t_full, calib_per_dataset_ms=1e3 * t_full / n_full,
                     calib_paper_size_s_extrapolated=np.nan,
                     classify_trajectory_ms=1e3 * t_traj, classify_one_histogram_ms=1e3 * t_one,
                     note=f"measured on the full fold ({n_full} datasets; {clf.n_iter_} epochs); on {ns} datasets: {ts:.1f} s"))
    print(rows[-1])

    out = pd.DataFrame(rows)
    out.to_csv(cfg["output_csv"], index=False)
    env = (f"Python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, scikit-learn {sklearn.__version__}; "
           f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, single-threaded")
    print("\n" + env)
    print(out.round(3).to_string(index=False))
    with open(cfg["output_csv"].replace(".csv", "_environment.txt"), "w") as fh:
        fh.write(env + "\n")
    return out


if __name__ == "__main__":
    main()
