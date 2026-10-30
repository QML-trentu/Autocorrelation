"""
summarize_method_selection.py

Quantitative basis for choosing a classifier as a function of the acquisition
time, and a paired comparison of the majority-vote ensemble with the Bayesian
classifier alone.

For every seed written by run_seed_sweep.py (pooled out-of-fold predictions;
label 0 = single = positive class) it computes, at a set of iterations
(default 10, 30, 50, 60, 100, 200, 300):

  1. recall, precision and leakage of LM, Bayesian and MLP with 95% Wilson score
     intervals (Eq. (6) of the manuscript) -> method_selection_table.csv
     (one row per seed/method/iteration) and
     method_selection_summary.csv (primary seed with intervals, plus mean and
     range over all seeds; if iter_to_time_mapping_lookup.csv is present, the
     equivalent real time in seconds is added).
  2. the majority-vote ensemble (a dataset is called multi if at least two of the
     three methods call it multi), compared with the Bayesian classifier alone:
     number of true singles rescued by the vote (missed by Bayesian, caught by
     the vote), lost by the vote (caught by Bayesian, missed by the vote),
     exact two-sided McNemar (binomial) p-value on these discordant datasets,
     and the false positives of the two -> ensemble_vs_bayesian_paired.csv.

Usage:  python summarize_method_selection.py [--results-dir seed_sweep_results]
                                             [--mapping-csv iter_to_time_mapping_lookup.csv]
                                             [--iterations 10 30 50 60 100 200 300]
Reads:  <results-dir>/seed_<N>/{LM,Bayesian,MLP}_predictions_long.csv
Note:   the repeated seeds re-partition the same 5000 simulated datasets; they test
        stability and do not add independent samples, so p-values are reported per
        seed and are not combined.
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
from scipy.stats import binomtest

METHODS = ["LM", "Bayesian", "MLP"]
REFERENCE_SEED = 42
Z = 1.959964


def wilson(k, n, z=Z):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def key(f):
    return re.search(r"rep_(\d+_n\d+)", f).group(1)      # datasets matched by id (file names differ between methods)


def metrics(single, pred_single):
    tp, fn = (single & pred_single).sum(), (single & ~pred_single).sum()
    fp, tn = (~single & pred_single).sum(), (~single & ~pred_single).sum()
    out = dict(TP=int(tp), FN=int(fn), FP=int(fp), TN=int(tn))
    for name, k, n in (("Recall", tp, tp + fn), ("Precision", tp, tp + fp), ("Leakage", fp, fp + tn)):
        lo, hi = wilson(k, n)
        out[name], out[name + "_lo"], out[name + "_hi"] = (k / n if n else np.nan), lo, hi
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results-dir", default=os.path.join(here, "seed_sweep_results"))
    ap.add_argument("--mapping-csv", default=os.path.join(here, "iter_to_time_mapping_lookup.csv"))
    ap.add_argument("--iterations", type=int, nargs="+", default=[10, 30, 50, 60, 100, 200, 300])
    a = ap.parse_args()
    out_dir = os.path.abspath(a.results_dir)
    mp = pd.read_csv(a.mapping_csv).set_index("iteration")["representative_real_time_s"] if os.path.exists(a.mapping_csv) else None

    rows, paired = [], []
    for sdir in sorted(glob.glob(os.path.join(out_dir, "seed_[0-9]*")), key=lambda d: int(d.rsplit("_", 1)[1])):
        seed = int(sdir.rsplit("_", 1)[1])
        files = {m: os.path.join(sdir, f"{m}_predictions_long.csv") for m in METHODS}
        if not all(os.path.exists(f) for f in files.values()):
            print(f"[skip] seed {seed}: not all three methods finished"); continue
        P = {}
        for m, f in files.items():
            d = pd.read_csv(f)
            d = d[d.iteration.isin(a.iterations)].copy()
            d["id"] = d["file"].map(key)
            P[m] = d.set_index(["id", "iteration"])
        J = pd.concat({m: P[m]["pred_label"] for m in METHODS}, axis=1).join(P["LM"]["true_label"])
        for it, g in J.groupby(level="iteration"):
            single = (g["true_label"] == 0).values
            ps = {m: (g[m] == 0).values for m in METHODS}
            ps["Majority vote"] = (ps["LM"].astype(int) + ps["Bayesian"] + ps["MLP"]) >= 2   # >=2 of 3 say single <=> <=1 says multi
            for m, p in ps.items():
                rows.append(dict(seed=seed, method=m, iteration=int(it), **metrics(single, p)))
            b, v = ps["Bayesian"], ps["Majority vote"]
            rescued, lost = int((single & ~b & v).sum()), int((single & b & ~v).sum())
            paired.append(dict(seed=seed, iteration=int(it), singles_missed_Bayesian=int((single & ~b).sum()),
                               singles_missed_vote=int((single & ~v).sum()), rescued_by_vote=rescued, lost_by_vote=lost,
                               mcnemar_exact_p=(binomtest(rescued, rescued + lost, 0.5).pvalue if rescued + lost else 1.0),
                               FP_Bayesian=int((~single & b).sum()), FP_vote=int((~single & v).sum()),
                               FP_vote_only=int((~single & v & ~b).sum()), FP_Bayesian_only=int((~single & b & ~v).sum())))
        print(f"[ok] seed {seed}")
    T, Pr = pd.DataFrame(rows), pd.DataFrame(paired)
    if T.empty:
        raise SystemExit("no complete seeds found in " + out_dir)
    T.to_csv(os.path.join(out_dir, "method_selection_table.csv"), index=False)
    Pr.to_csv(os.path.join(out_dir, "ensemble_vs_bayesian_paired.csv"), index=False)

    ref = REFERENCE_SEED if REFERENCE_SEED in T.seed.values else T.seed.min()
    summ = []
    for (m, it), g in T.groupby(["method", "iteration"]):
        r = g[g.seed == ref].iloc[0]
        row = dict(method=m, iteration=it, time_s=(mp.get(it, np.nan) if mp is not None else np.nan))
        for k in ("Recall", "Precision", "Leakage"):
            row.update({f"{k}_seed{ref}": r[k], f"{k}_lo": r[k + "_lo"], f"{k}_hi": r[k + "_hi"],
                        f"{k}_mean": g[k].mean(), f"{k}_min": g[k].min(), f"{k}_max": g[k].max()})
        row["FP_seed%d" % ref] = r.FP; row["FN_seed%d" % ref] = r.FN
        summ.append(row)
    S = pd.DataFrame(summ).sort_values(["iteration", "method"])
    S.to_csv(os.path.join(out_dir, "method_selection_summary.csv"), index=False)

    print(f"\nPrimary seed {ref}; mean [min-max] over {T.seed.nunique()} seeds")
    for it in sorted(T.iteration.unique()):
        print(f"\niteration {it}" + (f" (~{mp.get(it):.1f} s)" if mp is not None and it in mp.index else ""))
        for m in METHODS + ["Majority vote"]:
            r = S[(S.method == m) & (S.iteration == it)].iloc[0]
            print(f"  {m:14s} recall {r[f'Recall_seed{ref}']:.4f} [{r.Recall_lo:.4f},{r.Recall_hi:.4f}]  "
                  f"precision {r[f'Precision_seed{ref}']:.4f}  leakage {r[f'Leakage_seed{ref}']:.4f} [{r.Leakage_lo:.4f},{r.Leakage_hi:.4f}]")
    print("\nMajority vote vs Bayesian alone (paired, per seed) at iteration", max(a.iterations))
    print(Pr[Pr.iteration == max(a.iterations)].to_string(index=False))


if __name__ == "__main__":
    main()
