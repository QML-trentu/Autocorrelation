"""
summarize_time_to_operating_point.py

Expresses the convergence speed of the three classifiers as the acquisition time
each one needs to reach a given operating point, instead of as an abstract
iteration number. For every seed written by run_seed_sweep.py it finds, per
method, the first evaluated iteration from which a criterion is met and stays met
at all later evaluated iterations (iterations 10, 20, ..., 300; pooled out-of-fold
predictions):

    leakage <= 1%         leakage <= 0.5%
    precision >= 0.99     precision >= 0.99 and recall >= 0.95

If a lookup of simulated iteration -> equivalent real integration time is
available (iter_to_time_mapping_lookup.csv, written by
Compute_iter_to_integration_time_mapping.py; column representative_real_time_s),
the iteration is also converted to seconds. This mapping is representative of the
emitters and detectors used to build it, not universal: the baseline count per
histogram bin scales with the product of the two detector count rates, so a
dimmer emitter needs proportionally more real time for the same iteration.

Usage:  python summarize_time_to_operating_point.py [--results-dir seed_sweep_results]
                                                    [--mapping-csv iter_to_time_mapping_lookup.csv]
Reads:  <results-dir>/seed_<N>/{LM,Bayesian,MLP}_predictions_long.csv
Writes: <results-dir>/seed_sweep_time_to_operating_point.csv (per seed) and
        seed_sweep_time_to_operating_point_summary.csv (first seed vs range over seeds)
"""
import argparse, glob, os
import numpy as np, pandas as pd

METHODS = ["LM", "Bayesian", "MLP"]
CRITERIA = {
    "Leakage <= 1%": lambda d: d.Leakage <= 0.01,
    "Leakage <= 0.5%": lambda d: d.Leakage <= 0.005,
    "Precision >= 0.99": lambda d: d.Precision >= 0.99,
    "Precision >= 0.99 and Recall >= 0.95": lambda d: (d.Precision >= 0.99) & (d.Recall >= 0.95),
}
REFERENCE_SEED = 42


def curve(df):
    rows = []
    for it, g in df.groupby("iteration"):
        s, p = g["true_label"].values == 0, g["pred_label"].values == 0   # label 0 = single (positive class)
        tp, fn, fp, tn = (s & p).sum(), (s & ~p).sum(), (~s & p).sum(), (~s & ~p).sum()
        rows.append(dict(iteration=int(it), Recall=tp / (tp + fn), Precision=tp / (tp + fp) if tp + fp else np.nan,
                         Leakage=fp / (fp + tn)))
    return pd.DataFrame(rows).set_index("iteration")


def sustained_from(ok):
    """first index from which the boolean series is True for all later entries (None if never)."""
    v, idx = ok.values, ok.index.values
    for i in range(len(v)):
        if v[i:].all():
            return int(idx[i])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results-dir", default=os.path.join(here, "seed_sweep_results"))
    ap.add_argument("--mapping-csv", default=os.path.join(here, "iter_to_time_mapping_lookup.csv"))
    a = ap.parse_args()
    out_dir = os.path.abspath(a.results_dir)
    mp = None
    if os.path.exists(a.mapping_csv):
        mp = pd.read_csv(a.mapping_csv).set_index("iteration")["representative_real_time_s"]
    else:
        print(f"[note] {a.mapping_csv} not found: reporting iterations only")

    rows = []
    for sdir in sorted(glob.glob(os.path.join(out_dir, "seed_[0-9]*")), key=lambda d: int(d.rsplit("_", 1)[1])):
        seed = int(sdir.rsplit("_", 1)[1])
        files = {m: os.path.join(sdir, f"{m}_predictions_long.csv") for m in METHODS}
        if not all(os.path.exists(f) for f in files.values()):
            print(f"[skip] seed {seed}: not all three methods finished"); continue
        for m, f in files.items():
            c = curve(pd.read_csv(f))
            for name, cond in CRITERIA.items():
                it = sustained_from(cond(c))
                rows.append(dict(Seed=seed, Method=m, Criterion=name, Iteration=it,
                                 Time_s=(round(float(mp[it]), 1) if (mp is not None and it is not None) else None)))
    if not rows:
        raise SystemExit(f"No complete seed folders found in {out_dir}")
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(out_dir, "seed_sweep_time_to_operating_point.csv"), index=False)

    summ = []
    ref_seed = REFERENCE_SEED if REFERENCE_SEED in set(per.Seed) else int(per.Seed.min())
    for (m, name), g in per.groupby(["Method", "Criterion"], sort=False):
        r = g[g.Seed == ref_seed].iloc[0]
        ok = g.dropna(subset=["Iteration"])
        summ.append(dict(Method=m, Criterion=name, Reference_seed=ref_seed,
                         Reference_iteration=r.Iteration, Reference_time_s=r.Time_s,
                         Seeds_reaching=f"{len(ok)}/{len(g)}",
                         Iteration_min=ok.Iteration.min() if len(ok) else None, Iteration_max=ok.Iteration.max() if len(ok) else None,
                         Time_min_s=ok.Time_s.min() if len(ok) else None, Time_max_s=ok.Time_s.max() if len(ok) else None))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(out_dir, "seed_sweep_time_to_operating_point_summary.csv"), index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(summ.to_string(index=False))
    print(f"\nWrote seed_sweep_time_to_operating_point.csv and ..._summary.csv to {out_dir}")


if __name__ == "__main__":
    main()
