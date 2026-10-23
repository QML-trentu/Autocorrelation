"""
summarize_seed_convergence_overlap.py

Companion to summarize_seed_sweep.py. For every random seed written by
run_seed_sweep.py it recomputes, from the per-iteration predictions of the three
classifiers (LM, Bayesian, MLP):

  1. Convergence speed and curve smoothness (as in the paper's Table 2):
       - the first evaluated iteration at which precision reaches 0.99
         ("never reached" if it does not), and the highest precision attained;
       - the number of "reversals" of the precision-vs-iteration curve, i.e. the
         number of direction changes (up->down or down->up) between consecutive
         non-zero steps of the curve evaluated at iterations 10, 20, ..., 300;
       - the lowest precision after the first crossing of 0.99.
  2. Overlap in misclassified true-single datasets at iteration 300 (as in the
     paper's Table 3 / Figure S1): total, unique and shared misses per method,
     and the number of datasets missed by all three / exactly two / exactly one
     method. A true single is missed by the majority-vote ensemble if and only if
     at least two methods miss it, so (all three + exactly two) is the number of
     singles the ensemble misses.
  3. Persistence across seeds: how many distinct datasets each method misses in
     at least one seed, and how many it misses in every seed.

Usage:  python summarize_seed_convergence_overlap.py [--results-dir seed_sweep_results]
Reads:  <results-dir>/seed_<N>/{LM,Bayesian,MLP}_predictions_long.csv
Writes (into the results dir): seed_sweep_convergence.csv, seed_sweep_overlap.csv,
        seed_sweep_vote_counts.csv, seed_sweep_persistence.csv; prints a summary.
"""
import argparse, glob, os, re
from collections import Counter
import numpy as np, pandas as pd

METHODS = ["LM", "Bayesian", "MLP"]
THRESHOLD = 0.99
OVERLAP_ITER = 300


def file_id(name):
    """Dataset identifier shared by all methods (the three scripts read different
    file variants of the same simulated dataset, e.g. g2_raw_rep_7_n2 / g2_norm_rep_7_n2)."""
    return re.search(r"rep_(\d+_n\d+)", name).group(1)


def precision_curve(df):
    out = []
    for it, g in df.groupby("iteration"):
        s, p = g["true_label"].values == 0, g["pred_label"].values == 0   # label 0 = single (positive class)
        tp, fp = (s & p).sum(), (~s & p).sum()
        out.append((it, tp / (tp + fp) if tp + fp else np.nan))
    return np.array(out)


def reversals(prec):
    sign = np.sign(np.diff(prec))
    sign = sign[sign != 0]
    return int((sign[1:] != sign[:-1]).sum())


def missed_singles(df, it):
    g = df[df["iteration"] == it]
    return set(g.loc[(g["true_label"] == 0) & (g["pred_label"] != 0), "file"].map(file_id))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results-dir", default=os.path.join(here, "seed_sweep_results"))
    a = ap.parse_args()
    out_dir = os.path.abspath(a.results_dir)

    conv, ovl, votes, missed = [], [], [], {m: {} for m in METHODS}
    for sdir in sorted(glob.glob(os.path.join(out_dir, "seed_[0-9]*")), key=lambda d: int(d.rsplit("_", 1)[1])):
        seed = int(sdir.rsplit("_", 1)[1])
        files = {m: os.path.join(sdir, f"{m}_predictions_long.csv") for m in METHODS}
        if not all(os.path.exists(f) for f in files.values()):
            print(f"[skip] seed {seed}: not all three methods finished"); continue
        dfs = {m: pd.read_csv(f) for m, f in files.items()}

        for m, df in dfs.items():
            c = precision_curve(df)
            its, prec = c[:, 0].astype(int), c[:, 1]
            hit = np.where(prec >= THRESHOLD)[0]
            first = int(its[hit[0]]) if len(hit) else None
            conv.append(dict(Seed=seed, Method=m,
                             First_iteration_precision_ge_0_99=first if first is not None else "never reached",
                             Reversals=reversals(prec),
                             Max_precision=round(float(np.nanmax(prec)), 4),
                             Iteration_of_max=int(its[int(np.nanargmax(prec))]),
                             Min_precision_after_first_crossing=(round(float(np.nanmin(prec[hit[0]:])), 4) if len(hit) else None)))
            missed[m][seed] = missed_singles(df, OVERLAP_ITER)

        A, B, C = (missed[m][seed] for m in METHODS)
        for m, S in zip(METHODS, (A, B, C)):
            others = set().union(*[x for x in (A, B, C) if x is not S])
            ovl.append(dict(Seed=seed, Method=m, Total_missed=len(S),
                            Unique_to_method=len(S - others), Shared_with_other_method=len(S & others)))
        count = Counter(i for S in (A, B, C) for i in S)
        n3, n2, n1 = (sum(v == k for v in count.values()) for k in (3, 2, 1))
        votes.append(dict(Seed=seed, Missed_by_all_three=n3, Missed_by_exactly_two=n2,
                          Missed_by_exactly_one=n1, Union=len(count), Missed_by_majority_vote=n3 + n2))
    if not votes:
        raise SystemExit(f"No complete seed folders found in {out_dir}")

    seeds = sorted(missed["LM"])
    pers = []
    for m in METHODS:
        c = Counter(i for s in seeds for i in missed[m][s])
        pers.append(dict(Method=m, Seeds=len(seeds), Distinct_datasets_missed_in_any_seed=len(c),
                         Datasets_missed_in_every_seed=sum(v == len(seeds) for v in c.values())))
    a3 = {s: missed["LM"][s] & missed["Bayesian"][s] & missed["MLP"][s] for s in seeds}
    c3 = Counter(i for s in seeds for i in a3[s])
    pers.append(dict(Method="All three simultaneously", Seeds=len(seeds), Distinct_datasets_missed_in_any_seed=len(c3),
                     Datasets_missed_in_every_seed=sum(v == len(seeds) for v in c3.values())))

    tables = {"seed_sweep_convergence.csv": pd.DataFrame(conv), "seed_sweep_overlap.csv": pd.DataFrame(ovl),
              "seed_sweep_vote_counts.csv": pd.DataFrame(votes), "seed_sweep_persistence.csv": pd.DataFrame(pers)}
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    for name, t in tables.items():
        t.to_csv(os.path.join(out_dir, name), index=False)
        print(f"\n=== {name} ===\n{t.to_string(index=False)}")
    print(f"\nWrote {len(tables)} files to {out_dir}")


if __name__ == "__main__":
    main()
