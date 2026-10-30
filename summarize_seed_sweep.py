"""
summarize_seed_sweep.py

Aggregate the per-seed predictions written by run_seed_sweep.py into a summary
of how stable the results are across different random train/test splits.

For every seed it recomputes recall / precision / leakage exactly as in the
paper's Table 1 (pooled out-of-fold predictions at iteration 10 and 300), plus
the majority-vote ensemble at iteration 300 (see summarize_method_selection.py
for the full comparison). It then reports, across
seeds: mean, SD, min and max of each metric, and for how many seeds each
method ranking discussed in the paper holds.

Usage:  python summarize_seed_sweep.py [--results-dir seed_sweep_results]
Writes: seed_sweep_per_seed.csv, seed_sweep_summary.csv, seed_sweep_rankings.csv,
        seed_sweep_figure.png, and prints a ready-to-adapt summary.
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["LM", "Bayesian", "MLP"]
COLORS = {"LM": "#1f77b4", "Bayesian": "#d62728", "MLP": "#2ca02c"}
ITERS = [10, 300]


def metrics(sub):
    y, p = sub["true_label"].values.astype(int), sub["pred_label"].values.astype(int)
    s, m, ps = y == 0, y == 1, p == 0
    TP, FN, FP, TN = (s & ps).sum(), (s & ~ps).sum(), (m & ps).sum(), (m & ~ps).sum()
    d = lambda a, b: a / b if b else np.nan
    return dict(Recall=d(TP, TP + FN), Precision=d(TP, TP + FP), Leakage=d(FP, FP + TN),
                FN=int(FN), FP=int(FP))


def key(s):
    return s.str.extract(r"(rep_\d+_n\d+)")[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results-dir", default=os.path.join(here, "seed_sweep_results"))
    a = ap.parse_args()
    os.chdir(a.results_dir)       # summary files are written next to the per-seed folders
    a.results_dir = "."

    rows, votes = [], []
    seeds = []
    for sdir in sorted(glob.glob(os.path.join(a.results_dir, "seed_[0-9]*"))):
        seed = int(sdir.rsplit("_", 1)[1])
        files = {m: os.path.join(sdir, f"{m}_predictions_long.csv") for m in METHODS}
        if not all(os.path.exists(f) for f in files.values()):
            print(f"[skip] seed {seed}: not all three methods finished"); continue
        seeds.append(seed)
        dfs = {m: pd.read_csv(f) for m, f in files.items()}
        for m, df in dfs.items():
            for it in ITERS:
                rows.append(dict(Seed=seed, Method=m, Iteration=it, **metrics(df[df.iteration == it])))
        # majority vote at 300 (>=2 of 3 say multi), as in summarize_method_selection.py
        base = None
        for m, df in dfs.items():
            d = df[df.iteration == 300][["file", "true_label", "pred_label"]].copy()
            d["k"] = key(d["file"]); d = d.rename(columns={"pred_label": m})
            base = d[["k", "true_label", m]] if base is None else base.merge(d[["k", m]], on="k")
        base["pred_label"] = (base[METHODS].sum(axis=1) >= 2).astype(int)
        votes.append(dict(Seed=seed, Method="Majority vote", Iteration=300, **metrics(base)))

    if not seeds:
        raise SystemExit("No complete seeds found.")
    per = pd.DataFrame(rows + votes).sort_values(["Iteration", "Method", "Seed"])
    per.to_csv("seed_sweep_per_seed.csv", index=False)

    g = per.groupby(["Method", "Iteration"])[["Recall", "Precision", "Leakage"]]
    summ = g.agg(["mean", "std", "min", "max"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ = summ.reset_index()
    summ.insert(2, "n_seeds", len(seeds))
    summ.to_csv("seed_sweep_summary.csv", index=False)

    # ranking claims, per seed
    w = per[per.Method.isin(METHODS)].pivot_table(index=["Seed", "Iteration"], columns="Method",
                                                  values=["Recall", "Precision", "Leakage"])
    rk = []
    def claim(name, cond):
        rk.append(dict(Claim=name, Seeds_holding=int(cond.sum()), Seeds_total=len(cond)))
    r10, r300 = w.xs(10, level="Iteration"), w.xs(300, level="Iteration")
    claim("It.10: MLP leakage < LM and Bayesian", (r10[("Leakage", "MLP")] < r10[("Leakage", "LM")]) & (r10[("Leakage", "MLP")] < r10[("Leakage", "Bayesian")]))
    claim("It.10: MLP recall < LM and Bayesian", (r10[("Recall", "MLP")] < r10[("Recall", "LM")]) & (r10[("Recall", "MLP")] < r10[("Recall", "Bayesian")]))
    claim("It.10: MLP precision > LM and Bayesian", (r10[("Precision", "MLP")] > r10[("Precision", "LM")]) & (r10[("Precision", "MLP")] > r10[("Precision", "Bayesian")]))
    claim("It.300: LM has highest recall", (r300[("Recall", "LM")] > r300[("Recall", "Bayesian")]) & (r300[("Recall", "LM")] > r300[("Recall", "MLP")]))
    claim("It.300: LM recall > Bayesian recall", r300[("Recall", "LM")] > r300[("Recall", "Bayesian")])
    claim("It.300: Bayesian recall > MLP recall", r300[("Recall", "Bayesian")] > r300[("Recall", "MLP")])
    claim("It.300: Bayesian leakage <= MLP leakage <= LM leakage", (r300[("Leakage", "Bayesian")] <= r300[("Leakage", "MLP")]) & (r300[("Leakage", "MLP")] <= r300[("Leakage", "LM")]))
    claim("It.300: Bayesian precision >= MLP precision >= LM precision", (r300[("Precision", "Bayesian")] >= r300[("Precision", "MLP")]) & (r300[("Precision", "MLP")] >= r300[("Precision", "LM")]))
    pd.DataFrame(rk).to_csv("seed_sweep_rankings.csv", index=False)

    # figure: per-seed points + mean, 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    mets = ["Recall", "Precision", "Leakage"]
    for ax, it in zip(axes, ITERS):
        for i, m in enumerate(METHODS):
            for j, met in enumerate(mets):
                v = per[(per.Method == m) & (per.Iteration == it)][met].values
                x = j + (i - 1) * 0.26
                ax.scatter(np.full(len(v), x) + np.linspace(-0.04, 0.04, len(v)), v, s=14, color=COLORS[m],
                           alpha=0.7, label=m if j == 0 else None, zorder=3)
                ax.hlines(v.mean(), x - 0.1, x + 0.1, color="black", lw=1.5, zorder=4)
        ax.set_xticks(range(3)); ax.set_xticklabels(mets)
        ax.set_title(f"Iteration {it}  ({len(seeds)} random splits)")
        ax.grid(axis="y", alpha=0.25); ax.set_ylabel("Value")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    plt.tight_layout(); plt.savefig("seed_sweep_figure.png", dpi=300, bbox_inches="tight")

    pd.set_option("display.width", 200)
    print(f"\nSeeds used ({len(seeds)}): {seeds}\n")
    print(summ.round(4).to_string(index=False))
    print("\nRanking claims across seeds:\n" + pd.DataFrame(rk).to_string(index=False))
    if 42 in seeds:
        print("\nSeed 42 vs the paper's Table 1 (should match exactly):")
        print(per[(per.Seed == 42) & per.Method.isin(METHODS)][["Method", "Iteration", "Recall", "Precision", "Leakage"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
