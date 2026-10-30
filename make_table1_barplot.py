"""
make_table1_barplot.py

Two-panel bar plot of classifier performance at the sparsest (iteration 10)
and most-converged (iteration 300) acquisition conditions, grouped by metric
(Recall / Precision / Leakage), with 95% Wilson score confidence-interval error
bars. It presents the same numbers as the paper's Table 1 in graphical form.

Self-contained: computes everything from the three per-file, per-iteration
prediction logs that each classifier script already writes out
(LM_predictions_long.csv, Bayesian_predictions_long.csv,
Encoder_predictions_long.csv), rather than depending on any previously
computed intermediate file. Re-run any time those three CSVs are
regenerated to reproduce the figure.

Wilson score interval is used instead of the normal approximation because
the normal approximation collapses to zero width whenever the observed
proportion is exactly 0 or 1 - which happens routinely for Leakage here
(e.g. Bayesian's Leakage = 0.0000 at iteration 300).

Usage:
    python make_table1_barplot.py [--data-dir DIR] [--out FILE]

Expects DIR (default: current directory) to contain:
    LM_predictions_long.csv
    Bayesian_predictions_long.csv
    Encoder_predictions_long.csv
each with columns: file, true_label, fold, iteration, g2_zero, pred_label
(true_label / pred_label convention: 0 = Single (N=1), 1 = Multi (N>=2)).
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METHOD_FILES = {
    "LM": "LM_predictions_long.csv",
    "Bayesian": "Bayesian_predictions_long.csv",
    "MLP": "Encoder_predictions_long.csv",
}
METHOD_COLOR = {"LM": "#1f77b4", "Bayesian": "#d62728", "MLP": "#2ca02c"}
METHODS = ["LM", "Bayesian", "MLP"]
METRICS = ["Recall", "Precision", "Leakage"]
ITERATIONS = [10, 300]
ITERATION_TITLES = {10: "Sparsest (iteration 10)", 300: "Most-converged (iteration 300)"}


def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score 95% CI for a binomial proportion k/n. Returns (p, lo, hi)."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def metrics_with_ci(sub):
    """sub: rows for ONE method at ONE iteration, with true_label/pred_label
    columns (0 = Single, 1 = Multi). Returns a dict of Recall/Precision/
    Leakage point estimates plus their Wilson 95% CI bounds."""
    y_true = sub["true_label"].values
    y_pred = sub["pred_label"].values

    single_mask = (y_true == 0)
    multi_mask = (y_true == 1)
    pred_single = (y_pred == 0)

    TP = int(np.sum(single_mask & pred_single))
    FN = int(np.sum(single_mask & ~pred_single))
    FP = int(np.sum(multi_mask & pred_single))
    TN = int(np.sum(multi_mask & ~pred_single))

    rec = wilson_ci(TP, TP + FN)
    prec = wilson_ci(TP, TP + FP)
    leak = wilson_ci(FP, FP + TN)

    return dict(
        Recall=rec[0], Recall_lo=rec[1], Recall_hi=rec[2],
        Precision=prec[0], Precision_lo=prec[1], Precision_hi=prec[2],
        Leakage=leak[0], Leakage_lo=leak[1], Leakage_hi=leak[2],
        TP=TP, FN=FN, FP=FP, TN=TN,
    )


def build_table(data_dir):
    rows = []
    for method, fname in METHOD_FILES.items():
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path)
        for it in ITERATIONS:
            sub = df[df["iteration"] == it]
            m = metrics_with_ci(sub)
            rows.append(dict(Method=method, Iteration=it, **m))
    return pd.DataFrame(rows)


def add_leakage_log_inset(ax, sub):
    """
    Small log-scale inset zooming in on just the Leakage group, for a panel
    where Leakage is too small to read on the panel's main linear scale.

    Exact-zero Leakage values can't be placed on a log axis, so - exactly
    matching the convention used in the leakage-versus-iteration figure (Figure 4) - they are
    floored to half the smallest nonzero Leakage value observed here, with
    a dotted horizontal line marking that floor so it reads as "below
    detection" rather than a real measured value.
    """
    vals = sub.loc[METHODS, "Leakage"].values.astype(float)
    lo = sub.loc[METHODS, "Leakage_lo"].values.astype(float)
    hi = sub.loc[METHODS, "Leakage_hi"].values.astype(float)

    nonzero = vals[vals > 0]
    floor = (nonzero.min() / 2) if len(nonzero) else 1e-4
    plot_vals = np.where(vals <= 0, floor, vals)
    plot_lo = np.where(lo <= 0, floor, lo)

    # Right side of the panel: that's the empty region here, since the
    # Recall/Precision bars are tall on the left/middle and the (linear-
    # scale) Leakage bars are nearly invisible on the right. Positioned to
    # start at/after the Leakage group's own left edge (not merely "to the
    # right of center") so neither the inset box nor its y-tick labels
    # (which render just outside its left edge) land on top of the much
    # taller Precision bars immediately to its left.
    inset = ax.inset_axes([0.72, 0.07, 0.26, 0.40])

    # Set the log scale and an explicit y-range BEFORE drawing the bars.
    # bar()'s default bottom=0 is undefined on a log axis (log(0) = -inf);
    # left as the default, matplotlib stretches every bar down to whatever
    # floor it can find, which is what produced the garbled, axis-spanning
    # bars/error-whiskers in the first version of this inset. Giving an
    # explicit, sensible `bottom` avoids that entirely.
    axis_bottom = floor / 3
    axis_top = max(hi.max(), plot_vals.max()) * 1.8
    inset.set_yscale("log")
    inset.set_ylim(axis_bottom, axis_top)

    x = np.arange(len(METHODS))
    colors = [METHOD_COLOR[m] for m in METHODS]
    inset.bar(x, plot_vals, width=0.6, bottom=axis_bottom, color=colors,
              edgecolor="white", linewidth=0.5, zorder=3)

    err_lo = np.clip(plot_vals - plot_lo, 1e-9, None)
    err_hi = np.clip(hi - plot_vals, 0, None)
    inset.errorbar(x, plot_vals, yerr=[err_lo, err_hi], fmt="none",
                    ecolor="black", elinewidth=0.8, capsize=2, capthick=0.8, zorder=5)

    inset.axhline(floor, color="black", linestyle=":", linewidth=0.8, zorder=2)
    inset.set_xticks(x)
    inset.set_xticklabels(METHODS, fontsize=6)
    inset.tick_params(axis="y", labelsize=6)
    inset.set_title("Leakage (log scale)", fontsize=6.5, pad=3)
    inset.patch.set_alpha(0.96)
    return inset


def plot_two_panel_barplot(table, out_path, leakage_log_inset=True):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)

    for ax, it in zip(axes, ITERATIONS):
        sub = table[table["Iteration"] == it].set_index("Method").loc[METHODS]

        x = np.arange(len(METRICS))
        width = 0.26
        for i, method in enumerate(METHODS):
            vals = sub.loc[method, METRICS].values.astype(float)
            lo = sub.loc[method, [m + "_lo" for m in METRICS]].values.astype(float)
            hi = sub.loc[method, [m + "_hi" for m in METRICS]].values.astype(float)
            err_lo = np.clip(vals - lo, 0, None)
            err_hi = np.clip(hi - vals, 0, None)
            offset = (i - 1) * width
            ax.bar(x + offset, vals, width, color=METHOD_COLOR[method], label=method,
                   edgecolor="white", linewidth=0.6)
            ax.errorbar(x + offset, vals, yerr=[err_lo, err_hi], fmt="none",
                         ecolor="black", elinewidth=1.0, capsize=3, capthick=1.0, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(METRICS)
        ax.set_ylim(0, 1.05)

        # Axes flipped relative to the usual bottom/left convention - x-axis
        # (ticks + label) on top, y-axis (ticks + label) on the right - and
        # all four spines kept visible, so the panel reads as a closed box
        # rather than the open bottom-left-only frame.
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("Value")
        ax.set_title(ITERATION_TITLES[it], fontsize=11, pad=28)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)

        if leakage_log_inset and sub.loc[METHODS, "Leakage_hi"].max() < 0.05:
            add_leakage_log_inset(ax, sub)

    # Upper-right corner of panel 1 is empty (Leakage bars are short there),
    # unlike upper-left where the tall Recall bars sit.
    axes[0].legend(loc="upper right", fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


def build_caption(table):
    """
    Short caption text for the figure. Exact values (with their 95% Wilson
    CIs) are NOT repeated here - they live in Table 1, which this figure
    accompanies - only a pointer to the leakage floor
    convention is included, since that one detail isn't otherwise written
    down anywhere.
    """
    return (
        "Figure X. Recall, precision, and leakage at the sparsest (iteration 10) and "
        "most-converged (iteration 300) integration times (see Table 1 for exact values). "
        "Error bars indicate 95% Wilson score confidence intervals. At iteration 300, "
        "leakage is shown in a log-scale inset, following the same zero-value floor "
        "convention as in Figure 4."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".",
                         help="Directory containing LM/Bayesian/Encoder_predictions_long.csv")
    parser.add_argument("--out", default="Figure_Table1_barplot.png",
                         help="Output image path")
    parser.add_argument("--csv-out", default="table1_with_CI.csv",
                         help="Also write the underlying numbers (with CIs) to this CSV")
    parser.add_argument("--caption-out", default=None,
                         help="Also write the figure caption text to this file "
                              "(default: <out>, with its extension replaced by _caption.txt)")
    parser.add_argument("--no-leakage-log-inset", dest="leakage_log_inset", action="store_false",
                         help="Disable the small log-scale Leakage inset (on by default) added "
                              "to any panel where Leakage is too small to read on the linear scale")
    parser.set_defaults(leakage_log_inset=True)
    args = parser.parse_args()

    table = build_table(args.data_dir)
    cols = ["Method", "Iteration", "Recall", "Recall_lo", "Recall_hi",
            "Precision", "Precision_lo", "Precision_hi",
            "Leakage", "Leakage_lo", "Leakage_hi", "TP", "FN", "FP", "TN"]
    table = table[cols]
    print(table.round(4).to_string(index=False))
    table.to_csv(args.csv_out, index=False)
    print(f"Saved: {args.csv_out}")

    plot_two_panel_barplot(table, args.out, leakage_log_inset=args.leakage_log_inset)

    caption = build_caption(table)
    print("\n" + caption)
    caption_out = args.caption_out or os.path.splitext(args.out)[0] + "_caption.txt"
    with open(caption_out, "w") as f:
        f.write(caption + "\n")
    print(f"Saved: {caption_out}")
