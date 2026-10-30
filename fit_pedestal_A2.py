# -*- coding: utf-8 -*-
"""
fit_pedestal_A2.py

Bunching pedestal of the experimental autocorrelation histograms.

Fits the reduced model of the paper (Eq. (1)),

    g2(tau) = 1 - A1 * exp(-|tau| / tau1) + A2,

to the corrected experimental histograms at the three longest integration
times (iter_7, iter_8, iter_9 = 100, 200 and 500 s). A2 is the relative
amplitude of the constant pedestal above the long-time (accidental) level.
Each fit is a weighted nonlinear least-squares fit (scipy curve_fit) with
Poisson weights, sigma = sqrt(max(C, 1)) / b, where C are the raw counts
and b is the baseline (raw counts divided by normalized counts, taken as the
median over |tau| >= 50 ns).

For every file it reports A2 (and its fit uncertainty), tau1, the zero-delay
value normalized to the long-time level, g2(0) = 1 - A1 + A2, and the value
normalized to the local plateau of the pedestal, g2(0) / (1 + A2). At the
longest integration time it prints, for each emitter number N, the median
and interquartile range of A2 and the median excess of the far-delay level
(|tau| >= 50 ns) over 1.

Input  (from Step_00, run from the folder that contains Experimental_Data/):
    Experimental_Data/exp_g2_raw_rep_<ID>_n<N>.csv           normalized g2(tau)
    Experimental_Data/exp_g2_raw_rep_<ID>_n<N>_rawcounts.csv raw counts
Output:
    A2_fits.csv   one row per file and integration time

Usage:  python fit_pedestal_A2.py
"""
import glob
import re

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

TAIL_NS = 50.0                       # |tau| >= TAIL_NS defines the far-delay region
ITERATIONS = ("iter_9", "iter_8", "iter_7")   # three longest integration times
P0 = [0.8, 2.0, 0.3]                 # initial guess for A1, tau1 (ns), A2
BOUNDS = ([0, 0.1, -0.5], [3, 30, 3])


def model(t, A1, tau1, A2):
    return 1 - A1 * np.exp(-np.abs(t) / tau1) + A2


def file_key(path):
    return int(re.search(r"rep_(\d+)_", path).group(1))


rows = []
files = sorted(glob.glob("Experimental_Data/exp_g2_raw_rep_*_n[0-9].csv"), key=file_key)
for gf in files:
    cf = gf.replace(".csv", "_rawcounts.csv")
    g, c = pd.read_csv(gf), pd.read_csv(cf)
    rep = int(re.search(r"rep_(\d+)_n(\d)", gf).group(1))
    n = int(re.search(r"_n(\d)", gf).group(1))
    t = g.tau.values
    tail = np.abs(t) >= TAIL_NS
    for it in ITERATIONS:
        y, C = g[it].values, c[it].values
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.nanmedian(C[tail] / y[tail])          # baseline (counts per unit g2)
        sigma = np.sqrt(np.maximum(C, 1)) / b
        out = dict(rep=rep, N=n, it=it, tail_mean=y[tail].mean(), counts_tail=C[tail].mean())
        try:
            p, cov = curve_fit(model, t, y, p0=P0, bounds=BOUNDS, sigma=sigma,
                               absolute_sigma=True, maxfev=20000)
            out.update(A1=p[0], tau1=p[1], A2=p[2], A2err=np.sqrt(cov[2, 2]),
                       g0=1 - p[0] + p[2], g0_plateau=(1 - p[0] + p[2]) / (1 + p[2]))
        except Exception:
            out.update(A1=np.nan, tau1=np.nan, A2=np.nan, A2err=np.nan,
                       g0=np.nan, g0_plateau=np.nan)
        rows.append(out)

df = pd.DataFrame(rows)
df.to_csv("A2_fits.csv", index=False)

longest = df[df.it == "iter_9"]
pd.set_option("display.width", 200)
print(longest[["rep", "N", "tail_mean", "A1", "tau1", "A2", "A2err", "g0", "g0_plateau"]]
      .round(3).to_string(index=False))
for n, grp in longest.groupby("N"):
    print("N=%d  n=%d  A2 median %.3f  IQR %.3f-%.3f  min %.3f  max %.3f | far-delay level - 1: median %.3f"
          % (n, len(grp), grp.A2.median(), grp.A2.quantile(.25), grp.A2.quantile(.75),
             grp.A2.min(), grp.A2.max(), (grp.tail_mean - 1).median()))
print("all emitters: A2 median %.3f, fraction with A2 < 0.1: %.2f, with A2 < 0.2: %.2f"
      % (longest.A2.median(), (longest.A2 < .1).mean(), (longest.A2 < .2).mean()))
