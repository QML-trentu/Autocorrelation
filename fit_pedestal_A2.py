"""Fit the reduced model g2(t) = 1 - A1 exp(-|t|/tau1) + A2 to the experimental histograms (Experimental_Data/exp_g2_raw_rep_*_nN.csv, with the matching *_rawcounts.csv for the Poisson weights) at the three longest integration times and report the pedestal A2 and the long-time- and plateau-normalized g2(0). Run from the folder that contains Experimental_Data/. Writes A2_fits.csv."""
import pandas as pd, numpy as np, glob, re, os
from scipy.optimize import curve_fit
def model(t, A1, tau1, A2): return 1 - A1*np.exp(-np.abs(t)/tau1) + A2
rows = []
for gf in sorted(glob.glob('Experimental_Data/exp_g2_raw_rep_*_n[0-9].csv'), key=lambda s: int(re.search(r'rep_(\d+)_', s).group(1))):
    cf = gf.replace('.csv', '_rawcounts.csv')
    g = pd.read_csv(gf); c = pd.read_csv(cf); rep = int(re.search(r'rep_(\d+)_n(\d)', gf).group(1)); n = int(re.search(r'_n(\d)', gf).group(1))
    t = g.tau.values
    for it in ('iter_9', 'iter_8', 'iter_7'):
        y = g[it].values; C = c[it].values
        with np.errstate(divide='ignore', invalid='ignore'):
            b = np.nanmedian(C[np.abs(t) >= 50] / y[np.abs(t) >= 50])
        sig = np.sqrt(np.maximum(C, 1)) / b
        tail = np.abs(t) >= 50
        out = dict(rep=rep, N=n, it=it, tail_mean=y[tail].mean(), counts_tail=C[tail].mean())
        try:
            p, cov = curve_fit(model, t, y, p0=[0.8, 2.0, 0.3], bounds=([0, 0.1, -0.5], [3, 30, 3]), sigma=sig, absolute_sigma=True, maxfev=20000)
            out.update(A1=p[0], tau1=p[1], A2=p[2], A2err=np.sqrt(cov[2, 2]), g0=1 - p[0] + p[2], g0_plateau=(1 - p[0] + p[2]) / (1 + p[2]))
        except Exception as e:
            out.update(A1=np.nan, tau1=np.nan, A2=np.nan, A2err=np.nan, g0=np.nan, g0_plateau=np.nan)
        rows.append(out)
df = pd.DataFrame(rows); df.to_csv('A2_fits.csv', index=False)
d9 = df[df.it == 'iter_9']
pd.set_option('display.width', 200)
print(d9[['rep', 'N', 'tail_mean', 'A1', 'tau1', 'A2', 'A2err', 'g0', 'g0_plateau']].round(3).to_string(index=False))
for n, grp in d9.groupby('N'):
    print('N=%d n=%d  A2 median %.3f  IQR %.3f-%.3f  min %.3f max %.3f | tail-1 median %.3f' % (n, len(grp), grp.A2.median(), grp.A2.quantile(.25), grp.A2.quantile(.75), grp.A2.min(), grp.A2.max(), (grp.tail_mean - 1).median()))
print('all: A2 median %.3f, frac A2<0.1: %.2f, frac A2<0.2: %.2f' % (d9.A2.median(), (d9.A2 < .1).mean(), (d9.A2 < .2).mean()))
