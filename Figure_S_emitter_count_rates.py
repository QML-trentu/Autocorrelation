# -*- coding: utf-8 -*-
"""
Figure_S_emitter_count_rates.py

Detected count rates, stability of the rate and coincidence levels of the 25
experimental emitters.

Reads the original raw files Experimental_Data_Raw/exp_g2_raw_rep_*.csv:
two header rows with the single-channel rates of APD1 and APD2 (Hz) at each
of the nine integration times, a "tau,iter_1,...,iter_9" header row, and the
raw coincidence counts per 0.1-ns delay bin. Files without the two rate rows
(timing-corrected or normalized files) are skipped.

For every emitter it computes
  - the total detected rate (APD1 + APD2) at each integration time and at
    500 s;
  - the relative standard deviation (RSD) of the total rate over the nine
    integration times;
  - the mean number of coincidences per bin at far delays (|tau| > FAR,
    after re-centering the antibunching dip at zero delay by the fixed
    timing offset OFFSET of the two detector channels) at each integration
    time, and its ratio to the accidental level r1 * r2 * dtau * T at 500 s.

Writes (next to this script)
  Figure_S_emitter_count_rates.png        a) total rate at 500 s, b) total rate of one
                                          representative emitter at the nine integration
                                          times, c) RSD of the rate for each emitter,
                                          d) coincidences per bin versus integration time
  Emitter_count_statistics.csv            per-emitter values
  Emitter_count_statistics_summary.csv    median, quartiles, minimum, maximum and mean

Usage:  python Figure_S_emitter_count_rates.py [folder with the raw files]
        (default: Experimental_Data_Raw next to this script)
"""
import glob, os, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'Experimental_Data_Raw')
OFFSET = 18.1         # ns: fixed timing offset of the two detector channels in the raw files (median of the offsets detected by Step_00)
T = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500.])
SHOW_PANEL_LETTERS = False   # True: draw (a)-(d) on the panels; False: leave them off (add the letters by hand)
FAR = 80.0            # ns: far-delay region |tau| > FAR (after the offset correction) used for the coincidence level
REPRESENTATIVE = 18   # emitter shown in panel (b)
# Find the ORIGINAL raw files, i.e. those whose first two lines hold the single-channel rates ("APD1,..." and "APD2,..." in Hz at the nine
# integration times) followed by a "tau,iter_1,...,iter_9" header and raw coincidence counts. Preprocessed files (timing-corrected or normalized,
# without the APD rows) and the "_rawcounts" copies made for the classification pipeline do not contain the rates and are skipped.
def read_raw(f):
    with open(f, encoding='utf-8-sig') as fh: lines = fh.read().splitlines()
    if len(lines) < 4 or not lines[0].upper().startswith('APD1') or not lines[1].upper().startswith('APD2') or not lines[2].lower().startswith('tau'): return None
    r1 = np.array(lines[0].split(',')[1:], float); r2 = np.array(lines[1].split(',')[1:], float)
    X = np.array([[float(v) for v in l.split(',')] for l in lines[3:] if l.strip()])
    return r1, r2, X[:, 0], X[:, 1:]
found = sorted(glob.glob(os.path.join(DATA, '**', 'exp_g2_raw_rep_*.csv'), recursive=True))
data, skipped = {}, []
for f in found:
    rep = int(re.search(r'rep_(\d+)', os.path.basename(f)).group(1))
    try: out = read_raw(f)
    except Exception: out = None
    if out is None: skipped.append(os.path.basename(f)); continue
    data.setdefault(rep, (f, out))
print(f'{len(data)} raw files with APD rates found in {os.path.abspath(DATA)}; {len(skipped)} files without rate rows skipped; missing indices 1-25: {[k for k in range(1, 26) if k not in data]}')
if not data: raise SystemExit('No raw files with the APD1/APD2 rate rows were found. Pass the folder of the ORIGINAL raw files (not the preprocessed ones) as the first argument or set DATA.')
if len(data) != 25: print('NOTE: the statistics are defined for the 25 original raw files.')
rows, C_far = [], []
for rep in sorted(data):
    f, (r1, r2, tau, X) = data[rep]
    tau = tau - OFFSET                                       # re-centre the antibunching dip at zero delay
    dt = float(np.median(np.diff(tau)))                      # ns
    far = np.abs(tau) > FAR
    c_far = X[far].mean(axis=0); C_far.append(c_far)
    tot = r1 + r2
    acc = r1 * r2 * dt * 1e-9 * T                            # accidental coincidences per bin
    rows.append(dict(rep=rep, apd1_kcps=r1[-1] / 1e3, apd2_kcps=r2[-1] / 1e3, total_kcps=tot[-1] / 1e3,
                     rsd_pct=100 * tot.std(ddof=1) / tot.mean(), c_far_1s=c_far[0], c_far_10s=c_far[3], c_far_100s=c_far[6], c_far_500s=c_far[8],
                     far_over_accidental_500s=c_far[8] / acc[8], total_by_T=tot / 1e3))
D = pd.DataFrame(rows); C_far = np.array(C_far)

# summary statistics
cols = {'APD1 rate, 500 s (kcps)': 'apd1_kcps', 'APD2 rate, 500 s (kcps)': 'apd2_kcps', 'Total rate, 500 s (kcps)': 'total_kcps',
        'RSD of the total rate over the nine integration times (%)': 'rsd_pct',
        'Coincidences per bin, 1 s': 'c_far_1s', 'Coincidences per bin, 10 s': 'c_far_10s',
        'Coincidences per bin, 100 s': 'c_far_100s', 'Coincidences per bin, 500 s': 'c_far_500s',
        'Far-delay level / accidental level, 500 s': 'far_over_accidental_500s'}
S = pd.DataFrame({k: dict(median=D[v].median(), q25=D[v].quantile(.25), q75=D[v].quantile(.75), min=D[v].min(), max=D[v].max(), mean=D[v].mean()) for k, v in cols.items()}).T
D.drop(columns='total_by_T').to_csv(os.path.join(HERE, 'Emitter_count_statistics.csv'), index=False)
S.to_csv(os.path.join(HERE, 'Emitter_count_statistics_summary.csv'))
pd.set_option('display.width', 200); print(S.round(2).to_string())
print('emitters with total rate > 100 kcps:', int((D.total_kcps > 100).sum()), 'of', len(D), '| fewer than 1 far-delay count per bin at 1 s:', int((D.c_far_1s < 1).sum()))

plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8})
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.8), dpi=300)
blue, grey = '#3a6ea5', '#6b6b6b'
a = ax[0, 0]; a.vlines(D.rep, 0, D.total_kcps, color=blue, lw=0.8); a.plot(D.rep, D.total_kcps, 'o', ms=3.5, color=blue)
a.axhline(D.total_kcps.mean(), color=grey, ls='--', lw=0.8); a.text(0.03, 0.95, f'dashed line: mean {D.total_kcps.mean():.0f} kcps', transform=a.transAxes, ha='left', va='top', fontsize=8, color=grey)
a.set_xlabel('Emitter'); a.set_ylabel('Total detected rate (kcps)'); a.set_xlim(0, 26); a.set_ylim(0, 700); (a.text(-0.2, 1.02, '(a)', transform=a.transAxes, fontsize=12) if SHOW_PANEL_LETTERS else None)
b = ax[0, 1]; rr = (D[D.rep == REPRESENTATIVE] if (D.rep == REPRESENTATIVE).any() else D).iloc[0]; REPRESENTATIVE = int(rr.rep); b.bar(range(9), rr.total_by_T, color=blue, edgecolor='#2b2b2b', lw=0.6)
b.set_xticks(range(9)); b.set_xticklabels([f'{int(t)}' for t in T]); b.set_xlabel('Integration time (s)'); b.set_ylabel('Total detected rate (kcps)')
b.set_ylim(0, 300); b.text(0.97, 0.93, f'Emitter {REPRESENTATIVE}: RSD = {rr.rsd_pct:.1f}%', transform=b.transAxes, ha='right', va='top', fontsize=8.5); (b.text(-0.2, 1.02, '(b)', transform=b.transAxes, fontsize=12) if SHOW_PANEL_LETTERS else None)
c = ax[1, 0]; c.bar(D.rep, D.rsd_pct, color=blue, edgecolor='#2b2b2b', lw=0.5, width=0.8); c.axhline(10, color=grey, ls='--', lw=0.8)
c.text(25.5, 10.2, '10%', ha='right', va='bottom', fontsize=8, color=grey); c.set_xlabel('Emitter'); c.set_ylabel('RSD of the rate (%)'); c.set_xlim(0, 26); c.set_ylim(0, 12)
(c.text(-0.2, 1.02, '(c)', transform=c.transAxes, fontsize=12) if SHOW_PANEL_LETTERS else None)
d = ax[1, 1]
for row in C_far: d.plot(T, row, color=blue, lw=0.6, alpha=0.35)
d.plot(T, np.median(C_far, axis=0), color='#1f3b5c', lw=2, label='median'); d.axhline(1, color=grey, ls='--', lw=0.8)
d.set_xscale('log'); d.set_yscale('log'); d.set_xlabel('Integration time (s)'); d.set_ylabel('Coincidences per bin (0.1 ns)')
d.legend(frameon=False, loc='upper left', fontsize=8); (d.text(-0.2, 1.02, '(d)', transform=d.transAxes, fontsize=12) if SHOW_PANEL_LETTERS else None)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'Figure_S_emitter_count_rates.png'), bbox_inches='tight', facecolor='white'); print('saved')
