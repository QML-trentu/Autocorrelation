# -*- coding: utf-8 -*-
"""
Step_02_Generate_g2_traces_with_exp_params.py

Generates large-scale synthetic HBT g2(tau) training/test data for the
LM/Bayesian/Encoder classifier comparison, using a Gillespie Monte Carlo
simulation of a 3-level emitter (ground/excited/shelving), with the
background fraction for each simulated repetition bootstrapped from real
data calibrated by Step_00/Step_01.

Each repetition:
  1. Draws a random number of emitters (the ground-truth classification
     target) and, for each, a random antibunching lifetime and bunching
     decay time (see the TWEAKABLE GENERATION PARAMETERS section for
     which quantities are real-data-calibrated and which are arbitrary).
  2. Draws a background fraction f_bg, either bootstrapped from the real
     calibration pool for that emitter count or, if F_BG_OVERRIDE is set,
     a fixed manually-chosen value.
  3. Simulates photon arrivals chunk by chunk across `iterations` steps,
     accumulating a coincidence histogram at each step - giving a
     noisy-to-clean progression analogous to increasing real integration
     time.
  4. Thins the accumulated histograms down to a realistic absolute count
     scale (see thin_histogram_series), and saves both raw (thinned)
     counts and normalized g2(tau) curves.

Outputs go to OUTPUT_DIR, one raw and one normalized CSV per repetition,
plus a master `training_labels.csv` with the ground-truth emitter count,
injected f_bg, and a self-consistency check (re-extracted f_bg/g2(0) from
the simulation's own output) for every repetition.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
import os
import pandas as pd

# =========================================================
# g2 model for fitting
# =========================================================
# NOTE: this is a full 5-parameter model (A1, tau1, A2, tau2, offset),
# used only for the self-consistency check at the end of each repetition
# (re-extracting f_bg from the simulation's own output). It differs from
# the 3-parameter model used in Step_00/Step_01: those fit REAL data,
# where bunching decays on a microsecond timescale far outside the
# measurement window and is therefore not a resolvable parameter, while
# the SIMULATED bunching decay time here (TAU2_RANGE) is deliberately
# kept short and resolvable (see TWEAKABLE GENERATION PARAMETERS below),
# so tau2 and offset can be fit directly.
def g2_model(t, A1, tau1, A2, tau2, offset):
    t = np.abs(t)
    return offset - A1*np.exp(-t/tau1) + A2*np.exp(-t/tau2)

def fit_g2(tau, g2):
    mask = np.isfinite(g2)
    tau_fit = tau[mask]
    g2_fit = g2[mask]

    # Check if we have enough data points to fit
    if len(g2_fit) < 5: 
        return None, None

    # Percentile-based (not raw min/max) initial guesses: raw np.min/max
    # are extreme-value statistics that can be thrown off by a single
    # noisy bin at low counts, producing guesses outside this function's
    # own bounds and making curve_fit reject the fit outright. Percentiles
    # are far less sensitive to any one bin.
    plateau_estimate = np.percentile(g2_fit, 90)
    dip_estimate = np.percentile(g2_fit, 2)
    A1_guess = np.clip(plateau_estimate - dip_estimate, 0, 2)
    tau1_guess = 0.5
    A2_guess = np.clip(plateau_estimate - 1, 0, 2)
    tau2_guess = 3.0
    offset_guess = 1.0

    p0 = [A1_guess, tau1_guess, A2_guess, tau2_guess, offset_guess]
    # Tighter bounds help stability
    bounds = ([0, 0.01, 0, 0.5, 0.5], [2, 10, 2, 20, 1.5])

    try:
        popt, pcov = curve_fit(
            g2_model, tau_fit, g2_fit, p0=p0, bounds=bounds, maxfev=20000
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except Exception:
        return None, None

# =========================================================
# REAL CALIBRATION DATA
# =========================================================
CALIBRATION_FILE = "experimental_calibration.csv"
calib_df = pd.read_csv(CALIBRATION_FILE)

# Below this many independent real files for a pool, blend in N=1's data
# too rather than trust a handful of points alone - matches
# Step_01_Extract_Params_from_experimental_data.py's same threshold, so
# both scripts stay consistent.
MIN_SAMPLES_FOR_CONFIDENCE = 10


def get_calibration_pool(n_emitters, calib_df, value_column=None):
    """
    N=1 stands on its own (enough independent real acquisitions). N=2 and
    N>=3 are pooled together (too few real N=3 measurements alone), and
    N=4,5 (no real data at all) fall back to this same pool. If even the
    pooled N>=2 group is still under-sampled, N=1's data is blended in too.
    """
    if n_emitters == 1:
        return calib_df[calib_df["N"] == 1]

    pool = calib_df[calib_df["N"].isin([2, 3, 4, 5])]

    if value_column is not None and pool[value_column].dropna().shape[0] < MIN_SAMPLES_FOR_CONFIDENCE:
        pool = pd.concat([pool, calib_df[calib_df["N"] == 1]], ignore_index=True)

    return pool


def sample_f_bg(n_emitters, calib_df):
    """Bootstrap a background fraction from the real, calibrated data for
    the appropriate emitter-count pool, grounding the simulated noise
    level in an actual measurement rather than an assumed constant."""
    pool = get_calibration_pool(n_emitters, calib_df, value_column="f_bg")
    values = pool["f_bg"].dropna().values

    if len(values) == 0:
        print(f"  [warn] no f_bg calibration data for n_emitters={n_emitters}; using a flat fallback of 0.2")
        return 0.2

    f_bg = np.random.choice(values)
    return np.clip(f_bg, 0.01, 0.99)

# =========================================================
# Physics Simulation (Gillespie)
# =========================================================
def simulate_single_emitter(n_photons, R_exc, gamma_rad, k_es, k_sg):
    # State mapping: 0 = Ground, 1 = Excited, 2 = Shelving
    state = 0 
    t = 0.0
    photons = []
    
    # Calculate inverse rates at the beginning of the simulation
    tau_exc = 1.0 / R_exc
    rate_E_total = gamma_rad + k_es
    tau_E = 1.0 / rate_E_total
    p_radiative = gamma_rad / rate_E_total
    tau_S = 1.0 / k_sg

    while len(photons) < n_photons:
        if state == 0:  # The system starts in the Ground state '0'
            t += np.random.exponential(tau_exc) # Waiting time until a photon is absorbed at a rate 1/R_exc
            state = 1 # Transition to Excited state '1' from which it will decay
            
        elif state == 1:  # The system is in the Excited state '1'
            t += np.random.exponential(tau_E) # Time the electron survives in the Excited state '1'
            # Branching: Radiative vs Intersystem Crossing
            if np.random.rand() < p_radiative: # Radiative path
                photons.append(t) # Emitted photon is recorded
                state = 0 # The system returns to the Ground state '0'
            else: # Non-radiative path
                state = 2 # The system transitions to the Shelving state '2' 
                
        elif state == 2:  # The system is in the Shelving state '2'
            t += np.random.exponential(tau_S) # Waiting time until the system transition to the Ground state '0'
            state = 0 # The system is back into the Ground state '0'
             
    return np.array(photons) # The function returns 'photons,' which is a 1D array of timestamps: [t1, t2, t3, ...]
    # Intuitively, small gaps between timestamps (t_2 - t_1) represent Ground-Excited-Ground cycling. Large gaps appear when there is shelving. This creates the "bunching" effect in g2

def generate_emitter_chunk(emitter_params, photons_per_emitter, R_exc, k_es):
    all_times = []
    for tau1, tau2 in emitter_params:
        gamma_rad = 1/tau1
        k_sg = 1/tau2
        times = simulate_single_emitter(
            photons_per_emitter, R_exc, gamma_rad, k_es, k_sg
        )
        all_times.append(times)
    return np.sort(np.concatenate(all_times))

def split_detectors(times, jitter, background_rate):
    if len(times) == 0:
        return np.array([]), np.array([])
        
    # Vectorized splitting (all 0/1 events 'flipped' at the same time, for speed)
    mask = np.random.rand(len(times)) < 0.5 # Boolean array: mask=1 photon to APD-A, mask=0, photon to APD-B
    A = times[mask] + np.random.normal(0, jitter, np.sum(mask)) # To the timestamp of the photons to APD-A we add jitter  
    B = times[~mask] + np.random.normal(0, jitter, np.sum(~mask)) # To the timestamp of the photons to APD-B we add jitter

    tmax = times[-1]
    # Add background (dark counts)
    nA = np.random.poisson(background_rate * tmax) # No. of 'fake' photons to add. If the rate is constant, the number of events follows a Poisson distribution.
    nB = np.random.poisson(background_rate * tmax)

    # Defensive cap: background_rate scales with f_bg/(1-f_bg), which is
    # unbounded as f_bg approaches 1. A degenerate f_bg value this close
    # to 1 (e.g. from a miscalibrated file - see Step_01's g2_0>1
    # exclusion) would otherwise request an enormous number of background
    # clicks per chunk, making a single repetition catastrophically slow.
    # This cap doesn't change behavior for any normal f_bg value; it only
    # guards against that specific failure mode, whether an extreme f_bg
    # comes from calibration data or a manual F_BG_OVERRIDE experiment.
    MAX_BACKGROUND_CLICKS_PER_CHUNK = 50_000
    if nA > MAX_BACKGROUND_CLICKS_PER_CHUNK or nB > MAX_BACKGROUND_CLICKS_PER_CHUNK:
        print(f"  [warn] background clicks capped this chunk (requested nA={nA}, nB={nB}, "
              f"background_rate={background_rate:.2f}) - check f_bg calibration for degenerate values near 1.0")
        nA = min(nA, MAX_BACKGROUND_CLICKS_PER_CHUNK)
        nB = min(nB, MAX_BACKGROUND_CLICKS_PER_CHUNK)

    # The 'fake' clicks are sprinkled randomly across the entire timeline. They are not correlated with the emitter
    
    if nA > 0:
        A = np.concatenate([A, np.random.uniform(0, tmax, nA)])
    if nB > 0:
        B = np.concatenate([B, np.random.uniform(0, tmax, nB)])
    # The 'Real' photons are in order (t=1, 2, 5...).The 'Noise' photons are random (t=4, 0.5, 9...). So 'Sort'restores chronological order    
    return np.sort(A), np.sort(B)

# =========================================================
# Histogram Logic (Corrected)
# =========================================================
def compute_raw_histogram(detA, detB, t_max, bin_w):
    """
    Computes raw coincidence counts. No normalization here.
    """
    bins = np.arange(-t_max, t_max + bin_w, bin_w)
    
    # Optimization: If arrays are empty, return zero hist
    if len(detA) == 0 or len(detB) == 0:
        return 0.5*(bins[:-1] + bins[1:]), np.zeros(len(bins)-1)

    diffs = []
    
    # Standard linear search (Note: for very heavy loads, numba is recommended here)
    j0 = 0
    len_B = len(detB)
    for tA in detA:
        # Move j0 to the start of the window
        while j0 < len_B and detB[j0] < tA - t_max:
            j0 += 1
        
        # Scan through the window
        j = j0
        while j < len_B and detB[j] <= tA + t_max:
            diffs.append(detB[j] - tA)
            j += 1

    hist, edges = np.histogram(diffs, bins=bins)
    centers = 0.5*(edges[:-1] + edges[1:])
    return centers, hist

def normalize_final_g2(hist_counts, centers, t_max, total_experiment_time):
    """
    Normalizes the accumulated raw counts into g2.
    """
    # Simple baseline normalization:
    # Average the values at the far wings (tau > 0.8 * t_max)
    baseline_mask = np.abs(centers) > 0.8 * t_max
    
    if np.sum(baseline_mask) == 0:
        return hist_counts # fallback
        
    avg_counts_at_inf = np.mean(hist_counts[baseline_mask])
    
    if avg_counts_at_inf == 0:
        return np.zeros_like(hist_counts)
        
    return hist_counts / avg_counts_at_inf


def thin_histogram_series(hist_columns, tau_axis, t_max, target_baseline_counts):
    """
    Brings the simulated absolute count scale down to a realistic level
    (target_baseline_counts sets the target far-tail baseline, in
    counts/bin, at the final/most-accumulated iteration) - without this,
    simulated counts land many orders of magnitude higher than a real
    acquisition, making the shot noise level unrealistically low.

    Binomial thinning of a Poisson count exactly preserves a Poisson
    distribution at the reduced mean, correctly shrinking scale AND adding
    the extra shot noise a real, lower-count acquisition would have, while
    leaving every ratio (f_bg, g2(0), shape) untouched, since it's applied
    uniformly across tau. The thinning probability is computed once from
    the FINAL (most accumulated) column, then applied identically to every
    earlier iteration snapshot, so the noise-to-clean progression across
    iterations stays internally consistent.

    NOTE: this means individual raw plots will look sparse/noisy by
    design - that is matching real data's own sparsity, not a bug. Use
    the fit (fit_g2), not eyeballing a raw trace, to check a specific
    file's underlying signal.
    """
    baseline_mask = np.abs(tau_axis) > 0.8 * t_max
    final_hist = hist_columns[-1]
    raw_baseline = np.mean(final_hist[baseline_mask])

    if raw_baseline <= 0:
        return hist_columns, 1.0

    thinning_prob = min(target_baseline_counts / raw_baseline, 1.0)

    thinned_columns = []
    for hist in hist_columns:
        hist_int = np.maximum(np.round(hist), 0).astype(np.int64)
        thinned = np.random.binomial(hist_int, thinning_prob)
        thinned_columns.append(thinned.astype(np.int64))

    return thinned_columns, thinning_prob

# =========================================================
# MAIN - ML Training Data Generation
# =========================================================
repetitions = 5000   # Number of unique emitter configurations, it'll generate a correponding number of files  
max_emitters = 5 # maximum number of possible emitters in each configuration
iterations = 300  # Number of snapshots (from noisy to clean)
photons_per_iter_per_emitter = 100 #number of photons per iteration generated by each emitter

# =========================================================
# TWEAKABLE GENERATION PARAMETERS (all in one place)
# =========================================================
# f_bg (background fraction) is the only parameter below that's extracted
# from real calibration data (see sample_f_bg above). Everything else
# here is either a fixed constant or an arbitrary, not-real-data-
# calibrated range, gathered here for easy manual tweaking in one spot.

# Set to None to bootstrap f_bg from the real calibration data, per
# repetition (the default, faithful behavior). Set to a specific number
# (e.g. 0.3) to force every repetition to use that exact value instead,
# overriding the real-data bootstrap - useful for manually exploring how
# classifier performance depends on background level, independent of what
# the current calibration file shows.
F_BG_OVERRIDE = None

# Per-emitter antibunching lifetime range - arbitrary, not extracted from
# real data (real tau1 is not reliably resolvable from a +/-100ns window -
# see Step_01's MODEL note). Kept away from very narrow values: a short
# tau1 combined with the bin width below causes real dip-broadening bias,
# since the bin-averaging that occurs when histogramming a rapidly-
# decaying exponential washes out part of the true dip depth when the bin
# width is comparable to the decay time.
TAU1_RANGE = (0.5, 1.5)

# Per-emitter bunching decay timescale range - arbitrary, not extracted
# from real data (same reason as tau1).
TAU2_RANGE = (2.0, 6.0)

# Shelving rate - fixed, not extracted from real data, shared by every
# emitter (not randomized per-emitter). Larger = more bunching.
k_es = 0.02

# Base per-emitter excitation rate - fixed constant, not extracted from
# real data.
R_exc = 0.1

# Detector timing jitter (ns, Gaussian std added per detected photon) -
# fixed constant, not extracted from real data. Kept small: jitter this
# large measurably broadens/shallows the antibunching dip, since it adds
# further uncertainty on top of the bin-averaging effect described above,
# particularly for narrow tau1 draws.
jitter = 0.05

t_max = 100.05
bin_w = 0.1

# background_rate is not a fixed constant - see sample_f_bg above,
# called once per repetition and converted to an actual rate from the
# real, injected signal rate.

OUTPUT_DIR = "Simulation_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target far-tail baseline count rate (counts/bin) at the final/most-
# accumulated iteration, controlling how much post-hoc thinning is
# applied (see thin_histogram_series). Chosen so the final iteration
# supports confident N classification: for N=1 (the tightest case, lowest
# raw count rate), the natural pre-thinning baseline with the current
# iterations/photons_per_iter_per_emitter settings is comfortably above
# this target, so the thinning probability stays meaningfully below 1
# (preserving genuine shot noise) rather than being clamped to "no
# thinning" for some cases. Note that thinning is applied uniformly
# across all iterations from a single target, so early iterations are
# also less sparse than a real short integration would be; a separate,
# non-uniform thinning schedule would be needed to match real sparsity at
# every iteration simultaneously.
TARGET_BASELINE_COUNTS = 60

# Master log to keep track of every simulation
labels_log = []

for rep in range(repetitions):
    rep_id = rep + 1
    
    # 1. Randomize the number of emitters (This is our ML Target Class)
    n_emitters = np.random.randint(1, max_emitters + 1)
    
    # 2. Randomize physics for each emitter to create a diverse dataset
    emitter_params = [
        (np.random.uniform(*TAU1_RANGE), np.random.uniform(*TAU2_RANGE))
        for _ in range(n_emitters)
    ]

    # Background fraction for this repetition: F_BG_OVERRIDE (if set)
    # takes priority over the real-data bootstrap - see the TWEAKABLE
    # GENERATION PARAMETERS section above.
    if F_BG_OVERRIDE is not None:
        f_bg = F_BG_OVERRIDE
    else:
        f_bg = sample_f_bg(n_emitters, calib_df)

    print(f"\n--- Repetition {rep_id}/{repetitions} ---")
    if F_BG_OVERRIDE is not None:
        print(f"Simulating {n_emitters} emitter(s), target f_bg = {f_bg:.4f} (MANUAL OVERRIDE - not from real data)")
    else:
        print(f"Simulating {n_emitters} emitter(s), target f_bg = {f_bg:.4f}")

    # ---------- Data Storage for this Repetition ----------
    accumulated_hist = None
    tau_axis = None
    hist_columns = [] 
    g2_columns = []

    # 3. The Iteration Loop (Generating the noise-to-clean evolution)
    for it in range(iterations):
        # Generate data chunk
        photons = generate_emitter_chunk(
            emitter_params, photons_per_iter_per_emitter, R_exc, k_es
        )

        # Convert the target f_bg into an actual background click rate,
        # using THIS chunk's real signal rate (so it automatically scales
        # with n_emitters rather than assuming a fixed rate constant):
        # f_bg = bg/(bg+signal)  =>  bg = f_bg/(1-f_bg) * signal
        #
        # signal_rate_actual below is the TOTAL rate (both detectors
        # combined, before the 50/50 detector split in split_detectors),
        # but background_rate is applied to EACH detector separately
        # after that split. Halving signal_rate_actual here correctly
        # matches the per-detector basis background_rate is actually used
        # on - using the un-halved total rate would calibrate background
        # against twice the actual per-detector signal rate, injecting
        # roughly double the intended background fraction.
        if len(photons) > 1:
            duration = photons[-1] - photons[0]
            signal_rate_actual = len(photons) / duration if duration > 0 else n_emitters * R_exc
        else:
            signal_rate_actual = n_emitters * R_exc
        background_rate = (f_bg / (1 - f_bg)) * (signal_rate_actual / 2)

        # Split into detectors
        detA, detB = split_detectors(photons, jitter, background_rate)
        
        # Compute raw histogram
        tau, hist_chunk = compute_raw_histogram(detA, detB, t_max, bin_w)
        
        # Accumulate statistics
        if accumulated_hist is None:
            accumulated_hist = hist_chunk.astype(float)
            tau_axis = tau
        else:
            accumulated_hist += hist_chunk

        # Store snapshots (for ML training on different noise levels)
        hist_columns.append(accumulated_hist.copy())
        
        # Normalize the cumulative data
        curr_g2 = normalize_final_g2(accumulated_hist, tau_axis, t_max, 0)
        g2_columns.append(curr_g2)

        if (it + 1) % 250 == 0:
            print(f"  Iteration {it+1}/{iterations}...")

    # =========================================================
    # 4. Save CSVs for this repetition
    # =========================================================
    # Bring the simulated absolute count scale down to match real data
    # (see thin_histogram_series) - the raw simulated scale otherwise
    # lands orders of magnitude above anything a real acquisition shows.
    hist_columns, thinning_prob = thin_histogram_series(
        hist_columns, tau_axis, t_max, TARGET_BASELINE_COUNTS
    )
    g2_columns = [normalize_final_g2(h.astype(float), tau_axis, t_max, 0) for h in hist_columns]
    print(f"  Thinning applied: keep-probability={thinning_prob:.3e} "
          f"(target baseline = {TARGET_BASELINE_COUNTS} counts/bin)")

    # Emitter count is encoded in the filename (n{n_emitters}) for easy identification
    csv_raw = os.path.join(OUTPUT_DIR, f"g2_raw_rep_{rep_id}_n{n_emitters}.csv")
    csv_norm = os.path.join(OUTPUT_DIR, f"g2_norm_rep_{rep_id}_n{n_emitters}.csv")
    
    header = ["tau"] + [f"iter_{i+1}" for i in range(iterations)]

    # Save the raw counts data
    with open(csv_raw, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(np.column_stack([tau_axis] + hist_columns))

    # Save the normalized g2 data (This is the primary ML input)
    with open(csv_norm, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(np.column_stack([tau_axis] + g2_columns))

    # Self-consistency check: re-extract f_bg/g2(0) from our own final
    # simulated curve using the same fit already defined above, and
    # compare to what was actually injected.
    popt, _ = fit_g2(tau_axis, g2_columns[-1])
    if popt is not None:
        A1, tau1_fit, A2, tau2_fit, offset = popt
        g2_0_recovered = offset - A1 + A2
    else:
        g2_0_recovered = np.min(g2_columns[-1])
    val = max(0, n_emitters * (1 - g2_0_recovered))
    rho_recovered = min(max(np.sqrt(val), 0), 1)
    f_bg_recovered = 1 - rho_recovered
    print(f"  Validation: injected f_bg={f_bg:.4f}  recovered f_bg={f_bg_recovered:.4f}  "
          f"(recovered g2(0)={g2_0_recovered:.4f})")

    # Add to the master label log
    labels_log.append([rep_id, n_emitters, f_bg, F_BG_OVERRIDE is not None,
                        thinning_prob, f_bg_recovered, g2_0_recovered])
    print(f"Finished. Saved: {csv_norm}")

# Save the final master labels file
labels_path = os.path.join(OUTPUT_DIR, "training_labels.csv")
with open(labels_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["rep_id", "n_emitters", "f_bg_injected", "f_bg_was_override", "thinning_prob",
                      "f_bg_recovered", "g2_0_recovered"])
    writer.writerows(labels_log)

print(f"\nAll simulations complete. Master labels saved to {labels_path}")