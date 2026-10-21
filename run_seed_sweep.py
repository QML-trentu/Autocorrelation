"""
run_seed_sweep.py

Repeat the full k-fold cross-validation of all three classifiers (LM,
Bayesian, MLP) under several different random train/test splits, to check
that the reported results do not depend on one particular split.

The three analysis scripts are NOT modified. Each already takes everything
random from SETTINGS["random_state"] (StratifiedKFold shuffle, and for the MLP
also oversampling + network initialisation via random_state + fold). This
wrapper imports each script as a module, overrides that one setting, runs the
script's own run_kfold_cv(), applies the script's own EMA smoothing (LM and
MLP; Bayesian has none, exactly as in its __main__ block) and writes the same
<Method>_predictions_long.csv schema into one folder per seed.

Seed 42 (the submitted analysis) is in the default list on purpose: its output
must reproduce the *_predictions_long.csv files written by the three analysis
scripts themselves, which validates the sweep.

Usage: put this file in the same folder as the three analysis scripts, with the
Simulation_Results folder inside it, and just run it (Spyder "Run" works, no
arguments needed). Everything defaults to that layout. Optional overrides:
    python run_seed_sweep.py --seeds 42 101 --methods Bayesian MLP
    python run_seed_sweep.py --scripts-dir D:/scripts --sim-dir D:/data --out-dir D:/out
Edit DEFAULT_SEEDS / DEFAULT_METHODS below to change what a plain "Run" does.
Existing outputs are skipped, so an interrupted run can simply be restarted.
"""
import argparse, importlib.util, os, sys, time
import matplotlib
matplotlib.use("Agg")   # never open plot windows during a long unattended run

MODULES = {
    "LM": "LM_analysis_Simulation_data_iterspecific_smoothed.py",
    "Bayesian": "Bayesian_analysis_simulation_data_iterspecific.py",
    "MLP": "Encoder_analysis_simulation_data_smoothed.py",
}
DEFAULT_SEEDS = [42, 101, 102, 103, 104, 105, 106, 107, 108, 109]


def load_module(scripts_dir, fname, alias):
    path = os.path.join(scripts_dir, fname)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, scripts_dir)
    spec.loader.exec_module(mod)   # __main__ guard keeps the analysis from running
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--scripts-dir", default=here,
                    help="folder holding the three analysis scripts (default: this script's own folder)")
    ap.add_argument("--sim-dir", default=None,
                    help="folder with the simulated g2 files (default: <scripts-dir>/Simulation_Results)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write per-seed results (default: <scripts-dir>/seed_sweep_results)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--methods", nargs="+", default=list(MODULES), choices=list(MODULES))
    a = ap.parse_args()
    a.sim_dir = a.sim_dir or os.path.join(a.scripts_dir, "Simulation_Results")
    a.out_dir = a.out_dir or os.path.join(a.scripts_dir, "seed_sweep_results")
    print(f"scripts: {a.scripts_dir}\nsim data: {a.sim_dir}\noutput: {a.out_dir}\n"
          f"seeds: {a.seeds}  methods: {a.methods}", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    mods = {m: load_module(a.scripts_dir, MODULES[m], f"sweep_{m}") for m in a.methods}

    for seed in a.seeds:
        sdir = os.path.join(a.out_dir, f"seed_{seed}")
        os.makedirs(sdir, exist_ok=True)
        for m, mod in mods.items():
            out = os.path.join(sdir, f"{m}_predictions_long.csv")
            if os.path.exists(out):
                print(f"[skip] seed {seed} {m}: already done"); continue
            t0 = time.time()
            print(f"\n=== seed {seed} | {m} ===", flush=True)
            settings = dict(mod.SETTINGS)
            settings["random_state"] = seed
            settings["simulation_dir"] = a.sim_dir
            long_df, _ = mod.run_kfold_cv(settings)
            if m in ("LM", "MLP"):
                long_df = mod.smooth_predictions_ema(long_df, alpha=settings["ema_alpha"])
            long_df.to_csv(out + ".tmp", index=False)
            os.replace(out + ".tmp", out)      # only a finished file counts as done
            print(f"seed {seed} {m} done in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
