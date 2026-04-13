#!/usr/bin/env python3
"""
Offline model-performance replay (MPI-compatible):
- Build cumulative train sets from query_runs.csv
- Train MultiMlearner (base for all tasks, meta for subset)
- Evaluate RMSE + NSE across train/test generations
- Split test molecules into seen-only vs unseen-bits subsets
- Save frmse_all_targets_cumulative.csv (long format)
"""

import os
import multiprocessing
from mpi4py import MPI
import ast
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from rdkit import Chem

# project imports
from .surrogates.mlearner_wrapper import MultiMlearner
from .dataset.datastorage import Dataset
from minervachem.fingerprinters import GraphletFingerprinter
from minervachem.transformers import FingerprintFeaturizer
from .utils_mpi import mpi_is_master, mpi_worker_loop, stop_all_workers
from .utils import Alpha

def compute_alpha_floor(n_train, gen, gen_max_decay=10, min_fraction=0.1):
        """
        Compute a smoothly decaying minimum alpha bound.

        Starts at 1/n and decays to 1/(min_fraction * n) over `gen_max_decay` generations.
        """
        initial = 1.0 / n_train
        final = 1.0 / (n_train / min_fraction)  # e.g., min_fraction = 0.1 → final = 1 / (10n)
        
        decay_frac = min(gen / gen_max_decay, 1.0)  # 0 → 1
        alpha_min = (1 - decay_frac) * initial + decay_frac * final
        return alpha_min

# -----------------------------
# Helpers
# -----------------------------
def parse_result_string_safe(r):
    if r is None:
        return None
    if isinstance(r, (list, tuple)):
        return list(r)
    if isinstance(r, str):
        try:
            vals = ast.literal_eval(r)
            if isinstance(vals, (list, tuple)):
                return list(vals)
        except Exception:
            return None
    return None


def pad_or_truncate_result(result_list, full_len):
    if result_list is None:
        return [None] * full_len
    out = list(result_list[:full_len])
    if len(out) < full_len:
        out.extend([None] * (full_len - len(out)))
    return out


def reconstruct_generations(df):
    """Recover generation indices: gen0=500, subsequent chunks=399, remainder last gen."""
    n = len(df)
    gens, count, gen = [], 0, 0
    while count < n:
        if gen == 0:
            chunk = min(500, n - count)
        else:
            chunk = min(299, n - count)
        gens.extend([gen] * chunk)
        count += chunk
        gen += 1
    return gens


# -----------------------------
# Offline evaluator
# -----------------------------
def offline_rmse_matrices_from_query_runs(
    experiment_folder: str,
    train_names: list,
    obj_target_indices: list = [0, 7, 8, 9],
    featurizer_max_len: int = 5,
    n_bootstrap: int = 100,
    seed: int = 0,
    use_alpha_update: bool = True
 ):

    target_names = [train_names[i] for i in obj_target_indices]
    csv_path = os.path.join(experiment_folder, "query_runs.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find {csv_path}")
    
    if mpi_is_master:
        # Allow rank 0 to use all CPUs on the node
        os.system("taskset -p 0xffffffffffffffff %d" % os.getpid())
        print(f"[Rank 0] Using {multiprocessing.cpu_count()} CPUs")
        n_jobs = multiprocessing.cpu_count()
    else:
        n_jobs = 1
    df = pd.read_csv(csv_path)
    df["generation"] = reconstruct_generations(df)
    df["parsed_results"] = df["result"].apply(parse_result_string_safe)
    df["parsed_results"] = df["parsed_results"].apply(lambda x: pad_or_truncate_result(x, len(train_names)))

    # convert each row to dict expected by Dataset.prepare_batch
    df["mol_dict"] = df.apply(lambda r: {
        "smiles": r["smiles"],
        "new_smiles": r.get("new_smiles", r["smiles"]),
        "result": r["parsed_results"]
    }, axis=1)

    # Create featurizer ONCE and assign to Dataset
    fingerprinter = GraphletFingerprinter(max_len=featurizer_max_len)
    featurizer = FingerprintFeaturizer(fingerprinter=fingerprinter, verbose=0, n_jobs=n_jobs)
    dataset = Dataset(seed=seed, standardize=False, featurizer=featurizer)

    generations = sorted(df["generation"].unique())
    n_targets = len(train_names)
    results_long = []

    out_csv = os.path.join(experiment_folder, "frmse_all_targets_cumulative.csv")

    # --- NEW: resume mechanism ---
    if os.path.exists(out_csv):
        df_prev = pd.read_csv(out_csv)
        completed_gens = set(df_prev["train_generation"].unique())
        results_long = df_prev.to_dict("records")
        print(f"[RESUME] Found previous results for generations {sorted(completed_gens)}")
    else:
        completed_gens = set()

    multi = MultiMlearner(
            n_targets=n_targets,
            n_bootstrap_samples=n_bootstrap,
            seed=seed,
            target_names=train_names
        )

    multi_fixed = MultiMlearner(
            n_targets=n_targets,
            n_bootstrap_samples=n_bootstrap,
            seed=seed + 1,  # different seed to avoid identical internal RNG if desired
            target_names=train_names
        )

    # Create Alpha object (used only to update the "update" model's bb_weights)
    initial_bb = [(1.0 / 200.0) for _ in range(4)]
    alpha = Alpha(start_weights=initial_bb.copy(), history_len=5)


    print('STARTING TRAINING LOOP', flush=True)
    for train_gen in generations:
        df_train = df[df["generation"] <= train_gen]
        print(f'[GENERATION] Training with {df_train.shape[0]} molecules from generations <= {train_gen}...', flush=True)

        print('Train transformation number denominator b4 prepare', len(df_train))
        # Prepare cumulative training data
        df_train_new = df[df["generation"] == train_gen]
        train_dicts = df_train_new["mol_dict"].tolist()

        X_train, Y_train, smiles_train = dataset.prepare_batch(
            smiles=train_dicts,
            all_properties=train_dicts,
            target_indices=list(range(n_targets)),
            verbose=False
        )
        dataset.add_multi_target_points(X_train, Y_train, list(range(n_targets)), smiles_train)

        dataset.fit_featurizer(verbose=False)

        # Train meta/base models

        print('Starting training with alpha weights:', alpha.alphas_history[-1], flush=True)
        multi.train(
            obj_indices=obj_target_indices,
            tasks=dataset.get_tasks(),
            test_sets=[[None, None] for _ in range(n_targets)],
            only_base=False,
            average_only=True,
            save_path=experiment_folder,
            bb_weights = alpha.alphas_history[-1],
            log_res=True,
            verbose=False
        )
        print('Completed training.', flush=True)

        multi_fixed.train(
            obj_indices=obj_target_indices,
            tasks=dataset.get_tasks(),
            test_sets=[[None, None] for _ in range(n_targets)],
            only_base=False,
            average_only=True,
            save_path=experiment_folder,
            bb_weights = initial_bb,  
            log_res=False,
            verbose=False
        )

        # Evaluate on all test generations
        for test_gen in generations:
            df_test = df[df["generation"] == test_gen]
            if len(df_test) == 0:
                continue
            test_dicts = df_test["mol_dict"].tolist()
            # Filter out test molecules with NaN properties for all targets before featurization
            filtered_test_dicts = []
            for td in test_dicts:
                results = td["result"]
                # Filter out molecules with NaN or None for any target
                smi = td.get('new_smiles', td.get('smiles'))
                mol = Chem.MolFromSmiles(smi)
                # Filter out molecules with invalid SMILES or valence errors
                if (mol is not None) and all((v is not None) and (not np.isnan(v)) for v in [results[t] for t in obj_target_indices]):
                    # Additional valence check: skip if any atom has valence error
                    try:
                        Chem.SanitizeMol(mol)
                        filtered_test_dicts.append(td)
                    except Exception:
                        # Skip molecules with valence errors
                        continue
            
            #print('B4 transformation number denominator', len(filtered_test_dicts))
            # Transform test molecules and get X_unseen (only valid ones)
            X_test, X_unseen, smiles_test, valid_mask = dataset.transform_smiles(
                filtered_test_dicts,
                verbose=True,
                return_unseen=True
            )
            Y_test = np.vstack([
                [np.nan if v is None else v for v in td["result"]]
                for td in filtered_test_dicts
            ])
            smiles_test_str = [td.get("new_smiles", td.get("smiles")) for td in filtered_test_dicts]

            # mask_valid is now always all True, but keep for alignment
            mask_valid = valid_mask # no slicing needed

            # Fragment-level novelty detection
            if X_unseen is None:
                # If X_unseen is None, treat all molecules as "seen" (or choose your logic)
                test_mols_all_seen = np.ones(len(filtered_test_dicts), dtype=bool)
            else:
                test_mols_all_seen = np.asarray(X_unseen.sum(axis=1) == 0).flatten()

            test_mols_all_seen_valid = test_mols_all_seen[mask_valid]
            #print('Number of test molecules (denominator)', len(test_mols_all_seen_valid))
            percent_no_novelty = 100.0 * test_mols_all_seen_valid.sum() / len(test_mols_all_seen_valid)
            percent_with_novelty = 100.0 * (~test_mols_all_seen_valid).sum() / len(test_mols_all_seen_valid)

            # Predict meta and base
            model_pred, _ = multi.predict(X_test, obj_indices=obj_target_indices)
            base_pred, _ = multi.predict_base(X_test, obj_indices=obj_target_indices, verbose=False)

            # predict non-updated model
            model_pred_fixed, _ = multi_fixed.predict(X_test, obj_indices=obj_target_indices)

            Y_array_test = np.array(Y_test)
            calibrations_for_alpha = []  # Initialize with valid float values
            for i, t_idx in enumerate(obj_target_indices):
                t_name = train_names[t_idx]
                y_true = Y_array_test[:, t_idx][:X_test.shape[0]]
                y_preds_meta = model_pred[:, i, :]
                y_preds_meta_fixed = model_pred_fixed[:, i, :]
                y_mean_meta = np.mean(y_preds_meta, axis=1)
                y_mean_meta_fixed = np.mean(y_preds_meta_fixed, axis=1)
                y_preds_base = base_pred[:, i, :]
                y_mean_base = np.mean(y_preds_base, axis=1)
                mask_valid_local = mask_valid[:len(y_mean_meta)]
                y_true_valid = y_true[mask_valid_local]
                y_mean_meta_valid = y_mean_meta[mask_valid_local]
                y_mean_base_valid = y_mean_base[mask_valid_local]
                y_mean_meta_fixed_valid = y_mean_meta_fixed[mask_valid_local]
                seen_valid = test_mols_all_seen[mask_valid_local]
                unseen_valid = ~seen_valid

                # Filter out any NaN values in y_true_valid and predictions
                valid_idx = (~np.isnan(y_true_valid)) & (~np.isnan(y_mean_meta_valid))
                y_true_valid = y_true_valid[valid_idx]
                y_mean_meta_valid = y_mean_meta_valid[valid_idx]
                y_mean_base_valid = y_mean_base_valid[valid_idx]
                y_mean_meta_fixed_valid = y_mean_meta_fixed_valid[valid_idx]
                seen_valid = seen_valid[valid_idx]
                unseen_valid = ~seen_valid

                rmse_all_meta = (np.sqrt(mean_squared_error(y_true_valid, y_mean_meta_valid))
                               if y_true_valid.size > 0 else None)
                rmse_seen_meta = (np.sqrt(mean_squared_error(y_true_valid[seen_valid],
                                                           y_mean_meta_valid[seen_valid]))
                                if seen_valid.any() else None)
                rmse_unseen_meta = (np.sqrt(mean_squared_error(y_true_valid[unseen_valid],
                                                             y_mean_meta_valid[unseen_valid]))
                                  if unseen_valid.any() else None)
                rmse_all_meta_fixed = (np.sqrt(mean_squared_error(y_true_valid, y_mean_meta_fixed_valid))
                                       if y_true_valid.size > 0 else None)

                # Base RMSE calculation
                base_valid_mask = ~np.isnan(y_mean_base_valid)
                if base_valid_mask.any():
                    y_true_base_valid = y_true_valid[base_valid_mask]
                    y_mean_base_clean = y_mean_base_valid[base_valid_mask]
                    seen_base_valid = seen_valid[base_valid_mask]
                    unseen_base_valid = ~seen_base_valid
                    rmse_all_base = (np.sqrt(mean_squared_error(y_true_base_valid, y_mean_base_clean))
                                   if y_true_base_valid.size > 0 else None)
                    rmse_seen_base = (np.sqrt(mean_squared_error(y_true_base_valid[seen_base_valid],
                                                               y_mean_base_clean[seen_base_valid]))
                                    if seen_base_valid.any() else None)
                    rmse_unseen_base = (np.sqrt(mean_squared_error(y_true_base_valid[unseen_base_valid],
                                                                 y_mean_base_clean[unseen_base_valid]))
                                      if unseen_base_valid.any() else None)
                else:
                    rmse_all_base = None
                    rmse_seen_base = None
                    rmse_unseen_base = None

                
                # Ratio of squared error to variance (meta & base; plus seen/unseen subsets for meta)
                if y_true_valid.size > 1:
                    # restrict prediction arrays to valid_idx (already applied to y_true_valid above)
                    y_preds_meta_valid = y_preds_meta[valid_idx, :]
                    y_preds_base_valid = y_preds_base[valid_idx, :]
                    y_preds_meta_fixed_valid = y_preds_meta_fixed[valid_idx, :]

                    # Per-sample ensemble mean/variance
                    ensemble_means_meta = np.mean(y_preds_meta_valid, axis=1)
                    ensemble_vars_meta  = np.var(y_preds_meta_valid, axis=1)

                    ensemble_means_meta_fixed = np.mean(y_preds_meta_fixed_valid, axis=1)
                    ensemble_vars_meta_fixed  = np.var(y_preds_meta_fixed_valid, axis=1)

                    # --- Per-sample calibration ratio statistics (median, Q1, Q3) ---
                    with np.errstate(divide="ignore", invalid="ignore"):
                        per_sample_ratio_meta = ((y_true_valid - ensemble_means_meta)**2) / ensemble_vars_meta
                        per_sample_ratio_meta_fixed = ((y_true_valid - ensemble_means_meta_fixed)**2) / ensemble_vars_meta_fixed

                    # Remove NaN/Inf values
                    valid_ratio_mask = np.isfinite(per_sample_ratio_meta)
                    per_sample_ratio_meta = per_sample_ratio_meta[valid_ratio_mask]
                    valid_ratio_mask_fixed = np.isfinite(per_sample_ratio_meta_fixed)
                    per_sample_ratio_meta_fixed = per_sample_ratio_meta_fixed[valid_ratio_mask_fixed]

                    if per_sample_ratio_meta.size > 0:
                        meta_ratio_mean = float(np.mean(per_sample_ratio_meta))
                        meta_ratio_median = float(np.median(per_sample_ratio_meta))
                        meta_ratio_q1 = float(np.percentile(per_sample_ratio_meta, 25))
                        meta_ratio_q3 = float(np.percentile(per_sample_ratio_meta, 75))

                    else:
                        meta_ratio_mean = np.nan
                        meta_ratio_median = np.nan
                        meta_ratio_q1 = np.nan
                        meta_ratio_q3 = np.nan
                    
                    if per_sample_ratio_meta_fixed.size > 0:
                        meta_ratio_mean_fixed = float(np.mean(per_sample_ratio_meta_fixed))
                        meta_ratio_median_fixed = float(np.median(per_sample_ratio_meta_fixed))

                    mse_vs_ensemble_var = (
                        np.mean((y_true_valid - ensemble_means_meta) ** 2) / np.mean(ensemble_vars_meta)
                        if np.mean(ensemble_vars_meta) > 0 else np.nan
                    )

                    # Seen / unseen calibration (meta)
                    if seen_valid.sum() > 1 and np.mean(ensemble_vars_meta[seen_valid]) > 0:
                        
                        with np.errstate(divide="ignore", invalid="ignore"):
                            per_sample_ratio_meta_seen = ((y_true_valid - ensemble_means_meta)**2) / ensemble_vars_meta
                        meta_seen_ratio_mean = float(np.mean(per_sample_ratio_meta_seen))

                        meta_mse_ens_var_seen = (
                            np.mean((y_true_valid[seen_valid] - ensemble_means_meta[seen_valid]) ** 2)
                            / np.mean(ensemble_vars_meta[seen_valid])
                        )
                    else:
                        meta_mse_ens_var_seen = np.nan
                        meta_seen_ratio_mean = np.nan

                    if unseen_valid.sum() > 1 and np.mean(ensemble_vars_meta[unseen_valid]) > 0:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            per_sample_ratio_meta_unseen = ((y_true_valid - ensemble_means_meta)**2) / ensemble_vars_meta
                        
                        meta_unseen_ratio_mean = float(np.mean(per_sample_ratio_meta_unseen))
                        meta_mse_ens_var_unseen = (
                            np.mean((y_true_valid[unseen_valid] - ensemble_means_meta[unseen_valid]) ** 2)
                            / np.mean(ensemble_vars_meta[unseen_valid])
                        )
                    else:
                        meta_mse_ens_var_unseen = np.nan
                        meta_unseen_ratio_mean = np.nan

                    # Base ensemble calibration (overall + seen/unseen)
                    ensemble_means_base = np.mean(y_preds_base_valid, axis=1)
                    ensemble_vars_base  = np.var(y_preds_base_valid, axis=1)
                    base_mse_vs_ensemble_var = (
                        np.mean((y_true_valid - ensemble_means_base) ** 2) / np.mean(ensemble_vars_base)
                        if np.mean(ensemble_vars_base) > 0 else np.nan
                    )

                    if seen_valid.sum() > 1 and np.mean(ensemble_vars_base[seen_valid]) > 0:
                        base_mse_ens_var_seen = (
                            np.mean((y_true_valid[seen_valid] - ensemble_means_base[seen_valid]) ** 2)
                            / np.mean(ensemble_vars_base[seen_valid])
                        )
                    else:
                        base_mse_ens_var_seen = np.nan

                    if unseen_valid.sum() > 1 and np.mean(ensemble_vars_base[unseen_valid]) > 0:
                        base_mse_ens_var_unseen = (
                            np.mean((y_true_valid[unseen_valid] - ensemble_means_base[unseen_valid]) ** 2)
                            / np.mean(ensemble_vars_base[unseen_valid])
                        )
                    else:
                        base_mse_ens_var_unseen = np.nan
                else:
                    mse_vs_ensemble_var = np.nan
                    meta_mse_ens_var_seen = np.nan
                    meta_mse_ens_var_unseen = np.nan
                    base_mse_vs_ensemble_var = np.nan
                    base_mse_ens_var_seen = np.nan
                    base_mse_ens_var_unseen = np.nan

                # Store calibration for alpha update if train_gen == test_gen - 1 (meta overall only)
                if test_gen == train_gen + 1:
                    calibrations_for_alpha.append(mse_vs_ensemble_var)

                #print(f"[INFO] Train {train_gen} Test {test_gen} {t_name}: "
                #      f"meta_rmse={rmse_all_meta} base_rmse={rmse_all_base} "
                #      f"meta_cal_all={mse_vs_ensemble_var:.3f} meta_cal_seen={meta_mse_ens_var_seen:.3f} meta_cal_unseen={meta_mse_ens_var_unseen:.3f} "
                #      f"base_cal_all={base_mse_vs_ensemble_var:.3f} "
                #      f"{percent_no_novelty:.1f}% seen / {percent_with_novelty:.1f}% novel")

                results_long.append({
                    "train_generation": train_gen,
                    "test_generation": test_gen,
                    "target": t_name,
                    "rmse_all_meta": rmse_all_meta,
                    "rmse_seen_meta": rmse_seen_meta,
                    "rmse_unseen_meta": rmse_unseen_meta,
                    "rmse_all_meta_fixed": rmse_all_meta_fixed,
                    "rmse_all_base": rmse_all_base,
                    "rmse_seen_base": rmse_seen_base,
                    "rmse_unseen_base": rmse_unseen_base,
                    "percent_seen": percent_no_novelty,
                    "percent_unseen": percent_with_novelty,
                    "meta_ratio": mse_vs_ensemble_var,
                    "meta_ratio_mean": meta_ratio_mean,
                    "meta_ratio_fixed_mean": meta_ratio_mean_fixed,
                    "meta_ratio_fixed_median": meta_ratio_median_fixed,
                    "meta_ratio_median": meta_ratio_median,
                    "meta_ratio_q1": meta_ratio_q1,
                    "meta_ratio_q3": meta_ratio_q3,
                    "meta_ratio_seen": meta_mse_ens_var_seen,
                    "meta_ratio_seen_mean": meta_seen_ratio_mean,
                    "meta_ratio_unseen": meta_mse_ens_var_unseen,
                    "meta_ratio_unseen_mean": meta_unseen_ratio_mean,
                    "base_ratio": base_mse_vs_ensemble_var,
                    "base_ratio_seen": base_mse_ens_var_seen,
                    "base_ratio_unseen": base_mse_ens_var_unseen
                })
            # Update alpha calibration history and weights after each train_gen
            if calibrations_for_alpha and all(isinstance(x, (float, np.floating)) and x is not None for x in calibrations_for_alpha) and use_alpha_update:
                #print(calibrations_for_alpha)
                alpha.calibration_history.append(calibrations_for_alpha)
                # Optionally update weights here, e.g.:
                n_t_points = dataset.get_tasks()[0][1].shape[0]
                low_clip = compute_alpha_floor(n_t_points, train_gen, 7, 0.001)
                alpha.update_bb_weights(clip=(low_clip, 50), verbose=True, n_train=n_t_points)
            
            #print(f'[GENERATION] Completed evaluation for train_gen={train_gen}', flush=True)
        # Save incrementally after each train_gen
        if mpi_is_master():
            temp_path = out_csv + ".tmp"
            pd.DataFrame(results_long).to_csv(temp_path, index=False)
            os.replace(temp_path, out_csv)
            print(f"[CHECKPOINT] Saved progress after train_gen={train_gen}")
    # Final save
    pd.DataFrame(results_long).to_csv(out_csv, index=False)
    if mpi_is_master():
        print(f"[OFFLINE] Saved evaluation results to {out_csv}")


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    if mpi_is_master():
        experiment_folder = "final-runs/meta_3_ensemble_woupdate_sco_kcal_octanol_gsolv_eV_octanol_hl_gap_eV_octanol_dipole"
        train_names = [
            "sco_kcal",
            "water_gsolv_eV", "water_hl_gap_eV", "water_dipole",
            "acetone_gsolv_eV", "acetone_hl_gap_eV", "acetone_dipole",
            "octanol_gsolv_eV", "octanol_hl_gap_eV", "octanol_dipole",
            "hexane_gsolv_eV", "hexane_hl_gap_eV", "hexane_dipole"
        ]
        offline_rmse_matrices_from_query_runs(
            experiment_folder=experiment_folder,
            train_names=train_names,
            obj_target_indices=[0, 7, 8, 9],
            featurizer_max_len=5,
            n_bootstrap=100,
            seed=0
        )
        stop_all_workers()
    else:
        mpi_worker_loop()
