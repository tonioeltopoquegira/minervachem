import os
import time
import csv
import numpy as np
import pandas as pd
import ast
import json

from .dataset.ligands_db.featurizer import featurize, transform_featurizer


def initialize_workflow(targets, name, target_names):

    n_targets = len(targets)
    target_indices = list(range(n_targets))
    folder_path = f'figs_logs/{name}' 
    
    for t in target_names:
        folder_path += '_' + t

    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(folder_path + '/2D_plots', exist_ok=True)
    os.makedirs(folder_path + '/ei_dist', exist_ok=True)
    os.makedirs(folder_path + '/error_corr', exist_ok=True)

    return n_targets, target_indices, target_names, folder_path


def timed(name=None, enabled=True):
    def wrapper(func):
        def inner(*args, **kwargs):
            t_start = time.time()
            result = func(*args, **kwargs)
            t_elapsed = time.time() - t_start
            if enabled:
                label = name or func.__name__
                print(f"[TIMER] {label} took {t_elapsed:.3f}s", flush=True)
            return result
        return inner
    return wrapper



def mask_valid(target_indices, all_properties, init_x, init_smiles):

    # Add dataset
    valid_mask_all = np.all(
    np.array([
        ~np.isnan([p['result'][t] if p['result'][t] is not None else np.nan for p in all_properties])
        for t in target_indices
    ]),
    axis=0
    )
    X_valid = init_x[valid_mask_all]
    print(f'Fully valid: {np.sum(valid_mask_all)}/{len(valid_mask_all)}')
    Y_valid = [all_properties[i]['result'] for i, valid in enumerate(valid_mask_all) if valid]
    smiles_array = np.array(init_smiles)
    smiles_valid = smiles_array[valid_mask_all]


    return X_valid, Y_valid, smiles_valid, valid_mask_all

def write_query_run(all_properties, folder_path, g, log_state=None):
    import csv
    import os

    # Add generation to all entries
    for prop in all_properties:
        prop['generation'] = g

    # Determine file path to use
    if log_state is not None and "query_log_path" in log_state:
        file_path = log_state["query_log_path"]
    else:
        file_path = os.path.join(folder_path, "query_runs.csv")

    fieldnames = sorted(set().union(*[p.keys() for p in all_properties]))

    try:
        file_exists = os.path.isfile(file_path)

        if file_exists:
            with open(file_path, newline='') as f:
                reader = csv.DictReader(f)
                existing_fields = reader.fieldnames

                if set(fieldnames) != set(existing_fields):
                    raise ValueError("Field mismatch")

        with open(file_path, mode="a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(all_properties)

    except Exception as e:
        # Create a new fallback versioned file
        fallback_path = os.path.join(folder_path, f"query_runs_gen_{g}.csv")
        print(f"[WARNING] Schema mismatch or error: {e}")
        print(f"[INFO] Switching to fallback log file: {fallback_path}")

        with open(fallback_path, mode="w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_properties)

        # Save fallback path for future use
        if log_state is not None:
            log_state["query_log_path"] = fallback_path



def update_bb_weights(
    params_history,
    alphas=[0.05, 1.0, 1.0, 0.05],
    target=1.0,
    eta=0.5,
    epsilon=1e-3,
    clip=(0.03, 50),
    normalize=False,
    history_len=-1,
    return_sigma=False):
    
    calibration_history, alphas_history = params_history
    if len(calibration_history) < 2:

        '''alpha_samples = [
            np.random.lognormal(mean=np.log(alphas[i]), sigma=0.5, size=1)
            for i in range(len(alphas))
        ]


        # Transpose and clip to get (5, n_targets)
        alphas = np.clip(np.array(alpha_samples).T, 0.03, 100)'''
        sigma = np.full((4,), 1.0).tolist()

        if return_sigma:
            return alphas, sigma  # not enough data to adapt
        else:
            return alphas
    
    print("HERE", flush=True)
    
    if history_len==-1:
        # Mean of recent calibration history (axis 0 = per-target)
        calib_mean = np.mean(calibration_history, axis=0)
        alpha_mean = np.mean(alphas_history, axis=0)

        # Estimate "sensitivity" as stddev of calibration (proxy for dC/dα)
        log_calib_history = np.log((np.array(calibration_history) + epsilon) / target)
        calib_std = np.std(log_calib_history, axis=0)
        alphas_std = np.std(alphas_history, axis=0)

    else:
        # Mean of recent calibration history (axis 0 = per-target)
        calib_mean = np.mean(calibration_history[-history_len:], axis=0)
        alpha_mean = np.mean(alphas_history[-history_len:], axis=0)

        # Estimate "sensitivity" as stddev of calibration (proxy for dC/dα)
        log_calib_history = np.log((np.array(calibration_history) + epsilon) / target)
        calib_std = np.std(log_calib_history[-history_len:], axis=0)
        alphas_std = np.std(alphas_history[-history_len:], axis=0)



    # Potentially put exponential decay, not full equal importance average


    k_fluct = calib_std

    k_max = calib_std / alphas_std # divide by 0
    k_min = clip[0]

    k = np.clip(k_fluct, k_min, k_max)

    print(f'Sensitivity: {k}')

    # Compute adjustment per target (can also use log-scaled calibration)
    distance = np.log((calib_mean + epsilon) / target) # 0.01 has same distance as 100
    d_alpha = -distance / (k + epsilon)


    updated_alphas = alpha_mean + d_alpha
   
    updated_alphas = np.maximum(updated_alphas, clip[0])

    sigmas = np.clip((alphas_std/calib_std), 0.01, 2.0)

    

    print("New alphas", updated_alphas)

    if return_sigma:
        return updated_alphas.tolist(), sigmas

    return updated_alphas.tolist()


def load_query_run(folder_path):
    """
    Attempt to load all_properties and candidate metadata from query_runs.csv.
    Also determine the latest generation number from pareto_fronts.csv.
    
    Returns:
        all_properties: list of dicts
        candidates: list of {smiles, coordList, functionalization}
        init_n: int, starting generation
    """

    file_path = os.path.join(folder_path, "query_runs.csv")

    if not os.path.isfile(file_path):
        print(f"No query_runs.csv found in {folder_path}. Will sample and query instead.", flush=True)
        return None, None, None

    # Read CSV minimally — don't fail on extra columns
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[WARNING] Failed to load query_runs.csv: {e}", flush=True)
        return None, None, None

    # Extract only necessary fields
    required_columns = {"smiles", "coordList", "functionalization", "result", "success", "running_time", "error", "generation"}
    existing_columns = set(df.columns)

    used_columns = list(required_columns & existing_columns)
    df = df[used_columns]

    # Convert to dicts
    all_properties = df.to_dict(orient="records")

    for p in all_properties:
        if isinstance(p.get("result"), str):
            try:
                p["result"] = ast.literal_eval(p["result"])
            except Exception as e:
                print(f"Failed to parse result: {p['result']}")
                raise e

    # Reconstruct candidates
    candidates = []
    for row in all_properties:
        candidate = {
            "smiles": row.get("smiles"),
            "new_smiles": row.get("new_smiles") or row.get("smiles"),
            "coordList": eval(row["coordList"]) if isinstance(row.get("coordList"), str) else row.get("coordList", []),
            "functionalization": eval(row["functionalization"]) if isinstance(row.get("functionalization"), str) else row.get("functionalization", []),
            "arch_structs": row.get("arch_structs") or []
        }

        candidates.append(candidate)

    # Determine generation number
    pareto_path = os.path.join(folder_path, "pareto_fronts.csv")
    if not os.path.isfile(pareto_path):
        init_n = 0
        print("No Pareto front file found — starting from generation 0.", flush=True)
    else:
        pf_df = pd.read_csv(pareto_path)
        if "generation" in pf_df.columns:
            init_n = int(np.max(pf_df['generation'])) + 1
            print(f"Resuming from generation {init_n}", flush=True)
        else:
            raise ValueError(f"'generation' column not found in {pareto_path}", flush=True)

    return all_properties, candidates, init_n


def retrieve_computed_candidates(folder_path):
    
    file_path = os.path.join(folder_path, "query_runs.csv")
    
    if not os.path.isfile(file_path):
        print(f"No query_runs.csv found in {folder_path}. Returning empty set.", flush=True)
        return set()

    df = pd.read_csv(file_path)

    if not all(k in df.columns for k in ['smiles', 'coordList', 'functionalization']):
        raise ValueError("query_runs.csv must contain 'smiles', 'coordList', and 'functionalization' columns")

    computed = set()

    for _, row in df.iterrows():
        smiles = row['smiles']
        try:
            coordList = ast.literal_eval(row['coordList']) if isinstance(row['coordList'], str) else row['coordList']
            functionalization = ast.literal_eval(row['functionalization']) if isinstance(row['functionalization'], str) else row['functionalization']
        except Exception as e:
            print(f"Failed to parse a row in query_runs.csv: {e}")
            continue

        candidate_key = (smiles, tuple(coordList), normalize_functionalization(functionalization))
        computed.add(candidate_key)

    return computed


def normalize_functionalization(funct_list):
    
    if not isinstance(funct_list, list):
        return ()
    
    normalized = []
    for fg in funct_list:
        fg_name = fg.get('functional_group', None)
        inds = tuple(sorted(fg.get('smiles_inds', [])))
        normalized.append((fg_name, inds))
    
    return tuple(sorted(normalized)) 


def retrieve_time_data(folder_path, featurizer):
    
    file_path = os.path.join(folder_path, "query_runs.csv")

    df = pd.read_csv(file_path)

    candidates = []
    success_list = []

    for _, row in df.iterrows():
        smiles = row['smiles']

        # Reconstruct full candidate
        coordList = ast.literal_eval(row['coordList']) if isinstance(row['coordList'], str) else row['coordList']
        functionalization = ast.literal_eval(row['functionalization']) if isinstance(row['functionalization'], str) else row.get('functionalization', [])

        candidate = {
            "smiles": smiles,
            "coordList": coordList,
            "functionalization": functionalization
        }

        # Determine success
        success_raw = row.get('success', None)
        if success_raw is not None:
            success = int(success_raw)
        else:
            result = ast.literal_eval(row['result']) if isinstance(row['result'], str) else row['result']
            success = int(all(r is not None for r in result))

        candidates.append(candidate)
        success_list.append(success)

    # Featurize full candidates
    #X_train = featurize(candidates)
    X_train, failed_idx  = transform_featurizer(candidates, featurizer)

    success_list = [s for s, f in zip(success_list, failed_idx) if f]

    return [X_train, np.array(success_list)]


def log_candidate_success_extremes(sampled_candidates, success_probs, generation, folder_path):
    """
    Log top, bottom, and median candidates based on predicted success probabilities.

    Args:
        sampled_candidates (List[Dict]): List of dicts with 'smiles', 'coordList', 'functionalization'
        success_probs (np.ndarray): Array of predicted success probabilities (shape: [n_samples])
        generation (int): Current generation index
        folder_path (str): Where to write the CSV log
    """
    assert len(sampled_candidates) == len(success_probs), "Length mismatch between candidates and success predictions."

    # Combine data
    data = list(zip(sampled_candidates, success_probs))
    data.sort(key=lambda x: x[1])  # sort by success probability

    n = len(data)
    k = min(20, n // 3)

    bottom = data[:k]
    median = data[n//2 - k//2 : n//2 + k//2]
    top = data[-k:]

    # Flatten and annotate
    labeled_data = []
    for label, section in [('bottom', bottom), ('median', median), ('top', top)]:
        for cand, prob in section:
            labeled_data.append({
                'generation': generation,
                'section': label,
                'smiles': cand['smiles'],
                'coordList': str(cand['coordList']),
                'functionalization': str(cand.get('functionalization', [])),
                'success_prob': prob
            })

    # Save to CSV
    os.makedirs(folder_path, exist_ok=True)
    csv_path = os.path.join(folder_path, "candidate_success_extremes.csv")
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=labeled_data[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(labeled_data)




def save_calib_csv(alphas, calibs, n, target_names, folder_path):
    os.makedirs(folder_path, exist_ok=True)

    n_samples, n_targets = alphas.shape

    # Create dict to hold columns
    data = {
        'generation': [n] * n_samples,
        'sample_id': list(range(n_samples))  # optional: track sample index
    }

    # Add alpha_* and calib_* columns per target
    for i, target in enumerate(target_names):
        data[f'alpha_{target}'] = alphas[:, i]
        data[f'calib_{target}'] = calibs[:, i]

    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    csv_path = os.path.join(folder_path, f'generation_{n:03d}.csv')
    df.to_csv(csv_path, index=False)




class Alpha:
    
    def __init__(self, start_weights, history_len):
        
        self.calibration_history = []
        self.alphas_history = [start_weights]
        self.history_len = history_len
        self.sigma = np.full((4,), 1.0).tolist()
        self.current_gamma = np.ones(len(start_weights)).tolist()
        self.gamma_history = [np.ones(len(start_weights)).tolist()]

    def update_gamma(self, y_true, y_pred, y_var, mask=None, clip=(0.75, 1.5), verbose=False):
        """Estimate gamma per target: MSE / variance → scale predicted variance to improve calibration."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_var = np.array(y_var)
        n_targets = y_true.shape[1]

        gammas = []

        for k in range(n_targets):
            yt, yp, var = y_true[:, k], y_pred[:, k], y_var[:, k]
            if mask is not None:
                yt, yp, var = yt[mask], yp[mask], var[mask]
            mse = np.mean((yt - yp) ** 2)
            mean_var = np.mean(var)
            gamma_k = np.clip(mse / (mean_var + 1e-8), clip[0], clip[1])
            gammas.append(gamma_k)
            if verbose:
                print(f"[GAMMA] Target {k}: MSE = {mse:.4f}, Var = {mean_var:.4f}, γ = {gamma_k:.4f}")

        self.current_gamma = np.array(gammas)
        self.gamma_history.append(self.current_gamma.tolist())

        '''# Apply short-term rolling average (window = 3)
        history_window = 3
        window_start = max(0, len(self.gamma_history) - history_window)
        smoothed_gamma = np.mean(self.gamma_history[window_start:], axis=0)

        self.current_gamma = smoothed_gamma'''


    def update_bb_weights(self,
                target=1.0,
                eta=0.5,
                epsilon=1e-3,
                clip=(0.03, 50),
                return_sigma=False,
                verbose=False,
                n_train=None):

        calibration_history, alphas_history = self.calibration_history, self.alphas_history

        if verbose:
             print(f"[ALPHA] Calibrations {calibration_history[-1]}", flush=True)

        if len(calibration_history) < 2 or len(alphas_history)<2:

            '''noise_level = 1e-6  # adjust as needed
            last_alpha = self.alphas_history[-1]
            noisy_alpha = [a + np.abs(np.random.normal(scale=noise_level)) for a in last_alpha]'''

            alphas = [1/n_train for _ in alphas_history[0]]

            self.alphas_history.append(alphas)

            return 
        
        if self.history_len==-1:
            # Mean of recent calibration history (axis 0 = per-target)
            calib_mean = np.mean(calibration_history, axis=0)
            alpha_mean = np.mean(alphas_history, axis=0)

            # Estimate "sensitivity" as stddev of calibration (proxy for dC/dα)
            log_calib_history = np.log((np.array(calibration_history) + epsilon) / target)
            calib_std = np.std(log_calib_history, axis=0)
            alphas_std = np.std(alphas_history, axis=0)

        else:
            # Mean of recent calibration history (axis 0 = per-target)
            calib_mean = np.mean(calibration_history[-self.history_len:], axis=0)
            alpha_mean = np.mean(alphas_history[-self.history_len:], axis=0)

            # Estimate "sensitivity" as stddev of calibration (proxy for dC/dα)
            log_calib_history = np.log((np.array(calibration_history) + epsilon) / target)
            calib_std = np.std(log_calib_history[-self.history_len:], axis=0)
            alphas_std = np.std(alphas_history[-self.history_len:], axis=0)



        # Potentially put exponential decay, not full equal importance average
        k_fluct = calib_std

        k_max = calib_std / alphas_std # divide by 0
        k_min = clip[0]

        k = np.clip(k_fluct, k_min, k_max)
        if verbose:
            print(f'[ALPHA] Sensitivity: {k}', flush=True)

        # Compute adjustment per target (can also use log-scaled calibration)
        distance = np.log((calib_mean + epsilon) / target) # 0.01 has same distance as 100
        d_alpha = -distance / (k + epsilon)

        updated_alphas = alpha_mean + d_alpha
    
        updated_alphas = np.maximum(updated_alphas, clip[0])

        sigmas = np.clip((alphas_std/calib_std), 0.01, 2.0)
        if verbose:
            print(f"[ALPHA] New alphas {updated_alphas}", flush=True)

        if return_sigma:
            self.sigma = sigmas
        
        self.alphas_history.append(updated_alphas.tolist())

    
    def save_histories(self, save_path: str = "."):
        """Save alpha & calibration histories (JSON + two CSVs). Returns dict of written paths."""
        os.makedirs(save_path, exist_ok=True)

        # Convert to plain Python lists (numpy scalars -> native)
        alphas = [np.asarray(a).tolist() for a in self.alphas_history]
        calibs = [np.asarray(c).tolist() for c in self.calibration_history]

        # JSON (complete)
        json_path = os.path.join(save_path, "histories.json")
        with open(json_path, "w") as f:
            json.dump({"alphas_history": alphas, "calibration_history": calibs}, f, indent=2)

        # Helper to write padded CSV
        def _write_csv(rows, path):
            if not rows:
                # still create an empty file
                open(path, "w").close()
                return
            maxcols = max(len(r) if isinstance(r, (list, tuple)) else 1 for r in rows)
            header = [f"c{i}" for i in range(maxcols)]
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                for r in rows:
                    rlist = list(r) if isinstance(r, (list, tuple, np.ndarray)) else [r]
                    # convert nested numpy scalars
                    rlist = [np.asarray(x).item() if (isinstance(x, (np.generic, np.ndarray)) and np.asarray(x).size==1) else x for x in rlist]
                    # pad
                    rlist += [""] * (maxcols - len(rlist))
                    w.writerow(rlist)

        alphas_csv = os.path.join(save_path, "alphas_history.csv")
        calibs_csv = os.path.join(save_path, "calibration_history.csv")
        _write_csv(alphas, alphas_csv)
        _write_csv(calibs, calibs_csv)

        return {"json": json_path, "alphas_csv": alphas_csv, "calibs_csv": calibs_csv}







    