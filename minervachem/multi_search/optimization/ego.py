import math
import time
import os
from collections import Counter
import numpy as np
import csv
from scipy.stats import norm
from .utils import ParetoFront
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from ..utils_mpi import register_mpi_function, mpi_map_registered
from ..dataset.datastorage import cluster_centroids
from mpi4py import MPI
import random

'''
@register_mpi_function("evaluate_ei_chunk")
def evaluate_ei_chunk(args):
    chunk, mode, pf_array, ref_point, base_hv = args
    N_chunk, D, B = chunk.shape
    scores = np.zeros(N_chunk)

    for i in range(N_chunk):
        if mode == "pi":
            count = 0
            for b in range(B):
                sample = chunk[i, :, b]
                if not is_dominated_by_front(sample, pf_array):
                    count += 1
            scores[i] = count / B

        elif mode == "ei":
            hv = HV(ref_point=ref_point)
            hv_improvement = 0
            for b in range(B):
                sample = chunk[i, :, b]
                extended_front = np.vstack([pf_array, sample])
                hv_with_sample = hv.do(extended_front)
                hv_improvement += max(hv_with_sample - base_hv, 0)
            scores[i] = hv_improvement / B

        else:
            raise ValueError(f"Unknown mode: {mode}")

    return scores.tolist()'''


@register_mpi_function("evaluate_ei_chunk")
def evaluate_ei_chunk(args):
    point, mode, pf_array, ref_point, base_hv = args
    D, B = point.shape
    score = 0.0

    if mode == "pi":
        count = 0
        for b in range(B):
            sample = point[:, b]
            if not is_dominated_by_front(sample, pf_array):
                count += 1
        score = count / B

    elif mode == "pd":
        count = 0
        for b in range(B):
            sample = point[:, b]
            if dominates_any_in_front(sample, pf_array):
                count += 1
        score = count / B


    elif mode == "ei":
        hv = HV(ref_point=ref_point)
        hv_improvement = 0
        for b in range(B):
            sample = point[:, b]
            extended_front = np.vstack([pf_array, sample])
            hv_with_sample = hv.do(extended_front)
            hv_improvement += max(hv_with_sample - base_hv, 0)
        score = hv_improvement / B

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return score


def is_dominated_by_front(point, front):
    return np.any(np.all(front <= point, axis=1) & np.any(front < point, axis=1))

def dominates_any_in_front(point, front):
    # Returns True if `point` dominates *any* point in the front
    return np.any(np.all(point <= front, axis=1) & np.any(point < front, axis=1))




class EGO:

    def __init__(self, x_init, prop, smiles, initial_set_size=None):

        # should store pareto points in a convenient way... could be usesul to have an object for them
        # {(prop1, prop2, ...), Xfeats}

        self.pf = ParetoFront()
        self.pf.initialize_pareto(x_init, prop, smiles)

        self.ei_avg = 0 
        self.ei_sel_avg = 0 
        self.prob_success_avg =  0
        self.sp_avg = 0
        self.it = 0
        self.num_proposed = initial_set_size if initial_set_size is not None else 0
        self.num_retained = initial_set_size if initial_set_size is not None else 0
        self.select_A = -1
        self.select_B = -1
       

    def evaluate_ei(self, pf, proposed):

        boxes = pf.get_ehvi_boxes2D()
        ref = np.array(pf.reference_point)
        N, D, _ = proposed.shape

       
        mu = proposed[:, :, 0]       # shape (N, D)
        sigma = np.sqrt(proposed[:, :, 1])  # shape (N, D)

        ei = np.zeros(N)

        for l, u, type in boxes:

            l = np.array(l)
            u = np.array(u)

            a = (l - mu) / sigma     # shape (N, D)
            b = (u - mu) / sigma     # shape (N, D)

            Phi_a = norm.cdf(a)
            Phi_b = norm.cdf(b)
            phi_a = norm.pdf(a)
            phi_b = norm.pdf(b)

            # (ref - mu) * (Phi(b) - Phi(a)) + sigma * (phi(a) - phi(b))
            ei_terms = (ref - mu) * (Phi_b - Phi_a) + sigma * (phi_a - phi_b)  # shape (N, D)

            box_ei = np.prod(ei_terms, axis=1)  # shape (N,)
            ei += box_ei

        return ei
    

    '''def evaluate_eiMC(self, proposed, mode="pi"):
        N, D, B = proposed.shape
        pf_array = np.array([pt.Y for pt in self.pf.points])

        if mode == "ei":
            if self.pf.reference_point is not None:
                ref_point = np.array(self.pf.reference_point)
            else:
                ref_point = np.max(pf_array, axis=0) + self.pf.margin
            base_hv = HV(ref_point=ref_point)(pf_array)
        else:
            ref_point = None
            base_hv = None

        # Split into batches
        n_batches = min(N, MPI.COMM_WORLD.Get_size() - 1 or 1)
        batches = np.array_split(proposed, n_batches)
        args_list = [(batch, mode, pf_array, ref_point, base_hv) for batch in batches]

        results = mpi_map_registered("evaluate_ei_chunk", args_list)
        all_scores = np.concatenate(results)

        self.ei_avg = all_scores.mean()
        return all_scores'''
    
    def evaluate_eiMC(self, proposed, mode="pi", verbose=False):
        t = time.time()
        N, D, B = proposed.shape
        pf_array = np.array([pt.Y for pt in self.pf.points])

        if mode == "ei":
            ref_point = (
                np.array(self.pf.reference_point)
                if self.pf.reference_point is not None
                else np.max(pf_array, axis=0) + self.pf.margin
            )
            base_hv = HV(ref_point=ref_point)(pf_array)
        else:
            ref_point = None
            base_hv = None

        # Prepare task list: one point per task
        inputs = [(proposed[i], mode, pf_array, ref_point, base_hv) for i in range(N)]

        # Dispatch all jobs to available workers
        results = mpi_map_registered("evaluate_ei_chunk", inputs)

        all_scores = np.array(results)
        self.ei_avg = all_scores.mean()
        if verbose:
            print(f"[ACQUISITION] Evaluated mcmc with {mode} in {time.time() - t}s", flush=True)
        return all_scores

    
    def select(self, ei, molecules, cluster, retain=0.3, random_sample=False,
           save_path=None, success_prob=None, beta=0.95, verbose=False,
           X=None, offline=False):
        


        # helper to normalize molecules input -> list of dicts with 'new_smiles' key
        def _ensure_molecule_dicts(items):
            if items is None:
                return []
            # convert numpy arrays/pandas series to list safely
            try:
                import pandas as pd
            except Exception:
                pd = None
            if isinstance(items, (np.ndarray,)) or (pd is not None and isinstance(items, pd.Series)):
                items = list(items)
            if not isinstance(items, (list, tuple)):
                items = [items]
            out = []
            for it in items:
                if isinstance(it, dict):
                    # normalize: ensure 'new_smiles' exists if only 'smiles' present
                    d = dict(it)
                    if 'new_smiles' not in d and 'smiles' in d:
                        d['new_smiles'] = d['smiles']
                    out.append(d)
                else:
                    # treat it as SMILES string or other scalar -> wrap
                    out.append({'new_smiles': str(it)})
            return out

        # control printing
        do_log = (not offline) and verbose

        # normalize molecules to list of dicts so code below can call m.get(...)
        molecules_dicts = _ensure_molecule_dicts(molecules)

        # If random_sample: simple path
        if random_sample:
            n_total = len(molecules_dicts)
            n_select = max(1, int(retain))
            self.num_proposed = len(molecules_dicts)
            self.num_retained = n_select
            ind = random.sample(range(n_total), n_select)
            selected_smiles = [molecules_dicts[i] for i in ind]
            if do_log:
                print(f'[SELECTING] Randomly chosen {len(selected_smiles)}', flush=True)
            return selected_smiles, ind

        # ensure ei is numpy array for robust indexing
        ei = np.asarray(ei)
        num_retain = max(1, int(cluster*retain))
        self.num_proposed = len(molecules_dicts)
        self.num_retained = num_retain

        # compute weighted ei if needed
        if beta is not None:
            num_retain_a = max(int(beta * num_retain), 1)
            num_retain_b = max(num_retain - num_retain_a, 1)

            if success_prob is not None:
                weighted_ei = compute_weighted_ei(ei, success_prob, getattr(self, "it", 0), total_train_its=100)
            else:
                weighted_ei = ei

            # Partition indices by molecule group. If group key missing -> not in A.
            group_A_idx = [i for i, m in enumerate(molecules_dicts) if m.get("group") == "A"]
            group_BC_idx = [i for i in range(len(molecules_dicts)) if i not in group_A_idx]

            if do_log:
                print(f"[SELECTING] Group A: {num_retain_a} of {len(group_A_idx)}, Group B/C: {num_retain_b} of {len(group_BC_idx)}")

            # Select from group A (pure EI)
            group_A_ei = [(i, ei[i]) for i in group_A_idx]
            group_A_ei_sorted = sorted(group_A_ei, key=lambda x: -x[1])
            selected_A = [i for i, _ in group_A_ei_sorted[:num_retain_a]]
            self.select_A = len(selected_A)

            # Select from groups B+C (weighted EI)
            group_BC_weighted_ei = [(i, weighted_ei[i]) for i in group_BC_idx]
            group_BC_sorted = sorted(group_BC_weighted_ei, key=lambda x: -x[1])
            selected_B = [i for i, _ in group_BC_sorted if i not in selected_A][:num_retain_b]
            self.select_B = len(selected_B)

            # Combine selections
            final_idx = selected_A + selected_B
            selected = [molecules_dicts[i] for i in final_idx]
            selected_ei = ei[final_idx]
            ind = np.array(final_idx, dtype=int)

        else:
            # beta is None -> single group
            if success_prob is not None:
                weighted_ei = compute_weighted_ei(ei, success_prob, getattr(self, "it", 0), total_train_its=150)
            else:
                weighted_ei = ei

            ind_all = np.argsort(-weighted_ei)
            ind = ind_all[:num_retain]
            selected = [molecules_dicts[i] for i in ind]
            selected_ei = ei[ind]
            self.select_A = -1
            self.select_B = -1

        # Optional clustering on the selected set
        if cluster is not None or cluster > 1:
            if verbose:
                print(f'Clustering the {len(selected)} into {retain} centroids (X{cluster} decrease)', flush=True)
            t = time.time()

            # Make sure X and selected_ei are numpy-friendly
            X_idx = np.asarray(ind, dtype=int)
            selected, selected_ei, ind = cluster_centroids(
                n_clusters=retain,
                X=X[X_idx] if X is not None else None,
                sampled_smiles=selected,
                ei=selected_ei,
                original_indices=ind
            )
            if verbose:
                print(f'[SELECTING] Clustering done in {time.time() - t}s', flush=True)
            # cluster_centroids returns (centroids_smiles, centroids_ei, rep_indices_global)
            # ensure ind is numpy array
            ind = np.asarray(ind, dtype=int)

        # safe numeric arrays
        selected_ei = np.asarray(selected_ei)
        self.ei_sel_avg = float(selected_ei.mean()) if selected_ei.size > 0 else 0.0

        # Compute success-related stats only if provided
        if success_prob is not None:
            sp_arr = np.asarray(success_prob)
            # only compute averages on the selected indices we actually kept
            sel_idx_for_avg = np.asarray(ind[:num_retain], dtype=int)
            # guard if sel_idx_for_avg is empty
            self.prob_success_avg = float(sp_arr[sel_idx_for_avg].mean()) if sel_idx_for_avg.size > 0 else 0.0
            self.sp_avg = float(sp_arr.mean()) if sp_arr.size > 0 else 0.0
        else:
            self.prob_success_avg = None
            self.sp_avg = None

        if do_log:
            print("[SELECTING] Top 5 EI (unweighted):", selected_ei[:5], flush=True)
            if success_prob is not None:
                selected_success_prob = np.asarray(success_prob)[ind[:num_retain]]
                print("[SELECTING] Top 5 success_prob:", selected_success_prob[:5], flush=True)

        # plotting only when not offline and save_path set
        if (not offline) and (save_path is not None):
            try:
                plt.figure(figsize=(8, 5))
                plt.hist(ei, bins=30, alpha=0.6, label='All EI', edgecolor='black')
                plt.hist(selected_ei, bins=min(30, max(5, len(selected_ei))), alpha=0.7, label=f'Selected Top {retain*100:.0f}%', edgecolor='black')
                plt.axvline(selected_ei.mean() if selected_ei.size>0 else 0.0, color='red', linestyle='--', label='Mean Selected EI')
                plt.xlabel('Expected Improvement (EI)')
                plt.ylabel('Count')
                plt.title('Distribution of EI Scores')
                plt.legend()
                plt.tight_layout()
                os.makedirs(os.path.join(save_path, 'ei_dist'), exist_ok=True)
                plt.savefig(os.path.join(save_path, f'ei_dist/gen_{getattr(self, "it", 0)}.png'))
                plt.close()
            except Exception:
                if do_log:
                    print("[SELECTING] Warning: failed to save EI plot", flush=True)

        # increment internal iteration counter
        self.it = getattr(self, "it", 0) + 1

        # Return selected (dicts) and indices (first num_retain)
        return selected, ind[:num_retain]





    def print_generation_statistics(self, sampled_smiles=None, prop=None, attempted=None, save_path=None):
        headers = []
        row = []

        # Always include these if prop is given
        if prop is not None:
            n_valid = sum(p.get('success', False) for p in prop)
            n_timeout = sum(p.get('_timeout', False) for p in prop)
            n_reattempted = sum(p.get('attempted', 0) != 1 for p in prop if 'attempted' in p)
            n_success_reattempted = sum(p.get('success', False) and p.get('attempted', 0) > 1 for p in prop if 'attempted' in p)

            avg_time_success = np.mean([p['running_time'] for p in prop if p.get('success') and not p.get('_timeout', False)]) or 0
            avg_time_fail = np.mean([p['running_time'] for p in prop if not p.get('success') and not p.get('_timeout', False)]) or 0

            row = [
                self.pf.it, attempted, self.num_proposed, self.num_retained, self.select_A, self.select_B,
                n_valid, n_reattempted, n_success_reattempted, n_timeout,
                avg_time_success, avg_time_fail,
                self.ei_avg, self.sp_avg, self.ei_sel_avg, self.prob_success_avg,
                self.pf.gen_added, len(self.pf.points)
            ]
            headers = [
                "generation", "num_attempted", "num_proposed", "num_retained", "num_selected_A", "num_selected_B",
                "num_valid", "num_repeated", "num_success_repeated", "num_timeout",
                "avg_time_success", "avg_time_fail",
                "ei_avg", "sp_avg", "ei_sel_avg", "sp_sel_avg",
                "added_points", "size_pareto"
            ]
        else:
            row = [
                self.pf.it, self.num_proposed, self.num_retained,
                self.ei_avg, self.sp_avg, self.ei_sel_avg, self.prob_success_avg,
                self.pf.gen_added, len(self.pf.points)
            ]
            headers = [
                "generation", "num_proposed", "num_retained",
                "ei_avg", "sp_avg", "ei_sel_avg", "sp_sel_avg",
                "added_points", "size_pareto"
            ]

        # If sampled_smiles were generated, count groups A, B, D
        if sampled_smiles is not None:
            #print(sampled_smiles)
            group_counts = Counter(sm.get("group", "unknown") for sm in sampled_smiles)
            group_A = group_counts.get("A", 0)
            group_B = group_counts.get("B", 0)
            group_D = group_counts.get("C", 0)

            row.extend([group_A, group_B, group_D])
            headers.extend(["num_group_A", "num_group_B", "num_group_C"])

        # Create output directory and path
        os.makedirs(save_path, exist_ok=True)
        csv_path = os.path.join(save_path, 'generation_stats.csv')

        # Write row with header if needed
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(headers)
            writer.writerow(row)

        # Reset stats
        self.ei_avg = 0
        self.ei_sel_avg = 0
        self.pf.gen_added = 0




def compute_weighted_ei(ei, success_prob, train_it, total_train_its, power_growth=False):
    if success_prob is None:
        return ei  # No weighting if success_prob isn't available

    # Normalize training iteration progress between 0 and 1
    progress = min(train_it / total_train_its, 1.0)
    
   
    weight = (1 - progress) + progress * success_prob

    if power_growth:
        alpha = 1 + 4 * progress  
        weight = success_prob ** alpha

    return ei * weight