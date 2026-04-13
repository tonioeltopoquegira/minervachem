"""
Post-processing tools for active learning experiment results.

This module provides utilities for analyzing and comparing Pareto fronts,
computing performance metrics, and generating statistical summaries from
experiment logs.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from ..optimization.utils import ParetoFront


def load_pareto_front(csv_path: str, target_names: List[str], 
                     generation: Optional[int] = None) -> ParetoFront:
    """Load Pareto front from CSV file.
    
    Args:
        csv_path: Path to pareto_fronts.csv file
        target_names: List of target property names
        generation: Specific generation to load; if None, loads last generation
        
    Returns:
        ParetoFront object with loaded data
    """
    df = pd.read_csv(csv_path)
    
    if generation is not None:
        filtered_df = df[df['generation'] == generation]
    else:
        last_gen = df['generation'].max()
        filtered_df = df[df['generation'] == last_gen]
    
    if filtered_df.empty:
        return ParetoFront()
    
    Y = filtered_df[target_names].to_numpy()
    smiles = filtered_df['SMILES'].tolist()
    
    pf = ParetoFront()
    pf.initialize_pareto(X_list=[0.0]*len(Y), Y_list=Y, smiles_list=smiles)
    return pf


def compute_c_metrics_across_generations(
    experiments: Dict[str, pd.DataFrame],
    target_names: List[str],
    max_gen: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Compute C-metrics for algorithm dominance across generations.
    
    Args:
        experiments: Dict mapping experiment names to their DataFrames
        target_names: List of target property names
        max_gen: Maximum generation to process; if None, uses data max
        
    Returns:
        Dictionary mapping metric names to arrays of values per generation
    """
    from plot_utils import c_metric
    
    exp_names = list(experiments.keys())
    dfs = list(experiments.values())
    
    # Determine max generation
    if max_gen is None:
        max_gen = min([df['generation'].max() for df in dfs])
    
    # Initialize storage
    metrics = {f"{exp_names[i]}_{exp_names[j]}_dominates": [] 
               for i in range(len(exp_names)) for j in range(len(exp_names)) if i != j}
    
    # Compute per generation
    for gen in range(1, max_gen + 1):
        pf_list = []
        for df in dfs:
            filtered_df = df[df['generation'] == gen]
            if filtered_df.empty:
                pf_list.append(ParetoFront())
                continue
            
            Y = filtered_df[target_names].to_numpy()
            smiles = filtered_df['SMILES'].tolist()
            pf = ParetoFront()
            pf.initialize_pareto(X_list=[0.0]*len(Y), Y_list=Y, smiles_list=smiles)
            pf_list.append(pf)
        
        # Compute C-metrics
        for i, pf_i in enumerate(pf_list):
            for j, pf_j in enumerate(pf_list):
                if i == j:
                    continue
                Y_i = pf_i.get_objectives()
                Y_j = pf_j.get_objectives()
                if Y_i.shape[0] > 0 and Y_j.shape[0] > 0:
                    c_val = c_metric(Y_i, Y_j)
                else:
                    c_val = 0.0
                metrics[f"{exp_names[i]}_{exp_names[j]}_dominates"].append(c_val)
    
    return metrics


def compute_improvement_per_generation(
    df: pd.DataFrame,
    target_names: List[str],
    reference_gen: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute improvement in targets across generations.
    
    Args:
        df: DataFrame with experiment results (must have 'generation' and target columns)
        target_names: List of target property names
        reference_gen: Generation to use as reference (default=1)
        
    Returns:
        Tuple of (improvement_values, generations) where improvement_values shape
        is (n_targets, n_generations)
    """
    # Get reference statistics
    ref_df = df[df['generation'] == reference_gen]
    Y_ref = ref_df[target_names].to_numpy()
    stds = np.std(Y_ref, axis=0)
    best_ref = np.min(Y_ref, axis=0)
    
    # Compute improvement per generation
    max_gen = df['generation'].max()
    arr_improv = np.zeros((len(target_names), max_gen))
    
    for gen in range(1, max_gen + 1):
        gen_df = df[df['generation'] == gen]
        if gen_df.empty:
            continue
        Y_gen = gen_df[target_names].to_numpy()
        best_gen = np.min(Y_gen, axis=0)
        # Improvement in units of standard deviations
        improvement = (best_ref - best_gen) / (stds + 1e-10)
        arr_improv[:, gen - 1] = improvement
    
    return arr_improv, np.arange(1, max_gen + 1)


def pareto_size_stats(
    experiments_dict: Dict[str, pd.DataFrame],
    target_names: List[str]
) -> pd.DataFrame:
    """Compute Pareto front sizes and statistics per generation.
    
    Args:
        experiments_dict: Dict mapping experiment names to DataFrames
        target_names: List of target property names
        
    Returns:
        DataFrame with columns: generation, exp_name, pareto_size, ...
    """
    results = []
    
    for exp_name, df in experiments_dict.items():
        max_gen = df['generation'].max()
        
        for gen in range(1, max_gen + 1):
            gen_df = df[df['generation'] == gen]
            if gen_df.empty:
                continue
            
            Y = gen_df[target_names].to_numpy()
            smiles = gen_df['SMILES'].tolist()
            
            pf = ParetoFront()
            pf.initialize_pareto(X_list=[0.0]*len(Y), Y_list=Y, smiles_list=smiles)
            
            results.append({
                'generation': gen,
                'experiment': exp_name,
                'pareto_size': len(pf.front),
                'total_samples': len(gen_df)
            })
    
    return pd.DataFrame(results)


def compute_model_calibration_metrics(
    mse_csv: str,
    target_names: List[str],
    block_size: int = 5
) -> pd.DataFrame:
    """Compute weighted calibration metrics from model performance CSV.
    
    Args:
        mse_csv: Path to mse_models.csv file
        target_names: List of target property names
        block_size: Size of block for averaging (default=5)
        
    Returns:
        DataFrame with weighted calibration values per block
    """
    df = pd.read_csv(mse_csv)
    num_blocks = len(df) // block_size
    weighted_calibs = []
    
    for i in range(num_blocks):
        block = df.iloc[i * block_size : (i + 1) * block_size]
        calibs = block[[f"{t}_NSE" for t in target_names]].values  # (block_size, n_targets)
        
        # Compute weights based on calibration error
        calib_errors = np.abs(np.log(calibs + 1e-6))
        scores = 1.0 / (calib_errors + 1e-3)
        weights = scores / np.sum(scores, axis=0, keepdims=True)
        
        # Weighted average
        weighted_avg = np.sum(calibs * weights, axis=0)
        weighted_calibs.append(weighted_avg)
    
    return pd.DataFrame(weighted_calibs, columns=target_names)


def ensemble_improvement_over_gens(
    experiments_dict: Dict[str, pd.DataFrame],
    target_names: List[str],
    reference_gen: int = 1,
    compute_std: bool = True
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Compute improvement curves with uncertainty for ensemble of runs.
    
    Args:
        experiments_dict: Dict with structure {exp_name: {seed: dataframe}}
        target_names: List of target property names
        reference_gen: Generation to use as reference (default=1)
        compute_std: Whether to compute standard deviations (default=True)
        
    Returns:
        Dict mapping exp_name to (mean_improvement, gens, std_improvement)
    """
    results = {}
    
    for exp_name, seeds_dict in experiments_dict.items():
        all_improvements = []
        
        for seed, df in seeds_dict.items():
            improv, gens = compute_improvement_per_generation(df, target_names, reference_gen)
            all_improvements.append(improv)
        
        # Stack: (n_seeds, n_targets, n_gens)
        all_improvements = np.array(all_improvements)
        
        # Compute statistics
        mean_improv = np.mean(all_improvements, axis=0)
        
        if compute_std:
            std_improv = np.std(all_improvements, axis=0)
        else:
            std_improv = None
        
        results[exp_name] = (mean_improv, gens, std_improv)
    
    return results


def compute_nse_divergence(
    mse_csv: str,
    target_names: List[str]
) -> Dict[str, float]:
    """Compute distance of calibration from ideal (NSE=1).
    
    Args:
        mse_csv: Path to mse_models.csv file
        target_names: List of target property names
        
    Returns:
        Dictionary mapping target name to mean absolute divergence from 1
    """
    df = pd.read_csv(mse_csv)
    divergence = {}
    
    for target in target_names:
        col = f"{target}_NSE"
        if col in df.columns:
            divergence[target] = np.mean(np.abs(1 - df[col]))
        else:
            divergence[target] = np.nan
    
    return divergence


def filter_experiments_by_generation_range(
    experiments_dict: Dict[str, pd.DataFrame],
    min_gen: int = 1,
    max_gen: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """Filter experiments to a specific generation range.
    
    Args:
        experiments_dict: Dict mapping experiment names to DataFrames
        min_gen: Minimum generation (inclusive, default=1)
        max_gen: Maximum generation (inclusive, default=None for max available)
        
    Returns:
        Filtered dictionary with same structure
    """
    filtered = {}
    
    for exp_name, df in experiments_dict.items():
        max_available = df['generation'].max()
        max_gen_use = max_gen if max_gen is not None else max_available
        
        filtered_df = df[(df['generation'] >= min_gen) & (df['generation'] <= max_gen_use)]
        filtered[exp_name] = filtered_df
    
    return filtered


def compute_hypervolume_trajectory(
    df: pd.DataFrame,
    target_names: List[str],
    ref_point: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute hypervolume indicator trajectory across generations.
    
    Args:
        df: DataFrame with experiment results
        target_names: List of target property names
        ref_point: Reference point for HV computation; if None, computed as worst objectives
        
    Returns:
        Tuple of (hypervolume_values, generations)
    """
    from pymoo.indicators.hv import HV
    
    max_gen = df['generation'].max()
    hv_values = []
    
    # Compute reference point if not provided
    if ref_point is None:
        all_Y = df[target_names].to_numpy()
        ref_point = np.max(all_Y, axis=0) * 1.1
    
    hv_indicator = HV(ref_point=ref_point)
    
    for gen in range(1, max_gen + 1):
        gen_df = df[df['generation'] == gen]
        if gen_df.empty:
            continue
        
        Y = gen_df[target_names].to_numpy()
        smiles = gen_df['SMILES'].tolist()
        
        pf = ParetoFront()
        pf.initialize_pareto(X_list=[0.0]*len(Y), Y_list=Y, smiles_list=smiles)
        Y_pf = pf.get_objectives()
        
        if Y_pf.shape[0] > 0:
            hv = hv_indicator(Y_pf)
            hv_values.append(hv)
        else:
            hv_values.append(0.0)
    
    return np.array(hv_values), np.arange(1, len(hv_values) + 1)
