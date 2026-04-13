"""
Visualization tools for active learning experiment analysis.

Provides functions for plotting Pareto fronts, algorithm comparisons,
calibration curves, and other metrics across experiments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import PathPatch, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patches as mpatches
from typing import List, Dict, Optional, Tuple
import imageio
import os

from ..optimization.utils import ParetoFront
from .plot_utils import c_metric


def plot_dominance_metrics_over_generations(
    metrics_dict: Dict[str, List[float]],
    title: str = "C-Metric Dominance Over Generations",
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> None:
    """Plot dominance metrics across generations.
    
    Args:
        metrics_dict: Dictionary mapping metric names to lists of values per generation
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: Optional path to save figure
    """
    plt.figure(figsize=figsize)
    
    for metric_name, values in metrics_dict.items():
        generations = list(range(1, len(values) + 1))
        plt.plot(generations, values, label=metric_name, marker='o', alpha=0.7)
    
    plt.xlabel('Generation')
    plt.ylabel('C-Metric Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_improvement_trajectories(
    improvements_dict: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    target_names: List[str],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """Plot improvement curves with uncertainty bands for multiple experiments.
    
    Args:
        improvements_dict: Dict from ensemble_improvement_over_gens (exp_name -> (mean, gens, std))
        target_names: List of target property names
        figsize: Figure size
        save_path: Optional path to save figure
    """
    n_targets = len(target_names)
    fig, axs = plt.subplots(n_targets, 1, figsize=figsize, sharex=True)
    
    if n_targets == 1:
        axs = [axs]
    
    colors = plt.cm.tab10.colors
    
    for ax, target_name in zip(axs, target_names):
        for color_idx, (exp_name, (mean_improv, gens, std_improv)) in enumerate(improvements_dict.items()):
            target_idx = target_names.index(target_name)
            
            ax.plot(gens, mean_improv[target_idx], 
                   label=exp_name, 
                   color=colors[color_idx % len(colors)],
                   linewidth=2)
            
            if std_improv is not None:
                ax.fill_between(gens, 
                               mean_improv[target_idx] - std_improv[target_idx],
                               mean_improv[target_idx] + std_improv[target_idx],
                               alpha=0.2,
                               color=colors[color_idx % len(colors)])
        
        ax.set_ylabel(f'{target_name}\nImprovement (std units)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axs[-1].set_xlabel('Generation')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_model_calibration_comparison(
    calibration_dicts: Dict[str, pd.DataFrame],
    target_names: List[str],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> None:
    """Plot model calibration (NSE) across algorithms.
    
    Args:
        calibration_dicts: Dict mapping algorithm names to calibration DataFrames
        target_names: List of target property names
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    colors = plt.cm.tab10.colors
    
    # Full range plot
    for color_idx, (algo_name, df) in enumerate(calibration_dicts.items()):
        for target in target_names:
            if target in df.columns:
                ax1.plot(range(len(df)), df[target], 
                        label=f"{algo_name}-{target}" if color_idx == 0 else "",
                        color=colors[color_idx % len(colors)],
                        alpha=0.7)
    
    ax1.set_ylabel('NSE')
    ax1.set_ylim(0, 100)
    ax1.set_title('Full Range')
    ax1.grid(True, alpha=0.3)
    
    # Zoomed plot
    for color_idx, (algo_name, df) in enumerate(calibration_dicts.items()):
        for target in target_names:
            if target in df.columns:
                ax2.plot(range(len(df)), df[target],
                        label=f"{algo_name}-{target}",
                        color=colors[color_idx % len(colors)],
                        alpha=0.7)
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('NSE')
    ax2.set_ylim(0, 5)
    ax2.set_title('Zoomed View')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_hypervolume_trajectories(
    hv_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """Plot hypervolume indicator trajectories for multiple experiments.
    
    Args:
        hv_dict: Dict mapping experiment names to (hv_values, generations) tuples
        figsize: Figure size
        save_path: Optional path to save figure
    """
    plt.figure(figsize=figsize)
    
    colors = plt.cm.tab10.colors
    for color_idx, (exp_name, (hv_values, gens)) in enumerate(hv_dict.items()):
        plt.plot(gens, hv_values,
                label=exp_name,
                color=colors[color_idx % len(colors)],
                marker='o',
                linewidth=2)
    
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.title('Hypervolume Indicator Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_pareto_sizes_over_generations(
    pareto_stats_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """Plot Pareto front sizes across generations for multiple experiments.
    
    Args:
        pareto_stats_df: DataFrame from pareto_size_stats() with columns:
                        generation, experiment, pareto_size, total_samples
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    experiments = pareto_stats_df['experiment'].unique()
    colors = plt.cm.tab10.colors
    
    for idx, exp in enumerate(experiments):
        exp_data = pareto_stats_df[pareto_stats_df['experiment'] == exp].sort_values('generation')
        ax1.plot(exp_data['generation'], exp_data['pareto_size'],
                label=exp,
                color=colors[idx % len(colors)],
                marker='o')
        
        ax2.plot(exp_data['generation'], 
                exp_data['pareto_size'] / exp_data['total_samples'] * 100,
                label=exp,
                color=colors[idx % len(colors)],
                marker='o')
    
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Pareto Front Size')
    ax1.set_title('Absolute Pareto Front Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Pareto Size (% of Total)')
    ax2.set_title('Relative Pareto Front Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def create_calibration_animation(
    calibration_dicts: Dict[str, pd.DataFrame],
    target_names: List[str],
    output_gif: str = "calibration_animation.gif",
    output_dir: str = "animation_frames",
    hold_extra: int = 500
) -> None:
    """Create animated GIF showing calibration curves per target.
    
    Args:
        calibration_dicts: Dict mapping algorithm names to calibration DataFrames
        target_names: List of target property names to animate
        output_gif: Output GIF filename
        output_dir: Directory for intermediate PNG frames
        hold_extra: Number of extra frames to hold first/last frame
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = []
    colors = plt.cm.tab10.colors
    
    for target_idx, target in enumerate(target_names):
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for algo_idx, (algo_name, df) in enumerate(calibration_dicts.items()):
            if target in df.columns:
                ax.plot(range(len(df)), df[target],
                       label=algo_name,
                       color=colors[algo_idx % len(colors)],
                       linewidth=2)
        
        ax.axhline(y=1.0, color='purple', linestyle='--', label='Ideal Calibration', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('NSE')
        ax.set_ylim(0, 5)
        ax.set_title(f'{target} Calibration Error Over Generations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        frame_file = os.path.join(output_dir, f"{target}.png")
        fig.tight_layout()
        fig.savefig(frame_file, dpi=100)
        plt.close(fig)
        frame_paths.append(frame_file)
    
    # Create animated GIF
    frame_duration = 3.0
    with imageio.get_writer(output_gif, mode='I', duration=frame_duration) as writer:
        for idx, frame in enumerate(frame_paths):
            image = imageio.imread(frame)
            
            # Hold first frame
            for _ in range(hold_extra):
                writer.append_data(image)
            
            writer.append_data(image)
            
            # Hold last frame
            if idx == len(frame_paths) - 1:
                for _ in range(hold_extra):
                    writer.append_data(image)
    
    print(f"✅ Animation saved to {output_gif}")


def plot_2d_pareto_grid(
    pf_list: List[ParetoFront],
    exp_names: List[str],
    target_names: List[str],
    reference_pf: Optional[ParetoFront] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> None:
    """Plot 2D projections of Pareto fronts in a grid.
    
    Args:
        pf_list: List of ParetoFront objects
        exp_names: Names for each Pareto front
        target_names: Names of target properties
        reference_pf: Optional reference Pareto front (e.g., true Pareto)
        figsize: Figure size; if None, auto-computed
        save_path: Optional path to save figure
    """
    n_targets = len(target_names)
    pair_indices = [(i, j) for i in range(n_targets) for j in range(i + 1, n_targets)]
    
    ncols = min(len(pair_indices), 3)
    nrows = (len(pair_indices) + ncols - 1) // ncols
    
    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten()
    
    colors = plt.cm.tab10.colors
    
    for idx, (x_idx, y_idx) in enumerate(pair_indices):
        ax = axs[idx]
        
        # Plot experimental Pareto fronts
        for i, (pf, exp_name) in enumerate(zip(pf_list, exp_names)):
            proj = pf.get_projected_pareto((x_idx, y_idx))
            if proj and len(proj) > 0:
                proj = np.array(proj)
                ax.scatter(proj[:, 0], proj[:, 1],
                          color=colors[i % len(colors)],
                          label=exp_name,
                          alpha=0.8,
                          s=50)
        
        # Plot reference Pareto front
        if reference_pf is not None:
            ref_proj = reference_pf.get_projected_pareto((x_idx, y_idx))
            if ref_proj and len(ref_proj) > 0:
                ref_proj = np.array(ref_proj)
                ax.scatter(ref_proj[:, 0], ref_proj[:, 1],
                          color='black',
                          marker='x',
                          label='Reference',
                          s=100,
                          linewidths=2)
        
        ax.set_xlabel(target_names[x_idx])
        ax.set_ylabel(target_names[y_idx])
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='best')
    
    # Hide unused subplots
    for idx in range(len(pair_indices), len(axs)):
        axs[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_nse_divergence_bars(
    divergence_dicts: Dict[str, Dict[str, float]],
    target_names: List[str],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """Plot NSE divergence from ideal (1.0) as bar chart.
    
    Args:
        divergence_dicts: Dict mapping algorithm names to divergence dicts (target -> value)
        target_names: List of target property names
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(target_names))
    width = 0.8 / len(divergence_dicts)
    
    colors = plt.cm.tab10.colors
    
    for algo_idx, (algo_name, divergence_dict) in enumerate(divergence_dicts.items()):
        values = [divergence_dict.get(t, np.nan) for t in target_names]
        ax.bar(x + algo_idx * width, values,
              width=width,
              label=algo_name,
              color=colors[algo_idx % len(colors)],
              alpha=0.8)
    
    ax.set_xlabel('Target Property')
    ax.set_ylabel('|NSE - 1.0| (Divergence from Ideal)')
    ax.set_title('Model Calibration Error by Target')
    ax.set_xticks(x + width * (len(divergence_dicts) - 1) / 2)
    ax.set_xticklabels(target_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
