import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pymoo.indicators.hv import HV
import numpy as np
import os

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_pareto_fronts(pf_list, actual_pf, exp_names, target_names, lim, save_path=None, filename="pareto_grid.png"):
    assert len(pf_list) == len(exp_names), "Mismatch between pf_list and exp_names"

    n_targets = len(target_names)
    pair_indices = [(i, j) for i in range(n_targets) for j in range(i + 1, n_targets)]

    ncols = min(len(pair_indices), 3)
    nrows = (len(pair_indices) + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows))
    axs = axs.flatten()

    colors = plt.cm.tab10.colors
    tolerance = 1e-8
    all_handles = []
    all_labels = []

    for idx, (x_idx, y_idx) in enumerate(pair_indices):
        ax = axs[idx]
        xlim, ylim = lim[x_idx], lim[y_idx]

        # Project PFs
        projected_pfs = []
        for pf in pf_list:
            proj = pf.get_projected_pareto((x_idx, y_idx))
            projected_pfs.append(np.array(proj) if proj else np.empty((0, 2)))

        actual_proj = np.array(actual_pf.get_projected_pareto((x_idx, y_idx))) if actual_pf else np.empty((0, 2))

        pf0 = projected_pfs[0]
        pf1 = projected_pfs[1]

        coinc_0_actual = set()
        coinc_1_actual = set()
        coinc_between_0_1 = set()

        for pt in pf0:
            if np.any(np.all(np.abs(actual_proj - pt) < tolerance, axis=1)):
                coinc_0_actual.add(tuple(pt))

        for pt in pf1:
            if np.any(np.all(np.abs(actual_proj - pt) < tolerance, axis=1)):
                coinc_1_actual.add(tuple(pt))

        for pt in pf0:
            if np.any(np.all(np.abs(pf1 - pt) < tolerance, axis=1)):
                pt_tuple = tuple(pt)
                if pt_tuple not in coinc_0_actual and pt_tuple not in coinc_1_actual:
                    coinc_between_0_1.add(pt_tuple)

        # Plot experimental PFs
        for i, pf_proj in enumerate(projected_pfs):
            if pf_proj.size > 0:
                scatter = ax.scatter(pf_proj[:, 0], pf_proj[:, 1],
                                     color=colors[i % len(colors)],
                                     label=exp_names[i], alpha=0.8)

                if exp_names[i] not in all_labels:
                    all_handles.append(scatter)
                    all_labels.append(exp_names[i])

                # Stepwise lines
                ref_x = np.max(xlim)
                ref_y = np.max(ylim)
                x0, y0 = pf_proj[0]
                ax.plot([x0, x0], [ref_y, y0], color=colors[i % len(colors)], linestyle='--')
                for k in range(len(pf_proj) - 1):
                    x1, y1 = pf_proj[k]
                    x2, y2 = pf_proj[k + 1]
                    ax.plot([x1, x2], [y1, y1], color=colors[i % len(colors)], linestyle='--')
                    ax.plot([x2, x2], [y1, y2], color=colors[i % len(colors)], linestyle='--')
                last_x, last_y = pf_proj[-1]
                ax.plot([last_x, ref_x], [last_y, last_y], color=colors[i % len(colors)], linestyle='--')

        # Plot actual PF
        if actual_proj.size > 0:
            scatter = ax.scatter(actual_proj[:, 0], actual_proj[:, 1],
                                 color='black', marker='x', label='Actual Pareto')
            if "Actual Pareto" not in all_labels:
                all_handles.append(scatter)
                all_labels.append("Actual Pareto")

            ref_x = np.max(xlim)
            ref_y = np.max(ylim)
            x0, y0 = actual_proj[0]
            ax.plot([x0, x0], [ref_y, y0], 'k--')
            for k in range(len(actual_proj) - 1):
                x1, y1 = actual_proj[k]
                x2, y2 = actual_proj[k + 1]
                ax.plot([x1, x2], [y1, y1], 'k--')
                ax.plot([x2, x2], [y1, y2], 'k--')
            last_x, last_y = actual_proj[-1]
            ax.plot([last_x, ref_x], [last_y, last_y], 'k--')

        # Plot coincidences with descriptive labels
        if coinc_0_actual:
            pts = np.array(list(coinc_0_actual))
            scatter = ax.scatter(pts[:, 0], pts[:, 1], color='red', s=80, marker='o',
                                 label=f"{exp_names[0]} coincides with Actual")
            label = f"{exp_names[0]} coincides with Actual"
            if label not in all_labels:
                all_handles.append(scatter)
                all_labels.append(label)

        if coinc_1_actual:
            pts = np.array(list(coinc_1_actual))
            scatter = ax.scatter(pts[:, 0], pts[:, 1], color='red', s=80, marker='o',
                                 label=f"{exp_names[1]} coincides with Actual")
            label = f"{exp_names[1]} coincides with Actual"
            if label not in all_labels:
                all_handles.append(scatter)
                all_labels.append(label)

        if coinc_between_0_1:
            pts = np.array(list(coinc_between_0_1))
            scatter = ax.scatter(pts[:, 0], pts[:, 1], color='purple', s=80, marker='s',
                                 label=f"{exp_names[0]} coincides with {exp_names[1]}")
            label = f"{exp_names[0]} coincides with {exp_names[1]}"
            if label not in all_labels:
                all_handles.append(scatter)
                all_labels.append(label)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(target_names[x_idx])
        ax.set_ylabel(target_names[y_idx])
        ax.grid(False)
        ax.set_aspect('equal', adjustable='box')

    # Hide unused axes
    for j in range(len(pair_indices), len(axs)):
        axs[j].axis('off')

    # Unified horizontal legend at top
    fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=min(4, len(all_labels)), frameon=False, prop={'size': 14},           # increase font size
           markerscale=1.5)

    fig.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for legend
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, filename))
        plt.close(fig)
    else:
        plt.show()




def dominates(a, b):
    if np.allclose(a, b, atol=1e-3, rtol=0):  # equal → not dominated
        return False
    return np.all(a <= b) and np.any(a < b)

def c_metric(A, B):
    dominated = 0
    for b in B:
        if any(dominates(a, b) for a in A):
            dominated += 1
    return dominated / len(B)




def evaluate_pareto_quality(pf_actual, pf_reference=None, normalize=True, ref_point=None):
    
    Y_pred = np.array([p.Y for p in pf_actual.points])

    if pf_reference is not None:
        Y_true = np.array([p.Y for p in pf_reference.points])
    else:
        Y_true = Y_pred  # for standalone HV

    
    print(Y_pred.shape)
    print(Y_true.shape)

    # === Normalization
    if normalize:
        combined = np.vstack([Y_pred, Y_true])
        min_vals = np.min(combined, axis=0)
        max_vals = np.max(combined, axis=0)
        range_vals = np.maximum(max_vals - min_vals, 1e-8)
        Y_pred = (Y_pred - min_vals) / range_vals
        Y_true = (Y_true - min_vals) / range_vals

    # === Hypervolume
    if ref_point is None:
        ref_point = np.max(Y_true, axis=0) * 1.1
    hv_indicator = HV(ref_point=ref_point)
    hv = hv_indicator.do(Y_pred)

    # === GD / IGD
    gd = np.mean(cKDTree(Y_true).query(Y_pred)[0])
    igd = np.mean(cKDTree(Y_pred).query(Y_true)[0])

    # === C-metrics (only if reference given)
    c_ba=None
    if pf_reference is not None:
        c_ba = c_metric(Y_true, Y_pred)

    return {
        "hypervolume": hv,
        "generational_distance": gd,
        "inverted_generational_distance": igd,
        "c_actual_dominates_pred": c_ba
    }


def check_dominated(actual, predicted):
    dominated_flags = []
    for i, y_a in enumerate(actual):
        for y_p in predicted:
            if dominates(y_p, y_a):
                dominated_flags.append((i, y_a, y_p))
                break  # one is enough to count as dominated
    return dominated_flags

def normalize_fronts(Y_pred, Y_actual):
    combined = np.vstack([Y_pred, Y_actual])
    min_vals = np.min(combined, axis=0)
    max_vals = np.max(combined, axis=0)
    range_vals = np.maximum(max_vals - min_vals, 1e-8)

    Y_pred_norm = (Y_pred - min_vals) / range_vals
    Y_actual_norm = (Y_actual - min_vals) / range_vals
    return Y_pred_norm, Y_actual_norm


import os
import numpy as np
import matplotlib.pyplot as plt

def _collect_target_vals_from_projected(pf_list, actual_pf, n_targets):
    """
    Collects values per target index by scanning all pair projections.
    Returns dict idx -> list of values seen for that target.
    """
    vals = {i: [] for i in range(n_targets)}
    pair_indices = [(i, j) for i in range(n_targets) for j in range(i + 1, n_targets)]

    for pf in pf_list:
        for (i, j) in pair_indices:
            proj = pf.get_projected_pareto((i, j))
            if not proj:
                continue
            arr = np.array(proj)  # shape (m,2)
            if arr.size == 0:
                continue
            vals[i].extend(arr[:, 0].tolist())
            vals[j].extend(arr[:, 1].tolist())

    # include actual_pf if present
    if actual_pf is not None:
        for (i, j) in pair_indices:
            proj = actual_pf.get_projected_pareto((i, j))
            if not proj:
                continue
            arr = np.array(proj)
            if arr.size == 0:
                continue
            vals[i].extend(arr[:, 0].tolist())
            vals[j].extend(arr[:, 1].tolist())

    # convert lists to numpy arrays
    for k in vals:
        vals[k] = np.array(vals[k]) if len(vals[k]) else np.array([])
    return vals


def _compute_limits_from_vals(vals_dict, p_low=5, p_high=95, expand_frac=0.05, preferred_limits=None):
    """
    Given dictionary target_idx -> numpy array, returns dict idx -> (min, max) limits.
    preferred_limits: dict target_idx -> (min, max) to override automatic limits.
    Percentile-based: lower = p_low percentile, upper = p_high percentile, then expand by expand_frac.
    """
    limits = {}
    for idx, arr in vals_dict.items():
        if preferred_limits and idx in preferred_limits and preferred_limits[idx] is not None:
            limits[idx] = tuple(preferred_limits[idx])
            continue

        if arr.size == 0:
            # fallback default
            limits[idx] = (0.0, 1.0)
            continue

        low = np.percentile(arr, p_low)
        high = np.percentile(arr, p_high)
        # if low == high (degenerate), broaden a bit
        if np.isclose(low, high):
            spread = np.abs(low) if low != 0 else 1.0
            low -= 0.1 * spread
            high += 0.1 * spread
        # expand by fraction
        span = high - low
        low -= expand_frac * span
        high += expand_frac * span
        # final safety: if low == high still, make +/- 1
        if np.isclose(low, high):
            low -= 1.0
            high += 1.0
        limits[idx] = (float(low), float(high))
    return limits


def _normalize_point_array(arr, lim, direction='min', map_best_to_zero=True):
    """
    Normalize a (N,)-array of values into [0,1] using lim=(low,high).
    direction: 'min' means lower values are better; 'max' means higher are better.
    If map_best_to_zero is True, the best value (according to direction) maps to 0.0 (lower is better).
    Returns array of same shape with NaNs preserved.
    """
    low, high = lim
    denom = high - low
    if denom == 0:
        # degenerate: return zeros (or NaNs?) — preserve as zeros to allow plotting
        norm = np.zeros_like(arr, dtype=float)
        norm[:] = np.nan if np.all(np.isnan(arr)) else 0.0
        return norm

    norm = (arr - low) / denom  # 0..1
    # clip
    norm = np.clip(norm, 0.0, 1.0)

    # if direction == 'min' and map_best_to_zero, lower value is better so norm already maps lower->0
    # if direction == 'max' and map_best_to_zero, invert so that best (largest) maps to 0
    if direction == 'max' and map_best_to_zero:
        norm = 1.0 - norm
    return norm


import os
import numpy as np
import matplotlib.pyplot as plt

def _safe_apply_transform(arr, transform):
    """Apply transform to numpy array `arr`. If transform is None return arr.
       Attempt to call transform(arr) (works for vectorized transforms like `lambda x: -x`).
       If that fails, fallback to elementwise application.
    """
    if transform is None:
        return arr
    try:
        out = transform(arr)
        # If transform returns a scalar for scalar input, convert to array
        return np.asarray(out)
    except Exception:
        # fallback elementwise
        vec = np.vectorize(lambda v: transform(v), otypes=[float])
        return vec(arr)

def _collect_target_vals_from_projected(pf_list, actual_pf, n_targets, value_transform=None):
    """
    Collects values per target index by scanning all pair projections, applying any value_transform.
    Returns dict idx -> numpy array of values (possibly empty).
    """
    vals = {i: [] for i in range(n_targets)}
    pair_indices = [(i, j) for i in range(n_targets) for j in range(i + 1, n_targets)]

    for pf in pf_list:
        for (i, j) in pair_indices:
            proj = pf.get_projected_pareto((i, j))
            if not proj:
                continue
            arr = np.array(proj)
            if arr.size == 0:
                continue
            # apply transforms to each column if provided
            col_x = arr[:, 0]
            col_y = arr[:, 1]
            tx = value_transform.get(i) if value_transform else None
            ty = value_transform.get(j) if value_transform else None
            col_x = _safe_apply_transform(col_x, tx)
            col_y = _safe_apply_transform(col_y, ty)
            vals[i].extend(col_x.tolist())
            vals[j].extend(col_y.tolist())

    # include actual_pf if present
    if actual_pf is not None:
        for (i, j) in pair_indices:
            proj = actual_pf.get_projected_pareto((i, j))
            if not proj:
                continue
            arr = np.array(proj)
            if arr.size == 0:
                continue
            col_x = arr[:, 0]
            col_y = arr[:, 1]
            tx = value_transform.get(i) if value_transform else None
            ty = value_transform.get(j) if value_transform else None
            col_x = _safe_apply_transform(col_x, tx)
            col_y = _safe_apply_transform(col_y, ty)
            vals[i].extend(col_x.tolist())
            vals[j].extend(col_y.tolist())

    # convert lists to numpy arrays
    for k in vals:
        vals[k] = np.array(vals[k]) if len(vals[k]) else np.array([])
    return vals


def _compute_limits_from_vals(vals_dict, p_low=5, p_high=95, expand_frac=0.05, preferred_limits=None):
    """
    Given dictionary target_idx -> numpy array, returns dict idx -> (min, max) limits.
    preferred_limits: dict target_idx -> (min, max) to override automatic limits.
    Percentile-based: lower = p_low percentile, upper = p_high percentile, then expand by expand_frac.
    """
    limits = {}
    for idx, arr in vals_dict.items():
        if preferred_limits and idx in preferred_limits and preferred_limits[idx] is not None:
            limits[idx] = tuple(preferred_limits[idx])
            continue

        if arr.size == 0:
            limits[idx] = (0.0, 1.0)
            continue

        low = np.percentile(arr, p_low)
        high = np.percentile(arr, p_high)
        if np.isclose(low, high):
            spread = np.abs(low) if low != 0 else 1.0
            low -= 0.1 * spread
            high += 0.1 * spread
        span = high - low
        low -= expand_frac * span
        high += expand_frac * span
        if np.isclose(low, high):
            low -= 1.0
            high += 1.0
        limits[idx] = (float(low), float(high))
    return limits


def _normalize_point_array(arr, lim, direction='min', map_best_to_zero=True):
    """
    Normalize a (N,)-array of values into [0,1] using lim=(low,high).
    direction: 'min' means lower values are better; 'max' means higher are better.
    If map_best_to_zero is True, the best value (according to direction) maps to 0.0.
    """
    low, high = lim
    denom = high - low
    if denom == 0:
        norm = np.full_like(arr, np.nan, dtype=float)
        return norm

    norm = (arr - low) / denom
    norm = np.clip(norm, 0.0, 1.0)
    if direction == 'max' and map_best_to_zero:
        norm = 1.0 - norm
    return norm


def plot_pareto_fronts_new(pf_list,
                       actual_pf,
                       exp_names,
                       target_names,
                       lim=None,
                       save_path=None,
                       filename="pareto_grid.png",
                       directions=None,
                       preferred_limits=None,
                       percentile_limits=(5, 95),
                       expand_frac=0.05,
                       normalize=False,
                       map_best_to_zero=True,
                       value_transform=None):
    """
    pf_list: list of ParetoFront objects.
    actual_pf: ParetoFront or None.
    exp_names: labels for pf_list.
    target_names: list of targets (strings).
    lim: optional dict or list-of-tuples for axis limits AFTER transform.
    directions: dict idx->'min'/'max' (for normalization mapping).
    preferred_limits: user overrides for limits (accepted by index or target name).
    normalize: if True, normalize axes to 0..1 AFTER transform.
    value_transform: dict index or target_name -> callable to apply to values BEFORE computing limits/plotting.
                     Example to flip dipole back to positive: {'octanol_dipole': lambda x: -x} or {3: lambda x: -x}
    """
    assert len(pf_list) == len(exp_names), "Mismatch between pf_list and exp_names"
    n_targets = len(target_names)
    pair_indices = [(i, j) for i in range(n_targets) for j in range(i + 1, n_targets)]
    ncols = min(len(pair_indices), 3)
    nrows = (len(pair_indices) + ncols - 1) // ncols

    # default directions
    if directions is None:
        directions = {i: 'min' for i in range(n_targets)}

    # build transform dict keyed by index
    vt_indexed = {}
    if value_transform:
        for k, v in value_transform.items():
            if isinstance(k, str):
                if k in target_names:
                    vt_indexed[target_names.index(k)] = v
            else:
                vt_indexed[int(k)] = v

    # preferred limits keyed by index
    pref_limits_indexed = {}
    if preferred_limits:
        for k, v in preferred_limits.items():
            if isinstance(k, str):
                if k in target_names:
                    pref_limits_indexed[target_names.index(k)] = v
            else:
                pref_limits_indexed[int(k)] = v

    # convert lim list to dict if needed
    if lim is not None and not isinstance(lim, dict):
        try:
            lim = {i: tuple(lim[i]) for i in range(n_targets)}
        except Exception:
            lim = None

    # collect values applying transforms
    vals_dict = _collect_target_vals_from_projected(pf_list, actual_pf, n_targets, value_transform=vt_indexed)

    # compute limits if needed (these limits are for transformed/display values)
    if lim is None:
        lim = _compute_limits_from_vals(vals_dict, p_low=percentile_limits[0], p_high=percentile_limits[1],
                                        expand_frac=expand_frac, preferred_limits=pref_limits_indexed)
    else:
        for i in range(n_targets):
            if i not in lim:
                lim[i] = _compute_limits_from_vals({i: vals_dict[i]}, p_low=percentile_limits[0],
                                                   p_high=percentile_limits[1],
                                                   expand_frac=expand_frac,
                                                   preferred_limits=pref_limits_indexed)[i]

    # start plotting
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows))
    axs = axs.flatten()
    colors = plt.cm.tab10.colors
    tolerance = 1e-8
    all_handles = []
    all_labels = []

    for idx, (x_idx, y_idx) in enumerate(pair_indices):
        ax = axs[idx]
        xlim, ylim = lim[x_idx], lim[y_idx]

        # Project PFs
        projected_pfs = []
        for pf in pf_list:
            proj = pf.get_projected_pareto((x_idx, y_idx))
            arr = np.array(proj) if proj else np.empty((0, 2))
            if arr.size > 0:
                # apply transform for display
                tx = vt_indexed.get(x_idx)
                ty = vt_indexed.get(y_idx)
                if tx is not None:
                    arr[:, 0] = _safe_apply_transform(arr[:, 0], tx)
                if ty is not None:
                    arr[:, 1] = _safe_apply_transform(arr[:, 1], ty)
            projected_pfs.append(arr)

        actual_proj = np.array(actual_pf.get_projected_pareto((x_idx, y_idx))) if actual_pf else np.empty((0, 2))
        if actual_proj.size > 0:
            tx = vt_indexed.get(x_idx)
            ty = vt_indexed.get(y_idx)
            if tx is not None:
                actual_proj[:, 0] = _safe_apply_transform(actual_proj[:, 0], tx)
            if ty is not None:
                actual_proj[:, 1] = _safe_apply_transform(actual_proj[:, 1], ty)

        # normalization if requested (normalizes transformed/display values)
        if normalize:
            for i_pf, pf_proj in enumerate(projected_pfs):
                if pf_proj.size == 0:
                    continue
                xvals = pf_proj[:, 0]
                yvals = pf_proj[:, 1]
                xnorm = _normalize_point_array(xvals, lim[x_idx], direction=directions.get(x_idx, 'min'),
                                               map_best_to_zero=map_best_to_zero)
                ynorm = _normalize_point_array(yvals, lim[y_idx], direction=directions.get(y_idx, 'min'),
                                               map_best_to_zero=map_best_to_zero)
                projected_pfs[i_pf] = np.vstack([xnorm, ynorm]).T
            if actual_proj.size > 0:
                xvals = actual_proj[:, 0]; yvals = actual_proj[:, 1]
                actual_proj = np.vstack([
                    _normalize_point_array(xvals, lim[x_idx], direction=directions.get(x_idx, 'min'),
                                           map_best_to_zero=map_best_to_zero),
                    _normalize_point_array(yvals, lim[y_idx], direction=directions.get(y_idx, 'min'),
                                           map_best_to_zero=map_best_to_zero)
                ]).T
            xlim = (0.0, 1.0)
            ylim = (0.0, 1.0)

        # coincidence detection (on transformed/normalized coordinates as constructed)
        pf0 = projected_pfs[0] if len(projected_pfs) > 0 else np.empty((0, 2))
        pf1 = projected_pfs[1] if len(projected_pfs) > 1 else np.empty((0, 2))

        coinc_0_actual = set()
        coinc_1_actual = set()
        coinc_between_0_1 = set()

        if pf0.size > 0 and actual_proj.size > 0:
            for pt in pf0:
                if np.any(np.all(np.abs(actual_proj - pt) < tolerance, axis=1)):
                    coinc_0_actual.add(tuple(pt))

        if pf1.size > 0 and actual_proj.size > 0:
            for pt in pf1:
                if np.any(np.all(np.abs(actual_proj - pt) < tolerance, axis=1)):
                    coinc_1_actual.add(tuple(pt))

        if pf0.size > 0 and pf1.size > 0:
            for pt in pf0:
                if np.any(np.all(np.abs(pf1 - pt) < tolerance, axis=1)):
                    pt_tuple = tuple(pt)
                    if pt_tuple not in coinc_0_actual and pt_tuple not in coinc_1_actual:
                        coinc_between_0_1.add(pt_tuple)

        # Plot PFs
        for i, pf_proj in enumerate(projected_pfs):
            if pf_proj.size > 0:
                scatter = ax.scatter(pf_proj[:, 0], pf_proj[:, 1],
                                     color=colors[i % len(colors)],
                                     label=exp_names[i], alpha=0.8)
                if exp_names[i] not in all_labels:
                    all_handles.append(scatter)
                    all_labels.append(exp_names[i])

                # stepwise lines
                ref_x = np.max(xlim)
                ref_y = np.max(ylim)
                x0, y0 = pf_proj[0]
                ax.plot([x0, x0], [ref_y, y0], color=colors[i % len(colors)], linestyle='--')
                for k in range(len(pf_proj) - 1):
                    x1, y1 = pf_proj[k]
                    x2, y2 = pf_proj[k + 1]
                    ax.plot([x1, x2], [y1, y1], color=colors[i % len(colors)], linestyle='--')
                    ax.plot([x2, x2], [y1, y2], color=colors[i % len(colors)], linestyle='--')
                last_x, last_y = pf_proj[-1]
                ax.plot([last_x, ref_x], [last_y, last_y], color=colors[i % len(colors)], linestyle='--')

        # Plot actual PF
        if actual_proj.size > 0:
            scatter = ax.scatter(actual_proj[:, 0], actual_proj[:, 1],
                                 color='black', marker='x', label='Actual Pareto')
            if "Actual Pareto" not in all_labels:
                all_handles.append(scatter)
                all_labels.append("Actual Pareto")
            ref_x = np.max(xlim)
            ref_y = np.max(ylim)
            x0, y0 = actual_proj[0]
            ax.plot([x0, x0], [ref_y, y0], 'k--')
            for k in range(len(actual_proj) - 1):
                x1, y1 = actual_proj[k]
                x2, y2 = actual_proj[k + 1]
                ax.plot([x1, x2], [y1, y1], 'k--')
                ax.plot([x2, x2], [y1, y2], 'k--')
            last_x, last_y = actual_proj[-1]
            ax.plot([last_x, ref_x], [last_y, last_y], 'k--')

        # coincidences highlights
        if coinc_0_actual:
            pts = np.array(list(coinc_0_actual))
            scatter = ax.scatter(pts[:, 0], pts[:, 1], color='red', s=80, marker='o',
                                 label=f"{exp_names[0]} coincides with Actual")
            label = f"{exp_names[0]} coincides with Actual"
            if label not in all_labels:
                all_handles.append(scatter)
                all_labels.append(label)

        if coinc_1_actual:
            pts = np.array(list(coinc_1_actual))
            scatter = ax.scatter(pts[:, 0], pts[:, 1], color='red', s=80, marker='o',
                                 label=f"{exp_names[1]} coincides with Actual")
            label = f"{exp_names[1]} coincides with Actual"
            if label not in all_labels:
                all_handles.append(scatter)
                all_labels.append(label)

        if coinc_between_0_1:
            pts = np.array(list(coinc_between_0_1))
            scatter = ax.scatter(pts[:, 0], pts[:, 1], color='purple', s=80, marker='s',
                                 label=f"{exp_names[0]} coincides with {exp_names[1]}")
            label = f"{exp_names[0]} coincides with {exp_names[1]}"
            if label not in all_labels:
                all_handles.append(scatter)
                all_labels.append(label)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(target_names[x_idx])
        ax.set_ylabel(target_names[y_idx])
        ax.grid(False)

        if normalize:
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            if map_best_to_zero:
                ax.annotate("best →", xy=(0.02, 0.98), xycoords='axes fraction', fontsize=8, va='top')

    # Hide unused axes
    for j in range(len(pair_indices), len(axs)):
        axs[j].axis('off')

    # Legend
    if all_handles:
        fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                   ncol=min(4, len(all_labels)), frameon=False, prop={'size': 12}, markerscale=1.2)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, filename), dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig


import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable


def reconstruct_generations(df):
    """Recover generation indices: gen0=500 rows, next=399 rows, then remaining."""
    n = len(df)
    gens, count, gen = [], 0, 0
    while count < n:
        if gen == 0:
            chunk = min(600, n - count)
        else:
            chunk = min(1000, n - count)
        gens.extend([gen] * chunk)
        count += chunk
        gen += 1
    
    print(f"Reconstructed {gen} generations for {n} entries.")
    return gens


def _projected_nd(points2d: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Compute 2D non-dominated set (minimization) from Nx2 points."""
    if points2d.size == 0:
        return np.empty((0, 2))
    pts = np.asarray(points2d, dtype=float)
    nd = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            if (q[0] <= p[0] + tol) and (q[1] <= p[1] + tol) and (
                (q[0] < p[0] - tol) or (q[1] < p[1] - tol)
            ):
                dominated = True
                break
        if not dominated:
            nd.append(tuple(p))
    nd = np.array(sorted(nd, key=lambda x: x[0]))
    return nd.reshape((-1, 2)) if nd.size else np.empty((0, 2))


def plot_pareto_vs_spin_from_experiment(
    experiment_folder: str,
    selected_indices: list,
    target_names: list,
    spin_target_name: str,
    query_runs_file: str,
    save_path: str = None,
    filename: str = "pareto_vs_spin.png",
    axis_limits: dict = None,
    tol: float = 1e-8,
    highlight_points: list = None,
    axis_labels: list = None   # <---- NEW ARGUMENT
):
    # --------------------------------
    # Setup fonts
    # --------------------------------
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 16
    })

    # If no custom axis labels given → default to target_names
    if axis_labels is None:
        axis_labels = target_names
    if len(axis_labels) != len(target_names):
        raise ValueError("axis_labels must have same length as target_names.")

    # Mapping index → plot label
    axis_label_map = {t: axis_labels[i] for i, t in enumerate(target_names)}

    # --------------------------------
    # Load Pareto front file
    # --------------------------------
    pareto_csv = os.path.join(experiment_folder, "pareto_fronts.csv")
    if not os.path.exists(pareto_csv):
        raise FileNotFoundError(f"Pareto CSV not found at {pareto_csv}")

    pf_df = pd.read_csv(pareto_csv)

    missing_cols = [c for c in target_names if c not in pf_df.columns]
    if missing_cols:
        raise ValueError(f"Pareto CSV missing expected columns: {missing_cols}")

    spin_idx = target_names.index(spin_target_name)
    other_indices = [i for i in range(len(target_names)) if i != spin_idx]

    # Project full PF
    pf_projected = {}
    for idx in other_indices:
        xcol = target_names[idx]
        pts = pf_df[[xcol, spin_target_name]].to_numpy(float)
        pts = pts[np.isfinite(pts).all(axis=1)]
        pf_projected[idx] = _projected_nd(pts, tol)

    # --------------------------------
    # Load evaluated query points
    # --------------------------------
    qdf = pd.read_csv(query_runs_file)

    parsed = []
    for r in qdf.get("result", []):
        if pd.isna(r):
            parsed.append(None)
            continue
        try:
            parsed.append(np.array(ast.literal_eval(r), dtype=float))
        except Exception:
            parsed.append(None)

    # --------------------------------
    # Reconstruct generations
    # --------------------------------
    reconstructed_gens = reconstruct_generations(qdf)

    eval_list = []
    gen_list = []
    for i, v in enumerate(parsed):
        if v is None:
            continue
        if len(v) <= max(selected_indices):
            continue
        eval_list.append(v[selected_indices].astype(float))
        gen_list.append(reconstructed_gens[i])

    if len(eval_list) == 0:
        raise RuntimeError("No valid evaluated points extracted.")

    eval_vals = np.vstack(eval_list)
    gen_arr = np.array(gen_list)

    mask = np.isfinite(eval_vals).all(axis=1)
    eval_vals = eval_vals[mask]
    gen_arr = gen_arr[mask]

    # --------------------------------
    # Color mapping
    # --------------------------------
    n_gen_max = int(np.max(gen_arr))

    cmap = LinearSegmentedColormap.from_list(
        "green_black",
        ["#00cc00", "#000000"],
        N=n_gen_max + 1
    )
    norm = Normalize(vmin=0, vmax=n_gen_max)

    colors = cmap(norm(gen_arr))

    # Highlight points
    highlight_vals = None
    if highlight_points is not None and len(highlight_points) > 0:
        highlight_vals = np.array(highlight_points, float)
        if highlight_vals.ndim != 2 or highlight_vals.shape[1] != len(target_names):
            raise ValueError(
                f"highlight_points must be shape (M, {len(target_names)})."
            )

    highlight_cmap = plt.get_cmap("tab10")

    valid_indices = [i for i in range(10) if i != 2]  # skip green
    highlight_colors = [
        highlight_cmap(valid_indices[i % len(valid_indices)])
        for i in range(len(highlight_points or []))
]


    # --------------------------------
    # Figure layout
    # --------------------------------
    ncols = len(other_indices)
    fig, axs = plt.subplots(
        2, ncols,
        figsize=(4.5 * ncols, 4.8),
        gridspec_kw={"height_ratios": [20, 1]},
        squeeze=False
    )
    top_axes = axs[0]
    bottom_axes = axs[1]

    # --------------------------------
    # Scatter + PF plots
    # --------------------------------
    for ax, idx in zip(top_axes, other_indices):

        ax.scatter(
            eval_vals[:, idx],
            eval_vals[:, spin_idx],
            c=colors,
            s=18,
            alpha=0.75,
        )

        if highlight_vals is not None:
            for i, (hx, hy) in enumerate(zip(highlight_vals[:, idx], highlight_vals[:, spin_idx])):
                ax.scatter(hx, hy, marker="D", s=300,
                           color=highlight_colors[i],
                           edgecolor="k", linewidth=0.6)

        # PF staircase
        pf_xy = pf_projected[idx]
        if pf_xy.size > 0:
            xs, ys = pf_xy[:, 0], pf_xy[:, 1]
            order = np.argsort(xs)
            xs, ys = xs[order], ys[order]

            zz_x = [xs[0]]
            zz_y = [ys[0]]
            for k in range(1, len(xs)):
                zz_x.extend([xs[k], xs[k]])
                zz_y.extend([zz_y[-1], ys[k]])

            ax.plot(zz_x, zz_y, "-", color="tab:blue", linewidth=3)
            ax.scatter(xs, ys, color="tab:blue", s=20)

        # Use new axis labels
        ax.set_xlabel(axis_label_map[target_names[idx]], fontsize=18)
        ax.set_ylabel(axis_label_map[spin_target_name], fontsize=18)

        if axis_limits:
            if target_names[idx] in axis_limits:
                ax.set_xlim(axis_limits[target_names[idx]])
            if spin_target_name in axis_limits:
                ax.set_ylim(axis_limits[spin_target_name])

        ax.grid(False)
        #ax.set_xscale('log')
        #ax.set_yscale('log')

    # --------------------------------
    # Wide colorbar
    # --------------------------------
    full_cax = fig.add_axes([0.18, 0.02, 0.64, 0.03])

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(np.arange(0, n_gen_max + 1))

    cbar = fig.colorbar(sm, cax=full_cax, orientation="horizontal")
    cbar.set_label("Generation", fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    for ax in bottom_axes:
        ax.remove()

    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        outp = os.path.join(save_path, filename)
        plt.savefig(outp, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved figure to {outp}")
    else:
        plt.show()

    return fig
