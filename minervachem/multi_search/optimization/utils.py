import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import random
import time
from collections.abc import Hashable


def identity_key(self):
    return (self.smiles, tuple(self.coordList), tuple(self.functionalization) if self.functionalization else ())

def make_hashable_functionalization(func_list):
    """
    Converts a list of functionalization dicts to a hashable, sorted tuple
    """
    if not func_list:
        return ()

    if isinstance(func_list, list):
        processed = []
        for d in func_list:
            if isinstance(d, dict):
                # Convert each dict into a sorted tuple of key-value pairs
                sorted_items = tuple(sorted(
                    (k, tuple(v) if isinstance(v, list) else v)
                    for k, v in d.items()
                ))
                processed.append(sorted_items)
            else:
                processed.append(d)
        return tuple(processed)

    # Unexpected type fallback
    return func_list




class ParetoPoint:
    def __init__(self, X, Y, smiles):
        # Accept dict or string
        if isinstance(smiles, dict):
            # prefer the legacy 'smiles' key, fallback to 'new_smiles'
            self.smiles = smiles.get('smiles') or smiles.get('new_smiles')
            self.coordList = smiles.get('coordList', []) if isinstance(smiles.get('coordList', []), list) else []
            self.functionalization = smiles.get('functionalization', None)
        else:
            self.smiles = smiles
            self.coordList = []
            self.functionalization = None
        self.X = X
        self.Y = tuple(Y)




class ParetoFront:

    def __init__(self, margin=100.0, dominating_reference_point=(-10.0, -10.0)):
        self.points = []
        self.reference_point = None
        self.margin = margin
        self.dominating_reference_point = dominating_reference_point
        self.gen_added = 0
        self.gen_deleted = 0
        self.it = 0

    def _extract_identity(self, smiles_obj):
        """
        Returns a unique, hashable identity tuple (smiles, coordList, functionalization).
        Handles dicts, ParetoPoint, and raw strings.
        """
        def get_field(obj, key):
            if isinstance(obj, dict):
                return obj.get(key, None)
            elif isinstance(obj, ParetoPoint):
                return getattr(obj, key, None)
            return None

        if isinstance(smiles_obj, str):
            smiles = smiles_obj
            coordList, functionalization = [], ()
        else:
            # try 'smiles', fallback to 'new_smiles'
            smiles = get_field(smiles_obj, "smiles")
            if smiles is None:
                smiles = get_field(smiles_obj, "new_smiles")

            coordList = get_field(smiles_obj, "coordList") or []
            functionalization = get_field(smiles_obj, "functionalization")

        return (
            smiles,
            tuple(coordList),
            make_hashable_functionalization(functionalization)
        )








    def _dominates(self, y1, y2):
        return all(a <= b for a, b in zip(y1, y2)) and any(a < b for a, b in zip(y1, y2))
    
    def is_dominated_by_front(self, y, smiles=None):
        new_id = self._extract_identity(smiles)
        for p in self.points:
            existing_id = self._extract_identity(p)
            if self._dominates(p.Y, y) or existing_id == new_id:
                return True
        return False




    def _extract_smiles(self, smiles_obj):
        if isinstance(smiles_obj, dict):
            s = smiles_obj.get("smiles")
            if s is None:
                s = smiles_obj.get("new_smiles")
            return s
        
        return smiles_obj




    def _add_raw(self, X, Y, smiles=None):
        new_point = ParetoPoint(X, Y, smiles)
        if self.is_dominated_by_front(Y, self._extract_smiles(smiles)):
            return False

        updated_points = [p for p in self.points if not self._dominates(Y, p.Y)]
        updated_points.append(new_point)
        self.points = sorted(updated_points, key=lambda p: p.Y)
        return True


    def add_points(self, X_list, Y_list, smiles_list=None, verbose=False):
        t = time.time()
        for i, (X, Y) in enumerate(zip(X_list, Y_list)):
            if any(v is None or np.isnan(v) for v in Y):
                continue
            smiles = smiles_list[i] if smiles_list is not None else None
            success = self._add_raw(X, Y, smiles)
            if success:
                self.gen_added += 1
        self.update_reference_point()
        if verbose:
            print(f"[NOT PARALLEL] Adding points {time.time() - t}s", flush=True)

    def add_w_filtering(self, valid_Y, valid_X, valid_smiles, verbose=False):
        
        filtered_Y, filtered_X, filtered_smiles = self.filter_non_dominated_batch(valid_Y, valid_X, valid_smiles, verbose=verbose)
        
        self.add_points(filtered_X, filtered_Y, filtered_smiles, verbose=verbose)



    def initialize_pareto(self, X_list, Y_list, smiles_list=None):
        self.points = []
        filtered_Y, filtered_X, filtered_smiles = self.filter_non_dominated_batch(Y_list,X_list,  smiles_list)
        self.add_points(filtered_X, filtered_Y, filtered_smiles)


    def update_reference_point(self):
        if not self.points:
            self.reference_point = None
            return

        all_Y = np.array([p.Y for p in self.points])
        worst = np.max(all_Y, axis=0)
        self.reference_point = tuple(worst + self.margin)

    def get_objectives(self):
        return np.array([p.Y for p in self.points])

    def get_features(self):
        return np.array([p.X for p in self.points])

    def get_ehvi_boxes2D(self):
        """
        Generate EHVI boxes in 2D objective space.
        Returns: list of (lower_bound, upper_bound, box_type)
        """
        if self.reference_point is None:
            self.update_reference_point()
        if not self.points:
            return []

        R1, R2 = self.reference_point
        D1, D2 = self.dominating_reference_point
        sorted_points = sorted(self.points, key=lambda p: p.Y[0])
        boxes = []

        # Dominating region
        y_prev = R2
        for point in sorted_points:
            a, b = point.Y
            if a < R1 and b < y_prev:
                boxes.append(((a, b), (R1, y_prev), "dominating"))
            y_prev = b

        # Extending region
        current_x = D1
        current_y = R2
        for point in sorted_points:
            a, b = point.Y
            if current_x < a and D2 < current_y:
                boxes.append(((current_x, D2), (a, current_y), "extending"))
            current_x = a
            if b < current_y:
                current_y = b

        if current_x < R1 and D2 < current_y:
            boxes.append(((current_x, D2), (R1, current_y), "extending"))

        return boxes
    
    def filter_non_dominated_batch(self, Y_list, X_list=None, smiles_list=None, verbose=False):
        t = time.time()
        num_points = len(Y_list)
        dominated = set()
        seen_identities = set()

        for i in range(num_points):
            if i in dominated:
                continue

            #print("Before", smiles_list[i], flush=True)

            id_i = self._extract_identity(smiles_list[i]) if smiles_list is not None else None

            #print("After", id_i, flush=True)

            

            if id_i is not None:
                if id_i in seen_identities:
                    dominated.add(i)
                    continue
                seen_identities.add(id_i)

            for j in range(i + 1, num_points):
                if j in dominated:
                    continue

                id_j = self._extract_identity(smiles_list[j]) if smiles_list is not None else None

                if id_i is not None and id_j == id_i:
                    dominated.add(j)
                    continue

                if self._dominates(Y_list[i], Y_list[j]):
                    dominated.add(j)
                elif self._dominates(Y_list[j], Y_list[i]):
                    dominated.add(i)
                    break

        idxs = [idx for idx in range(num_points) if idx not in dominated]
        filtered_Y = [Y_list[idx] for idx in idxs]

        results = [filtered_Y]
        if X_list is not None:
            filtered_X = [X_list[idx] for idx in idxs]
            results.append(filtered_X)
        if smiles_list is not None:
            filtered_smiles = [smiles_list[idx] for idx in idxs]
            results.append(filtered_smiles)

        self.it += 1
        if verbose:
            print(f"[PARETO] Filter non dominated {time.time() - t:.2f}s", flush=True)

        return tuple(results) if len(results) > 1 else filtered_Y
    
    def get_smiles(self):
        return [p.smiles for p in self.points if p.smiles is not None]


    def print_objective_values(self, dataset=None, obj_names=None, save_path=None):
        if not self.points:
            print("Pareto front is empty.")
            return

        if save_path is None:
            save_path = "."
        os.makedirs(save_path, exist_ok=True)

        csv_path = os.path.join(save_path, 'pareto_fronts.csv')

        num_objectives = len(self.points[0].Y)
        obj_names = obj_names or [f"Obj_{i}" for i in range(num_objectives)]

        # Compute percentiles (if dataset provided)
        percentiles = None
        if dataset is not None:
            valid_Y = np.array(dataset.get_valid_objective_vectors(list(range(num_objectives))))
            percentiles = [valid_Y[:, i] for i in range(num_objectives)]

        # Build data rows
        rows = []
        for point in self.points:
            row = [self.it]  # generation number
            for j, val in enumerate(point.Y):
                row.append(f"{val:.6f}")
                if percentiles:
                    perc = np.sum(percentiles[j] < val) / len(percentiles[j]) * 100
                    row.append(f"{perc:.1f}")
                else:
                    row.append("")
            row.append(point.smiles if point.smiles else "<no_smiles>")
            row.append(point.coordList if point.coordList else "")
            row.append(point.functionalization if point.functionalization else "[]")
            rows.append(row)

        # Build headers
        headers = ["generation"]
        for name in obj_names:
            headers.append(name)
            headers.append(f"{name}_percentile")
        headers.append("SMILES")
        headers.append("coordList")
        headers.append("functionalization")

        # Write to CSV (append mode)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(headers)
            writer.writerows(rows)


    def get_random_point_info(self):
        if not self.points:
            print("Pareto front is empty.")
            return None

        p = random.choice(self.points)

        info = {
            "X": p.X,
            "Objectives": p.Y,
            "SMILES": p.smiles,
            "coordList": p.coordList,
            "functionalization": p.functionalization
        }  

        return info


    def get_projected_pareto(self, obj_indices):

        i, j = obj_indices
        projected_points = [ (p.Y[i], p.Y[j]) for p in self.points ]

        # Compute non-dominated set in 2D
        non_dominated = []
        for pt in projected_points:
            if not any((other[0] <= pt[0] and other[1] <= pt[1] and (other[0] < pt[0] or other[1] < pt[1]))
                    for other in projected_points if other != pt):
                non_dominated.append(pt)

        return sorted(non_dominated, key=lambda p: p[0])
        

    
    def plot_multiple_2D(self,
                     targets_to_plot,
                     central_target,
                     targets_names,
                     all_points=None,
                     proposed_points=None,
                     selected_points=None,
                     front=None,
                     additional_front=None,
                     predicted_points=None,
                     computed_points=None,
                     show_dist=False,
                     dist_style='alpha',
                     lim=None,
                     folder_path=None,
                     filename="multiple_2D"):
        
        if selected_points is not None and computed_points is not None:

            filename = filename + f'_{self.it}_' + 'computed' + '.png'
        
        elif selected_points is not None and proposed_points is not None:

            filename = filename + f'_{self.it}_' + 'selected' + '.png'

        elif selected_points is not None and proposed_points is None:
            filename = filename + f'_{self.it}_' + 'selected_only' + '.png'
        
        elif proposed_points is not None and selected_points is None:
    
            filename = filename + f'_{self.it}_' + 'proposed' + '.png'

        else:
             filename = filename + f'_{self.it}' + '.png'


        target_indices = list(range(len(targets_names)))
        pairs = [(i, j) for i in target_indices for j in target_indices if i < j]

        ncols = min(len(pairs), 3)
        nrows = (len(pairs) + ncols - 1) // ncols

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows))
        axs = np.array(axs).flatten()

        if central_target is None:

            for idx, (i, j) in enumerate(pairs):
                ax = axs[idx]

                

                self.plot_2D(
                    targets_to_plot=[i, j],
                    targets_name=[targets_names[i], targets_names[j]],
                    all_points=all_points,
                    proposed_points=proposed_points,
                    selected_points=selected_points,
                    front=front,
                    additional_front=additional_front,
                    predicted_points=predicted_points,
                    computed_points=computed_points,
                    show_dist=show_dist,
                    dist_style=dist_style,
                    xlim=lim[i],
                    ylim=lim[j],
                    ax=ax,
                    save_path=None
                )

            # Hide any unused axes
            for i in range(len(pairs), len(axs)):
                axs[i].axis('off')


        else:
            for i, t in enumerate(targets_to_plot):

                if central_target==i:
                    continue
                ax = axs[i]
                self.plot_2D(
                    targets_to_plot=[central_target, t],
                    targets_name=[targets_names[central_target], targets_names[t]],
                    all_points=all_points,
                    proposed_points=proposed_points,
                    selected_points=selected_points,
                    front=front,
                    additional_front=additional_front,
                    predicted_points=predicted_points,
                    computed_points=computed_points,
                    show_dist=show_dist,
                    dist_style=dist_style,
                    xlim=lim[central_target],
                    ylim=lim[t],
                    ax=ax,
                    save_path=None  # avoid individual saving
                )
            
            for j in range(len(targets_to_plot), len(axs)):
                axs[j].axis('off')


        fig.tight_layout()

        if folder_path:
            os.makedirs(folder_path, exist_ok=True)
            out_path = os.path.join(folder_path, filename)
            fig.savefig(out_path)
            plt.close(fig)
        else:
            plt.show()

        



    def plot_2D(self,
            targets_to_plot,
            targets_name,
            all_points=None,
            proposed_points=None,
            selected_points=None,
            front=None,
            predicted_points=None,
            computed_points=None,
            additional_front=None,
            show_dist=False,
            dist_style='alpha',
            xlim=(-1, 12),
            ylim=(-1, 12),
            ax=None,
            save_path=None):
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure

        obj1, obj2 = targets_to_plot[0], targets_to_plot[1]

        # Plot all evaluated points
        if all_points is not None:
            all_obj = np.array(all_points)
            ax.scatter(all_obj[:, obj1], all_obj[:, obj2], color='gray', alpha=0.4, label='All Points')

        # Plot proposed points
        if proposed_points is not None:
            proposed_obj = np.array(proposed_points)
            ax.scatter(proposed_obj[:, obj1], proposed_obj[:, obj2], color='blue', alpha=0.6, label='Proposed Points')

        # Plot predicted distributions
        if show_dist and predicted_points is not None:
            n_points = predicted_points.shape[0] 
            for i in range(n_points):
                samples = predicted_points[i, [obj1, obj2], :]  
                if dist_style == 'alpha':
                    # Plot as scatter cloud for this point
                    ax.scatter(samples[0], samples[1], alpha=0.1, label=f'Point {i}' if i < 5 else "", s=10)
                elif dist_style == "kde":
                    # KDE via seaborn
                    sns.kdeplot(x=samples[0], y=samples[1], ax=ax, fill=True,
                                alpha=0.3, levels=10, label=f'Point {i}' if i < 5 else "")
                
                elif dist_style == "ellipse":
                    # Mean and covariance-based ellipse
                    
                    mean = samples.mean(axis=-1)
                    cov = np.cov(samples, rowvar=False)

                    if cov.shape == (2, 2) and np.all(np.isfinite(cov)):
                        vals, vecs = np.linalg.eigh(cov)
                        angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
                        width, height = 2 * np.sqrt(vals)  # 1-std ellipse

                        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                                        alpha=0.3, edgecolor='blue', facecolor='none',
                                        label=f'Point {i}' if i < 5 else "")
                        ax.add_patch(ellipse)
                    


        # Plot selected points
        if selected_points is not None:
            selected_obj = np.array(selected_points)
            ax.scatter(selected_obj[:, obj1], selected_obj[:, obj2], color='green', alpha=0.8, label='Selected Points')
        
        if computed_points is not None:
            comp_obj = np.array(computed_points)
            ax.scatter(comp_obj[:, obj1], comp_obj[:, obj2], color='pink', alpha=0.8, label='Computed Points')

        
        # Plot Pareto front
        pf_proj = self.get_projected_pareto((obj1, obj2))
        if pf_proj:
            pf_proj = np.array(pf_proj)
            ax.scatter(pf_proj[:, 0], pf_proj[:, 1], color='red', label='Pareto Front (Projected)')


            # Step-wise front
            if self.reference_point:
                ref_x = np.max(xlim)
                ref_y = np.max(ylim)
                x0, y0 = pf_proj[0]
                ax.plot([x0, x0], [ref_y, y0], 'r--')

                for i in range(len(pf_proj) - 1):
                    x1, y1 = pf_proj[i]
                    x2, y2 = pf_proj[i + 1]
                    ax.plot([x1, x2], [y1, y1], 'r--')
                    ax.plot([x2, x2], [y1, y2], 'r--')

                last_x, last_y = pf_proj[-1]
                ax.plot([last_x, ref_x], [last_y, last_y], 'r--')

        if additional_front is not None:
            actual_proj = additional_front.get_projected_pareto((obj1, obj2))
            if actual_proj:
                actual_proj = np.array(actual_proj)
                ax.scatter(actual_proj[:, 0], actual_proj[:, 1], color='black', marker='x', label='Actual QM9 Pareto')

                # Optional: Step-wise dashed front for actual Pareto
                if self.reference_point:  # reuse same reference
                    ref_x = np.max(xlim)
                    ref_y = np.max(ylim)
                    x0, y0 = actual_proj[0]
                    ax.plot([x0, x0], [ref_y, y0], 'k--')

                    for i in range(len(actual_proj) - 1):
                        x1, y1 = actual_proj[i]
                        x2, y2 = actual_proj[i + 1]
                        ax.plot([x1, x2], [y1, y1], 'k--')
                        ax.plot([x2, x2], [y1, y2], 'k--')

                    last_x, last_y = actual_proj[-1]
                    ax.plot([last_x, ref_x], [last_y, last_y], 'k--')
    
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(targets_name[0])
        ax.set_ylabel(targets_name[1])
        #ax.set_title("Projected Pareto Front Visualization")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        

       


if __name__ == "__main__":
    from pprint import pprint

    # Initialize ParetoFront with patched _extract_identity
    pareto = ParetoFront()

    # Define test data
    X_list = [
        [0.1, 0.2],
        [0.2, 0.3],
        [0.15, 0.25],
        [0.5, 0.5],
    ]

    Y_list = [
        [1.0, 1.0],
        [0.9, 1.2],
        [1.1, 0.95],
        [1.0, 1.0],
    ]

    smiles_list = [
        {'smiles': 'C1=CC=CC=C1', 'coordList': [0, 1], 'functionalization': {'OH': [0]}},
        {'smiles': 'C1=CC=CC=C1', 'coordList': [0, 1], 'functionalization': {'OH': [0]}},  # identical to [0]
        {'smiles': 'C1=CC=CC=C1', 'coordList': [1, 2], 'functionalization': {'NH2': [1]}}, # different coords + functionalization
        {'smiles': 'C1=CC=CC=C1', 'coordList': [0, 1], 'functionalization': {'Cl': [0]}}    # same coords, different functionalization
    ]

    # Call filter_non_dominated_batch to test
    filtered_Y, filtered_X, filtered_smiles = pareto.filter_non_dominated_batch(
        Y_list, X_list, smiles_list, verbose=True
    )

    # Display results
    print("\nFiltered Non-Dominated Points:")
    for x, y, s in zip(filtered_X, filtered_Y, filtered_smiles):
        pprint({
            "X": x,
            "Y": y,
            "identity": pareto._extract_identity(s)
        })
