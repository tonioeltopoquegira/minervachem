import numpy as np
import pandas as pd
import ast
import time
import os
from ...utils import normalize_functionalization
from .functionalization import functionalize

# Get the directory where this module is located
_module_dir = os.path.dirname(os.path.abspath(__file__))

df = pd.read_pickle(os.path.join(_module_dir, 'initial_ligands.pkl'))

def sample_single_candidate(n_samples: int, pareto=None, functionalization=0,
                            seed: int = 42, beta=0.95, gamma=0.5) -> dict:
    
    rng = np.random.default_rng(seed)

    def sample_group_a():
        idx = rng.integers(len(df))
        row = df.iloc[idx]
        coord_raw = row['lig_bind_inds']
        coord = ast.literal_eval(coord_raw) if isinstance(coord_raw, str) else coord_raw

        return {
            'smiles': row['lig_smiles'],
            'new_smiles': row['lig_smiles'],
            'coordList': coord,
            'functionalizations': [],
            'group': 'A'
        }

    # Decide group: A (β) or B/C (1−β)
    if rng.random() < beta:

        point = sample_group_a()

        if rng.random() > gamma or not functionalization:
            return point
        
        else:

            result = functionalize(
                point['smiles'],
                inds_to_keep=point.get('coordList', []),
                positions_to_functionalize=functionalization,
                existing_functionalization=point.get('functionalizations', [])
            )

            group = 'A'

            if result is not None:
                return {
                    'smiles': result['smiles'],
                    'new_smiles': result['new_smiles'],
                    'coordList': point['coordList'],
                    'functionalizations': result['functionalizations'],
                    'group': group
                }



            


    # Try B/C sampling (functionalized) if allowed
    if functionalization and pareto is not None:
        point = pareto.get_random_point_info()

        if rng.random() > gamma:
            # Group B: Added functionalization to existing
            result = functionalize(
                point['SMILES'],
                inds_to_keep=point.get('coordList', []),
                positions_to_functionalize=functionalization,
                existing_functionalization=point.get('functionalizations', [])
            )
            group = "B"
        else:
            # Group C: New functionalization
            result = functionalize(
                point['SMILES'],
                inds_to_keep=point.get('coordList', []),
                positions_to_functionalize=functionalization
            )
            group = "C"

        if result is not None:
            return {
                'smiles': result['smiles'],
                'new_smiles': result['new_smiles'],
                'coordList': point['coordList'],
                'functionalizations': result['functionalizations'],
                'group': group
            }

    # Fallback if B/C fails or not allowed
    return sample_group_a()




def sample(n, pareto=None, seed=None, candidates_comp=None, functionalization=None):

    t = time.time()
    
    if candidates_comp is None:
        candidates_comp = set()

    selected = []
    seen = set()

    max_attempts = 100 * n
    attempts = 0

    while len(selected) < n and attempts < max_attempts:
        candidate = sample_single_candidate(1, pareto=pareto, seed=seed + attempts, functionalization=functionalization)

        key = (
            candidate["smiles"],
            tuple(candidate["coordList"]),
            normalize_functionalization(candidate.get("functionalizations", []))
        )

        if key not in candidates_comp and key not in seen:
            selected.append(candidate)
            seen.add(key)
        else:            
            #print(f"Skipping duplicate candidate: {key}", flush=True)
            pass
        
        attempts += 1

    if len(selected) < n:
        print(f"Only found {len(selected)} unique new candidates (requested {n}).", flush=True)

    print(f'Sampled {len(selected)} unique candidates after {attempts} attempts in {time.time()-t}', flush=True)




    return selected, attempts  # List of full candidate dicts

