import random
import numpy as np
from rdkit import Chem
from architector.io_ptable import functional_groups_dict
from architector import get_obmol_smiles, get_smiles_obmol

def functionalize(orig_mol_smiles, inds_to_keep=(0,), positions_to_functionalize=2,
                  same_functionalization=False, subs_dict=None,
                  existing_functionalization=None):
    mol = Chem.MolFromSmiles(orig_mol_smiles)
    if mol is None:
        print(f"[functionalize] Invalid SMILES: {orig_mol_smiles}")
        return None

    mol_with_Hs = Chem.AddHs(mol)

    indices_with_h = [
        atom.GetIdx()
        for atom in mol_with_Hs.GetAtoms()
        if atom.GetAtomicNum() != 1 and any(nbr.GetAtomicNum() == 1 for nbr in atom.GetNeighbors())
    ]
    indices_with_h = [i for i in indices_with_h if i not in inds_to_keep]

    if subs_dict is None:
        subs_dict = functional_groups_dict

    functionalizations = {}
    if existing_functionalization:
        for item in existing_functionalization:
            atom_idx = item['smiles_inds'][0]
            functionalizations[atom_idx] = item['functional_group']
            if atom_idx in indices_with_h:
                indices_with_h.remove(atom_idx)

    n_available = len(indices_with_h)
    if n_available == 0:
        print("[functionalize] No valid atoms left for functionalization.")
        return None

    max_possible = min(positions_to_functionalize, n_available)

    # --- Linear decay weights ---
    weights = np.linspace(1.0, 0.1, num=max_possible)
    weights /= weights.sum()

    n_to_functionalize = np.random.choice(np.arange(1, max_possible + 1), p=weights)

    for _ in range(n_to_functionalize):
        pos = random.choice(indices_with_h)
        fg = (
            list(functionalizations.values())[0]
            if (same_functionalization and functionalizations)
            else random.choice(list(subs_dict.keys()))
        )
        functionalizations[pos] = fg
        indices_with_h.remove(pos)

    fgs = [{'functional_group': fg, 'smiles_inds': [idx]} for idx, fg in functionalizations.items()]

    try:
        OBmol = get_obmol_smiles(orig_mol_smiles, functionalizations=fgs)
        new_smiles = get_smiles_obmol(OBmol)
    except Exception as e:
        print(f"[functionalize] Failed to apply functionalization: {e}")
        return None

    return {
        'smiles': orig_mol_smiles,
        'new_smiles': new_smiles,
        'coordList': inds_to_keep,
        'functionalizations': fgs
    }


'''
@register_mpi_function("batched_sample")
def mpi_sample_batch_worker(args):
    num_samples, seed_base, seen_keys_serialized, pareto = args
    seen_keys = set(pickle.loads(seen_keys_serialized))

    local_seen = set()
    selected = []
    attempts = 0
    max_attempts = 100 * num_samples
    
    while len(selected) < num_samples and attempts < max_attempts:
        candidate = sample_single_candidate(1, pareto=pareto, seed=seed_base + attempts)
        key = (
            candidate["smiles"],
            tuple(candidate["coordList"]),
            normalize_functionalization(candidate.get("functionalizations", []))
        )

        if key not in seen_keys and key not in local_seen:
            selected.append(candidate)
            local_seen.add(key)

        attempts += 1

    return {
        "candidates": selected,
        "attempts": attempts
    }



def parallel_sample_batching(n, pareto=None, seed=42, candidates_comp=None, max_rounds=3):
    if candidates_comp is None:
        candidates_comp = set()

    seen = set(candidates_comp)
    selected = []
    total_attempts = 0
    remaining = n
    round_count = 0
    seed_counter = seed

    while remaining > 0 and round_count < max_rounds:
        print(f"[Master] Sampling round {round_count+1} for {remaining} candidates", flush=True)

        num_workers = size - 1
        batch_sizes = [remaining // num_workers] * num_workers
        for i in range(remaining % num_workers):
            batch_sizes[i] += 1

        inputs = []
        for i, k in enumerate(batch_sizes):
            args = (
                k,
                seed_counter + 1000 * round_count + i,  # offset seeds per worker
                pickle.dumps(seen),
                pareto
            )
            inputs.append(args)

        results = mpi_map_registered("batched_sample", inputs)
        round_count += 1

        proposed = []
        for r in results:
            total_attempts += r["attempts"]
            proposed.extend(r["candidates"])

        # De-duplicate against global seen
        new_selected = []
        for c in proposed:
            key = (
                c["smiles"],
                tuple(c["coordList"]),
                normalize_functionalization(c.get("functionalization", []))
            )
            if key not in seen:
                seen.add(key)
                new_selected.append(c)

        selected.extend(new_selected)
        remaining = n - len(selected)

        if remaining > 0:
            print(f"[Master] Got {len(new_selected)} new candidates, {remaining} still needed", flush=True)

    if len(selected) < n:
        print(f"[Master] Warning: Only {len(selected)} unique candidates sampled (needed {n})", flush=True)

    print(f"[Master] Total attempts: {total_attempts}", flush=True)
    return selected[:n], total_attempts



adapt this to work with the new sample_single_ and to have same functionality (but in parallel) of the sample

def sample_single_candidate(n_samples: int, pareto=None, functionalization=0,
                            seed: int = 42, beta=0.5, gamma=0.5) -> dict:
    rng = np.random.default_rng(seed)

    if np.random.rand() > beta and functionalization and pareto is not None:
        point = pareto.get_random_point_info()

        if np.random.rand() > gamma:
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
        else:
            #print("[sample_single_candidate] Falling back to base ligand (functionalization failed).")
            pass

    # Group A: Base ligand fallback
    indices = rng.choice(len(df), size=n_samples, replace=False)
    row = df.iloc[indices[0]]
    coord_raw = row['lig_bind_inds']
    coord = ast.literal_eval(coord_raw) if isinstance(coord_raw, str) else coord_raw

    functionalization = row.get('functionalizations', [])
    if isinstance(functionalization, str):
        try:
            functionalization = ast.literal_eval(functionalization)
        except:
            functionalization = []

    return {
        'smiles': row['lig_smiles'],
        'new_smiles': row['lig_smiles'],
        'coordList': coord,
        'functionalizations': functionalization,
        'group': 'A'
    }




def sample(n, pareto=None, seed=None, candidates_comp=None, functionalization=None):
    
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

    print(f'Sampled {len(selected)} unique candidates after {attempts} attempts.', flush=True)


    print(selected[:5])


    return selected, attempts  # List of full candidate dicts
'''