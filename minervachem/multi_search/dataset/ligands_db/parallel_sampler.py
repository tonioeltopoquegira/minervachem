import numpy as np
import pandas as pd
import time
import ast
import pickle
from ...utils import normalize_functionalization
from ...utils_mpi import mpi_map_registered, register_mpi_function
from .sampler import sample_single_candidate
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()




@register_mpi_function("batched_sample")
def mpi_sample_batch_worker(args):
    num_samples, seed_base, seen_keys_serialized, pareto, functionalization = args
    seen_keys = set(pickle.loads(seen_keys_serialized))

    local_seen = set()
    selected = []
    attempts = 0
    max_attempts = 100 * num_samples
    
    while len(selected) < num_samples and attempts < max_attempts:
        candidate = sample_single_candidate(1, pareto=pareto, seed=seed_base + attempts, functionalization=functionalization)
        key = (
            candidate["smiles"],
            tuple(candidate["coordList"]),
            normalize_functionalization(candidate.get("functionalization", []))
        )

        if key not in seen_keys and key not in local_seen:
            selected.append(candidate)
            local_seen.add(key)

        attempts += 1

    return {
        "candidates": selected,
        "attempts": attempts
    }



def parallel_sample_batching(n, pareto=None, seed=42, candidates_comp=None, functionalization=0, max_rounds=8, verbose=False):
   

    t = time.time()
    if candidates_comp is None:
        candidates_comp = set()

    seen = set(candidates_comp)
    selected = []
    total_attempts = 0
    remaining = n
    round_count = 0

    base_ss = np.random.SeedSequence(seed)

    while remaining > 0 and round_count < max_rounds:
        if verbose:
            print(f'Round {round_count}')
            print(f"[Master] Sampling round {round_count+1} for {remaining} candidates", flush=True)

        num_workers = size - 1
        batch_sizes = [remaining // num_workers] * num_workers
        for i in range(remaining % num_workers):
            batch_sizes[i] += 1

        # Unique child seeds for each worker
        child_seeds = base_ss.spawn(len(batch_sizes))
        inputs = []

        for i, (k, ss) in enumerate(zip(batch_sizes, child_seeds)):
            args = (
                k,
                ss.generate_state(1)[0],  # Use single uint32 for worker seed
                pickle.dumps(seen),
                pareto,
                functionalization
            )
            inputs.append(args)

        results = mpi_map_registered("batched_sample", inputs)
        round_count += 1

        proposed = []
        for r in results:
            total_attempts += r["attempts"]
            proposed.extend(r["candidates"])

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

        if remaining > 0 and verbose:
            print(f"[Sampling] Got {len(new_selected)} new candidates, {remaining} still needed", flush=True)

    print(f"[Sampling] Sampled {len(selected)} after attempts: {total_attempts} in {time.time()-t:.2f}s", flush=True)
    return selected[:n], total_attempts

