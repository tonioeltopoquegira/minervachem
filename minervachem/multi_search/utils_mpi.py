# utils_mpi.py
import os
#import torch
from mpi4py import MPI
import pickle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



#if torch.cuda.is_available():
   #print(f"[RANK {rank}] Using GPU {torch.cuda.get_device_name(0)}")
    # GPU-heavy task
    
    #...
    #torch.cuda.synchronize()
#else:
    #print(f"[RANK {rank}] No GPU visible!")

# Barrier to sync all tasks before CPU phase
#comm.Barrier()

# Phase 2: CPU-only work
#print(f"[RANK {rank}] Starting CPU-only postprocessing")


#print(f"[RANK {rank}] OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', 'not set')}")
#print(f"[RANK {rank}] MKL_NUM_THREADS = {os.environ.get('MKL_NUM_THREADS', 'not set')}")


MPI_FUNCTIONS = {}  # global function registry


def register_mpi_function(name):
    def wrapper(func):
        MPI_FUNCTIONS[name] = func
        return func
    return wrapper


def get_mpi_function(name):
    return MPI_FUNCTIONS.get(name)


def mpi_is_master():
    return rank == 0


def mpi_worker_loop():
    while True:
        #print(f"{rank} inside the loop", flush=True)
        task = comm.recv(source=0)
        if task == "STOP":
            break
        
        func_name, idx, data_serialized = task
        func = get_mpi_function(func_name)
        if func is None:
            raise ValueError(f"[Rank {rank}] Function '{func_name}' not registered")

        arg = pickle.loads(data_serialized)
        result = func(arg)

        comm.send({
            "worker": rank,
            "index": idx,
            "result": result
        }, dest=0)


def mpi_map_registered(func_name, inputs, retry_enabled=0):

    if rank == 0:
        idle_workers = set()
        results = [None] * len(inputs)
        pending = list(enumerate(inputs))
        active = 0
        completed = 0
        total = len(inputs)
        log_every = max(1, total // 10)
        log_every = 100

        retry_count = {}  # maps original index to retry attempt count

        # Initial dispatch
        for r in range(1, size):
            if pending:
                i, inp = pending.pop(0)
                comm.send((func_name, i, pickle.dumps(inp)), dest=r)
                active += 1
            else:
                idle_workers.add(r)

        while completed < total:
            result = comm.recv()
            i = result["index"]
            worker = result["worker"]
            res = result["result"]
            worker = result["worker"]
            idle_workers.add(worker)

            results[i] = res
            completed += 1
            active -= 1

            # Retry logic if enabled 
            if retry_enabled > 0:
                should_retry = (
                    isinstance(res, dict) and
                    not res.get("success", True) and
                    not res.get("_timeout", False)  
                )

                if should_retry:
                    original_input = inputs[i]
                    if isinstance(original_input, tuple) and isinstance(original_input[1], dict):
                        ligand = original_input[0]
                        kwargs = original_input[1]

                        # Extract SMILES string for hashing
                        smiles_str = ligand["smiles"] if isinstance(ligand, dict) else str(ligand)

                        # Extract or infer initial seed
                        initial_seed = kwargs.get("_initial_seed", kwargs.get("seed", 42))

                        # Key used to track retry count
                        retry_key = (smiles_str, initial_seed)

                        count = retry_count.get(retry_key, 0)
                        
                        if count == 0:  # Only generate retries once
                            retry_count[retry_key] = retry_enabled  # record the intent to retry N times
                            for retry_attempt in range(1, retry_enabled + 1):
                                new_seed = initial_seed + retry_attempt
                                #print(f"\n[Rank 0] Scheduling retry {retry_attempt} for job {i} with seed {new_seed}", flush=True)

                                new_kwargs = {
                                    k: v for k, v in kwargs.items() if k != "_initial_seed"
                                }
                                new_kwargs["seed"] = new_seed
                                new_kwargs["_initial_seed"] = initial_seed  # used for grouping results later

                                new_input = (ligand, new_kwargs)
                                inputs.append(new_input)
                                results.append(None)
                                pending.append((len(results) - 1, new_input))
                                total += 1
                            
                            
                            while pending and idle_workers:
                                next_i, next_inp = pending.pop(0)
                                next_worker = idle_workers.pop()
                                comm.send((func_name, next_i, pickle.dumps(next_inp)), dest=next_worker)
                                active += 1


            # Progress logging
            if (completed) % log_every == 0 or completed == total:
                print(f"\r[Rank 0] {func_name} progress: {completed}/{total} tasks complete", end='', flush=True)

            # Dispatch next task if any
            if pending:
                next_i, next_inp = pending.pop(0)
                comm.send((func_name, next_i, pickle.dumps(next_inp)), dest=worker)
                active += 1
                idle_workers.discard(worker)
            else:
                idle_workers.add(worker)

        if retry_enabled > 0:
            results = filter_results_retry(inputs, results)
            
        return results

    else:
        mpi_worker_loop()


'''if count < retry_enabled:
                            retry_count[retry_key] = count + 1
                            new_seed = initial_seed + count + 1

                            print(f"\n[Rank 0] Retrying job {i} (attempt {count + 1}) with seed {new_seed}", flush=True)

                            # Prepare new kwargs
                            new_kwargs = {
                                k: v for k, v in kwargs.items() if k != "_initial_seed"
                            }
                            new_kwargs["seed"] = new_seed
                            new_kwargs["_initial_seed"] = initial_seed  # hidden param used only for tracking

                            new_input = (ligand, new_kwargs)
                            inputs.append(new_input)
                            results.append(None)
                            pending.append((len(results) - 1, new_input))
                            total += 1'''


def filter_results_retry(inputs, results):
    final_results = {}
    result_map = {}

    for idx, (inp, res) in enumerate(zip(inputs, results)):
        if not isinstance(inp, tuple) or not isinstance(inp[1], dict):
            final_results[idx] = res  # non-retryable inputs
            continue

        ligand, kwargs = inp
        smiles_str = ligand["smiles"] if isinstance(ligand, dict) else str(ligand)
        initial_seed = kwargs.get("_initial_seed", kwargs.get("seed", 42))
        retry_key = (smiles_str, initial_seed)

        if retry_key not in result_map:
            result_map[retry_key] = {"success": [], "fail": [], "all": []}

        result_map[retry_key]["all"].append(res)

        if isinstance(res, dict) and res.get("success", False):
            result_map[retry_key]["success"].append(res)
        else:
            result_map[retry_key]["fail"].append(res)

    # Flatten results — pick first success, or first fail, and always add "attempted"
    for group in result_map.values():
        attempts = len(group["all"])

        if group["success"]:
            chosen = group["success"][0]
        else:
            chosen = group["fail"][0]

        if isinstance(chosen, dict):
            chosen = chosen.copy()  # avoid modifying shared result
            chosen["attempted"] = attempts

        final_results[len(final_results)] = chosen

    # Replace results list with cleaned-up one
    results = [final_results[i] for i in range(len(final_results))]
    return results



def stop_all_workers():
    print('Stopping all workers')
    if mpi_is_master():
        for r in range(1, size):
            comm.send("STOP", dest=r)

