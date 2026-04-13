import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import time
    import random
    from collections import deque
    import os
    import multiprocessing
    import pandas as pd
    import numpy as np
    from mpi4py import MPI
    from .utils_mpi import MPI_FUNCTIONS

    from .dataset.datastorage import Dataset
    from .optimization.ego import EGO
    from .dataset.qm9_toy.featurizer import featurize
    from .dataset.qm9_toy.sampler import sample
    from .dataset.qm9_toy.compute_pareto import get_pareto
    from .compat import ensure_smiles_dicts
    from .surrogates.mlearner_wrapper import MultiMlearner
    from .utils import initialize_workflow, timed, mask_valid, update_bb_weights, Alpha

    from .dataset.solvers import query
    from .utils_mpi import mpi_map_registered, mpi_is_master, register_mpi_function, mpi_worker_loop, stop_all_workers

def compute_alpha_floor(n_train, gen, gen_max_decay=10, min_fraction=0.1):
        """
        Compute a smoothly decaying minimum alpha bound.

        Starts at 1/n and decays to 1/(min_fraction * n) over `gen_max_decay` generations.
        """
        initial = 1.0 / n_train
        final = 1.0 / (n_train / min_fraction)  # e.g., min_fraction = 0.1 → final = 1 / (10n)
        
        decay_frac = min(gen / gen_max_decay, 1.0)  # 0 → 1
        alpha_min = (1 - decay_frac) * initial + decay_frac * final
        return alpha_min

#############################################################

def NDchemicalsearch(name, target_names,
                     targets,  n_generations,
                     n_bootstrap,
                     acquisition,
                     base_only, random_sample, cluster_size,
                     sample_size, retain, seed,
                     history_len,
                     start_weights = [1.0, 1.0, 1.0, 1.0],
                     update_weights=True):
    
    if mpi_is_master:
        # Allow rank 0 to use all CPUs on the node
        os.system("taskset -p 0xffffffffffffffff %d" % os.getpid())
        print(f"[Rank 0] Using {multiprocessing.cpu_count()} CPUs")
        n_jobs = multiprocessing.cpu_count()
    else:
        n_jobs = 1
    
    n_targets, target_indices, target_names, folder_path = initialize_workflow(targets, name, target_names)

    # Call sampler 
    t = time.time()

    init_smiles = sample(50, pareto=None, seed=seed) 

    # Instead of calling featurizer for the moment being we just search for the entry in X
    init_x = featurize(init_smiles)

    print(f'{time.time() - t}s SAMPLE & FEAT', flush=True)

    t = time.time()

    # Evaluate query functions [PARALLEL smiles] (potentially also parallel properties)
    all_properties = mpi_map_registered("query_offline", init_smiles)


    print(f'{time.time() - t}s QUERY', flush=True)
    #all_properties = run_query(init_smiles)

    data = Dataset(seed=seed)

    # ensure smiles_valid are dicts with 'new_smiles' key before adding to Dataset
    X_valid, Y_valid, smiles_valid, mask = mask_valid(target_indices, all_properties, init_x, init_smiles)
    smiles_valid_dicts = ensure_smiles_dicts(smiles_valid, prefer_key="new_smiles")
    data.add_multi_target_points(X_valid, Y_valid, target_indices, smiles_valid_dicts)

    # Initialize Pareto Frontier
    ego = EGO(X_valid, Y_valid, smiles_valid) 

    # Initialize Meta-learner wrapper
    multi = MultiMlearner(n_targets=n_targets, n_bootstrap_samples=n_bootstrap, seed=seed, target_names=target_names)

    num_samples = X_valid.shape[0]
    num_test = int(0.1 * num_samples)

    # Randomly choose indices for the test set
    test_indices = np.random.choice(num_samples, num_test, replace=False)
    Y_init_pareto = np.array(Y_valid)

    # Create test_tasks for each target index
    test_tasks = [[X_valid[test_indices], Y_init_pareto[test_indices, i]] for i in range(len(target_indices))]

    alpha = Alpha(start_weights=[(1/(data.get_tasks()[0][1].shape[0])) for _ in start_weights], history_len=history_len) # HERE COULD CAUSE ERROR IF NOT CHANGED IN FUTURE

    # Meta-train round
    multi.train([0,1,2,3], data.get_tasks(), test_sets=test_tasks, only_base=base_only, bb_weights =  alpha.alphas_history[-1], average_only=True, save_path = folder_path)
   
    #params_history[0].append(calibrations)


    if mpi_is_master():

        ego.print_generation_statistics(save_path=folder_path)
        ego.pf.print_objective_values(dataset=data, obj_names=target_names, save_path=folder_path)



    # [II] Generations

    for n in range(n_generations):
        print()
        print(f"Generation {n+1}")
        print("-"*60)

        
        # Call sampler (functionalzation) & featurizer [PARALLEL]
        
        sampled_smiles = sample(sample_size, pareto=None, seed=seed+n)

        X = featurize(sampled_smiles)

        # Compute Estimates with models [PARALLEL]
        t = time.time()
        model_pred, success_prob = multi.predict(X)
        print(f"[NOT PARALLEL] Model prediction {time.time() - t}s", flush=True)

       
        t = time.time()

        # Compute Expected Improvement and select best
        ei = ego.evaluate_eiMC(model_pred, mode=acquisition)
        print(f"[NOT PARALLEL] evaluating mcmc {time.time() - t}s", flush=True)

        
        selected_smiles, ind = ego.select(ei, sampled_smiles, retain=retain, random_sample=random_sample, cluster=cluster_size, X=X, save_path=folder_path)

       

        print(f"Selected {len(selected_smiles)}", flush=True)
        
        t = time.time()
        # Evaluate query functions
        all_properties = mpi_map_registered("query_offline", selected_smiles)
        print(f"[PARALLEL] Query {time.time() - t}s", flush=True)

    
        X_selected = X[ind]
        valid_X, valid_Y, valid_smiles, mask = mask_valid(target_indices, all_properties, X_selected, selected_smiles)
       
        smiles_valid_dicts = ensure_smiles_dicts(valid_smiles, prefer_key="new_smiles")

        mask = np.array([sm in valid_smiles for sm in selected_smiles])

        # Create new test tasks based on the new sampled points
        y_array = np.array(valid_Y)
        test_tasks = [[valid_X, y_array[:, i]] for i in range(4)]

        # Here evaluate and save the losses and calibrations on the new points
       
        calibrations = multi.evaluate_on_test_set(valid_Y, model_pred[ind][mask], None, valid_smiles, mask, target_names=['E_at',  'zvpe', 'e_gap', 'C_v'], save_path=folder_path, obj_indices=[0,1,2,3])

        Y_valid_pareto = [np.array(y)[[0,1,2,3]].tolist() for y in valid_Y]

        print(valid_smiles)
        ego.pf.add_w_filtering(Y_valid_pareto, valid_X, valid_smiles, verbose=False) 

        data.add_multi_target_points(valid_X, valid_Y, target_indices, smiles_valid_dicts)
        
        # Meta-train round
        multi.train([0,1,2,3], data.get_tasks(), test_sets=test_tasks,  only_base=base_only, average_only=True,  bb_weights = alpha.alphas_history[-1], save_path = folder_path)

        # Update all the model specific dynamic parameters
        alpha.calibration_history.append(calibrations)


        if update_weights:
            n_t_points = data.get_tasks()[0][1].shape[0]
            #low_clip = low_clip_min + (low_clip_max - low_clip_min)*(n_generations - (n+1))/n_generations
            low_clip = compute_alpha_floor(n_t_points, n, 7, 0.001) #0.0001
            alpha.update_bb_weights(clip=(low_clip, 50), verbose=True, n_train=n_t_points)


        if mpi_is_master():

            ego.print_generation_statistics(save_path=folder_path)
            ego.pf.print_objective_values(dataset=data, obj_names=target_names, save_path=folder_path)
    
    alpha.save_histories(save_path=folder_path)
            
            
    

if __name__ == '__main__':

    targets =[0, 1, 2, 3]

    @register_mpi_function("query")
    def mpi_query(smiles):
        return query(smiles, target=targets)
    
    if mpi_is_master():

        actual_pf = get_pareto()

        for n in range(0, 5):

            NDchemicalsearch(name='run_qm9_test', target_names=['E_at',  'zvpe', 'e_gap', 'C_v'],
                targets=targets, n_generations=5,
                        n_bootstrap=500,
                        sample_size=100, retain=400,
                        acquisition='pi',
                        seed=52,
                        base_only=False,
                        random_sample=False,
                        cluster_size=5,
                        start_weights=[0.03, 0.03, 0.03, 0.03],
                        update_weights=True,
                        history_len=-1
                        )
            
            
        
# save alpha
# cluster working on QM9
        
        stop_all_workers()

       
    else:
        # Workers must enter mpi_map's worker path
        # This dummy call ensures they block and receive work
        mpi_worker_loop()