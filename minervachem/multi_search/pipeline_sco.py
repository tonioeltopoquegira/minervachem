import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

    import os
    #os.environ["MPI_DISABLED"] = "1"
    os.environ["OMP_NUM_THREADS"] = os.environ.get("SLURM_CPUS_PER_TASK", "1")

    import time
    import random

    import csv

    import pandas as pd
    import numpy as np
    import multiprocessing
    from .utils_mpi import MPI_FUNCTIONS

    from .dataset.datastorage import Dataset, cluster_centroids
    from .optimization.ego import EGO
    from .dataset.ligands_db.featurizer import featurize, fit_featurizer, transform_featurizer, transform
    from .dataset.ligands_db.sampler import sample
    from .dataset.ligands_db.parallel_sampler import parallel_sample_batching
    #from .dataset.ligands_db.compute_pareto import get_pareto
    from .surrogates.mlearner_wrapper import MultiMlearner
    from minervachem.fingerprinters import GraphletFingerprinter
    from minervachem.transformers import FingerprintFeaturizer

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

    
    from .utils import initialize_workflow, timed, mask_valid, write_query_run, load_query_run, retrieve_computed_candidates, retrieve_time_data, log_candidate_success_extremes, update_bb_weights, Alpha

    from .dataset.solvers import query_lig as query
    from .utils_mpi import mpi_map_registered, mpi_is_master, register_mpi_function, mpi_worker_loop, stop_all_workers

warnings.filterwarnings("ignore", category=UserWarning, module="architector.io_lig")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="architector.io_core")
warnings.filterwarnings("ignore", category=FutureWarning, module="scipy.optimize._qap")
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")
warnings.filterwarnings(
    "ignore",
    message="flaml.automl is not available. Please install flaml\\[automl\\] to enable AutoML functionalities.",
    category=UserWarning,
    module="flaml.__init__"
)

#print(f"[Rank {rank}] sees CPU count = {cpu_count}, affinity = {affinity}, PID = {os.getpid()}")

#############################################################

def NDchemicalsearch(name, seed, train_names,
                     
                     start_from_scratch,
                     
                     # Decide which targets to optimize 
                    obj_targets,

                     # Generations Options
                     initial_set_size,
                     n_generations,
                     n_bootstrap,

                     # Surrogates Options
                     base_only, 
                     
                     # Sampling Options
                     random_sample,
                     sample_size, retain, 
                     acquisition,
                     functional=0,
                     cluster=None,

                     # Calibrations Options
                     update_weights = True,
                     start_weights = [0.03,0.03,0.03],
                     history_len=-1,
                     low_clip_min = 0.02,
                     low_clip_max = 0.03,
                     min_val_lowbound = 0.0001,
                     

                     # Featurizer
                     max_len = 5,

                    # Other Options
                    verbose = True,
                    plot=True,
                    lim = None):
    
    obj_names = [train_names[o] for o in obj_targets]
    print(f'Objectives: {obj_names}', flush=True)
    print(f'Total tasks: {train_names}', flush=True)
    train_targets = list(range(len(train_names)))

    if mpi_is_master:
        # Allow rank 0 to use all CPUs on the node
        os.system("taskset -p 0xffffffffffffffff %d" % os.getpid())
        print(f"[Rank 0] Using {multiprocessing.cpu_count()} CPUs")
        n_jobs = multiprocessing.cpu_count()
    else:
        n_jobs = 1

    
    log_state = {}
    n_targets, target_indices, obj_names, folder_path = initialize_workflow(train_targets, name, obj_names) # for testing purposes, should be train_targets and obj_names

    featurizer = FingerprintFeaturizer(
                fingerprinter=GraphletFingerprinter(max_len=max_len),
                verbose=0,         # Optional verbosity parameter
                # Parallel Arguments
                n_jobs=n_jobs,         # For joblib, this means all n_cores-2. 
                chunk_size='auto', # Optional, how many molecules each core should do in a batch.
            )
    
    data = Dataset(seed=seed, standardize=False, featurizer=featurizer)
    
    if start_from_scratch and verbose:

        print("Starting from scratch: sampling and querying new data.", flush=True)

        init_smiles, attempted  = parallel_sample_batching(initial_set_size, pareto=None, seed=seed, functionalization=functional, verbose=verbose)

        inputs = [(smiles, {"target": train_names, "seed": seed}) for smiles in init_smiles]
        all_properties = mpi_map_registered("query", inputs, retry_enabled=0)

        write_query_run(all_properties, folder_path, 0, log_state)
        

        init_n = 0

    else:
        all_properties, init_smiles, init_n = load_query_run(folder_path)

        if all_properties is None:
            print("Fallback: sampling because no prior data found.", flush=True)

            t = time.time()
            init_smiles, attempted  = parallel_sample_batching(initial_set_size, pareto=None, seed=seed, functionalization=functional, verbose=verbose)

            inputs = [(smiles, {"target": train_names, "seed": seed}) for smiles in init_smiles]
            all_properties = mpi_map_registered("query", inputs, retry_enabled=0)

            write_query_run(all_properties, folder_path, 0, log_state)

            init_n = 0

    X_valid, Y_valid, smiles_valid = data.prepare_batch(
        smiles=init_smiles,
        all_properties=all_properties,
        target_indices=train_targets, # it was target_indices
        verbose=verbose
    )


    data.add_multi_target_points(X_valid, Y_valid, target_indices, smiles_valid)

    print("Number of tasks: ", len(data.get_tasks()))

    # Initialize Pareto Frontier

    # this one should not take all of them... only Y_valid corresponding to obj_indices

    Y_valid_pareto = [np.array(y)[obj_targets].tolist() for y in Y_valid]


    ego = EGO(X_valid, Y_valid_pareto, smiles_valid, initial_set_size=initial_set_size) 

    # Initialize Meta-learner wrapper
    # should distinguish targets for which we need bootstrap with targets that need only one base solve without meta or bootstrap
    multi = MultiMlearner(n_targets=n_targets, n_bootstrap_samples=n_bootstrap, seed=seed, target_names=train_names) 

    num_samples = X_valid.shape[0]
    num_test = int(0.5 * num_samples)

    # Randomly choose indices for the test set
    test_indices = np.random.choice(num_samples, num_test, replace=False)
    Y_array = np.array(Y_valid)

    # Create test_tasks for each target index
    test_tasks = [[X_valid[test_indices], Y_array[test_indices, i]] for i in range(len(train_targets))] # was 

    time_task = retrieve_time_data(folder_path=folder_path, featurizer=featurizer)

    test_time = [time_task[0][test_indices], time_task[1][test_indices]] 

    alpha = Alpha(start_weights=[(1/(data.get_tasks()[0][1].shape[0])) for _ in start_weights], history_len=history_len) # HERE COULD CAUSE ERROR IF NOT CHANGED IN FUTURE

    # Meta-train round
    multi.train(obj_targets, data.get_tasks(), test_sets=test_tasks, time_task=time_task, test_time=test_time, only_base=base_only, bb_weights = alpha.alphas_history[-1], average_only=True, save_path = folder_path)


    if mpi_is_master():

        ego.print_generation_statistics(sampled_smiles=init_smiles, prop=all_properties, attempted=initial_set_size, save_path=folder_path)
        ego.pf.print_objective_values(dataset=data, obj_names=obj_names, save_path=folder_path) # correctly has obj_names


    # [II] Generations
    for n in range(init_n, n_generations):
        print()
        print(f"Generation {n+1}")
        print("-"*60)

        # retrieve computed candidates smiles, coordList, functionalization
        comp_candidates = retrieve_computed_candidates(folder_path)

        # Call sampler
        sampled_smiles, attempted = parallel_sample_batching(sample_size, pareto=ego.pf, candidates_comp=comp_candidates, seed=seed+n, functionalization=functional, verbose=verbose)

        
        # Transform the new smiles based on the previous featurization
        X, sampled_smiles, valid_mask = data.transform_smiles(sampled_smiles, verbose=verbose)


        # Predict new points based on the previous featurization
        model_pred, success_pred = multi.predict(X, obj_indices=obj_targets, verbose=verbose)

        log_candidate_success_extremes(sampled_smiles, success_pred, n, folder_path=folder_path)
        
        pred_cent = np.mean(model_pred, axis=-1)
        uq_cent = np.std(model_pred, axis=-1)
       
        # Compute Expected Improvement for the computed points
        ei = ego.evaluate_eiMC(model_pred, mode=acquisition, verbose=verbose)

        # Select the based points based on the previously computed improvement
        selected_smiles, ind = ego.select(ei, sampled_smiles, cluster=cluster, retain=retain, random_sample = random_sample, save_path=folder_path, success_prob=success_pred, verbose=verbose, X=X)
        
        # Compute the new properties for the selected points
        inputs = [(smiles, {"target": train_names, "seed": seed}) for smiles in selected_smiles]
        all_properties = mpi_map_registered("query", inputs, retry_enabled=0)

        write_query_run(all_properties, folder_path, n+1, log_state)

        # Filter out data with None properties
        valid_X, valid_Y, valid_smiles = data.prepare_batch(
        smiles=selected_smiles,
        all_properties=all_properties,
        target_indices=target_indices,
        verbose=verbose,
        fit=False)

        mask = np.array([sm in valid_smiles for sm in selected_smiles])

        # Create new test tasks based on the new sampled points
        y_array = np.array(valid_Y)
        test_tasks = [[valid_X, y_array[:, i]] for i in range(len(train_targets))]
        time_task = retrieve_time_data(folder_path=folder_path, featurizer=featurizer)
        test_time = [valid_X, mask]

        # Here evaluate and save the losses and calibrations on the new points
        # Compute losses for all of them + loss for the success 
        calibrations = multi.evaluate_on_test_set(valid_Y, model_pred[ind][mask], success_pred[ind], valid_smiles, mask, target_names=obj_names, save_path=folder_path, obj_indices=obj_targets)

        Y_valid_pareto = [np.array(y)[obj_targets].tolist() for y in valid_Y]
        ego.pf.add_w_filtering(Y_valid_pareto, valid_X, valid_smiles, verbose=verbose) 

        # Add valid data to dataset
        data.add_multi_target_points(valid_X, valid_Y, target_indices, valid_smiles)

        # Get new featurization
        data.fit_featurizer(verbose=verbose)

        # Re-transform valid_X with updated featurizer
        valid_X, _, _ = data.transform_smiles(valid_smiles, verbose=verbose)

        # Create test tasks now
        y_array = np.array(valid_Y)
        test_tasks = [[valid_X, y_array[:, i]] for i in range(len(target_indices))]
        time_task = retrieve_time_data(folder_path=folder_path, featurizer=data.featurizer)
        test_time = [valid_X, mask]


        # The train round should inside do some sort of folding to evaluate the loss and calibration
        multi.train(obj_targets, data.get_tasks(), test_sets=test_tasks, time_task=time_task, test_time=test_time, only_base=base_only,  bb_weights = alpha.alphas_history[-1],  average_only=True, save_path = folder_path)
        
        # Update all the model specific dynamic parameters
        alpha.calibration_history.append(calibrations)

        if update_weights:
            n_t_points = data.get_tasks()[0][1].shape[0]
            #low_clip = low_clip_min + (low_clip_max - low_clip_min)*(n_generations - (n+1))/n_generations
            low_clip = compute_alpha_floor(n_t_points, n, 7, min_val_lowbound) #0.0001
            alpha.update_bb_weights(clip=(low_clip, 50), verbose=verbose, n_train=n_t_points)
        
        if mpi_is_master():

            ego.print_generation_statistics(sampled_smiles=sampled_smiles, prop=all_properties, attempted=attempted, save_path=folder_path)
            ego.pf.print_objective_values(dataset=data, obj_names=obj_names, save_path=folder_path)
    
    alpha.save_histories(save_path=folder_path)
            
    

if __name__ == '__main__':

    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()

        # Core identifiers
        parser.add_argument("--name", type=str, required=True)
        parser.add_argument("--seed", type=int, default=0)
        
        # Dataset options
        parser.add_argument("--start_from_scratch", action="store_true")

        # Objective targets (hardcoded as [0, 7, 8, 9] in the call)
        parser.add_argument("--train_targets", type=int, nargs='+', required=True)
        parser.add_argument("--obj_targets", type=int, nargs='+', default=[0, 7, 8, 9])

        # Generations options
        parser.add_argument("--initial_set_size", type=int, default=500)
        parser.add_argument("--n_generations", type=int, default=15)
        parser.add_argument("--n_bootstrap", type=int, default=100)

        # Surrogate learning options
        parser.add_argument("--base_only", action="store_true")

        # Sampling options
        parser.add_argument("--random_sample", action="store_true")
        parser.add_argument("--sample_size", type=int, default=40000)
        parser.add_argument("--retain", type=float, default=0.025)
        parser.add_argument("--acquisition", type=str, default="pi")
        parser.add_argument("--functional", type=int, default=2)
        parser.add_argument("--cluster", type=int, default=None)

        # Calibration / Alpha options
        parser.add_argument("--update_weights", action="store_true")
        parser.add_argument("--start_weights", type=float, nargs='+', default=[0.01, 0.01, 0.01, 0.01])
        parser.add_argument("--history_len", type=int, default=-1)
        parser.add_argument("--lowbound", type=int, default=0.0001)

        # Featurizer
        parser.add_argument("--max_len", type=int, default=5)

        # Misc options
        parser.add_argument("--plot", action="store_true")
        parser.add_argument("--verbose", action="store_true")

        return parser.parse_args()



    
    if mpi_is_master():

        args = parse_args()

        train_names = [
            'sco_kcal',
            'water_gsolv_eV', 'water_hl_gap_eV', 'water_dipole',
            'acetone_gsolv_eV', 'acetone_hl_gap_eV', 'acetone_dipole',
            'octanol_gsolv_eV', 'octanol_hl_gap_eV', 'octanol_dipole',
            'hexane_gsolv_eV', 'hexane_hl_gap_eV', 'hexane_dipole'
        ]

       

        NDchemicalsearch(
            name=args.name,
            seed=args.seed,
            train_names=train_names,
            obj_targets=args.obj_targets,
            start_from_scratch=args.start_from_scratch,

            # Generation & training
            initial_set_size=args.initial_set_size,
            n_generations=args.n_generations,
            n_bootstrap=args.n_bootstrap,

            # Surrogates
            base_only=args.base_only,

            # Sampling
            sample_size=args.sample_size,
            retain=args.retain,
            acquisition=args.acquisition,
            random_sample=args.random_sample,
            functional=args.functional,
            cluster=args.cluster,

            # Alpha / calibration
            update_weights=args.update_weights,
            start_weights=args.start_weights,
            history_len=args.history_len,
            min_val_lowbound=args.lowbound,

            # Featurizer
            max_len=args.max_len,

            # Misc
            plot=args.plot,
            verbose=args.verbose,

            # Optional fixed limits
            lim=[
                (0, 50),
                (-500, 50),
                (-5.0, 50.0),
                (-50.0, 5.0),
            ]
        )

        
       
        
        stop_all_workers()

        with open("workflow_done.flag", "w") as f:
            f.write("done\n")

       
    else:
        # Workers must enter mpi_map's worker path
        # This dummy call ensures they block and receive work
        mpi_worker_loop()

        

