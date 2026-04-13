import os
import time
import sys
import csv
import numpy as np
import pickle
from pickle import loads
from .mlearner.metalearner_v1 import MetaLearner
from ..utils_mpi import register_mpi_function, mpi_is_master, mpi_map_registered
from ..utils_mpi import MPI_FUNCTIONS, rank

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def bayesian_bootstrap_weights(ns, n_bootstrap, seed, bb_weights=None):

    rng = np.random.default_rng(seed)
    if bb_weights is None:
        bb_weights = [100000.0] * len(ns)

    assert len(bb_weights) == len(ns), "Length of bb_weights must match number of targets"


    print(f"Adaptive weights: {bb_weights}", flush=True)

    weights = []
    for n, alpha in zip(ns, bb_weights):
        w = rng.dirichlet(alpha=np.ones(n) * alpha, size=n_bootstrap).T  # shape: (n, n_bootstrap)
        weights.append(w)

    return weights

@register_mpi_function("fit_base_task")
def fit_base_task(args):
    t, b, task_serialized, w_serialized, config_serialized = args
    X_train, y_train = pickle.loads(task_serialized)
    w = pickle.loads(w_serialized)
    config = pickle.loads(config_serialized)

    learner = MetaLearner(**config)
    error_train, error_test, coeff, params = learner.fit_base([(X_train, y_train)], sample_weight=w)

    return t, b, error_train, error_test, coeff, params






@register_mpi_function("fit_meta_task")
def fit_meta_task(args):
    t, b, task_serialized, w_serialized, config_serialized, other_coeffs_serialized, other_params_serialized = args
    X_train, y_train = pickle.loads(task_serialized)
    w = pickle.loads(w_serialized)
    config = pickle.loads(config_serialized)
    other_coeffs = pickle.loads(other_coeffs_serialized)
    other_params = pickle.loads(other_params_serialized)

    learner = MetaLearner(**config)
    error_train, error_test = learner.fit_meta([(X_train, y_train)],
                                               sample_weight=w,
                                               other_coeff=other_coeffs,
                                               other_params=other_params)

    # Collect minimal diagnostics
    diagnostics = {
        'parallel_norm': getattr(learner, 'last_parallel_norm', None),
        'perp_norm': getattr(learner, 'last_perp_norm', None)
    }

    return t, b, error_train, error_test, learner.coeff, learner.params, pickle.dumps(diagnostics, protocol=pickle.HIGHEST_PROTOCOL)







class MultiMlearner:

    def __init__(self, n_targets: int, n_bootstrap_samples: int, seed: int, hierarchical: bool = False, target_names: list = None):
        self.n_targets = n_targets
        self.n_bootstrap = n_bootstrap_samples
        self.seed = seed
        self.hierarchical = hierarchical
        self.train_times = 0
        self.target_names = target_names
        self.best_time_params = None

        # Each [target][bootstrap] = MetaLearner
        self.meta_learners = [
            [MetaLearner(random_state=seed, hierarchical=hierarchical) for _ in range(n_bootstrap_samples)]
            for _ in range(n_targets)
        ]

        # Per target and bootstrap: base fit results
        self.base_coeff = [[None] * n_bootstrap_samples for _ in range(n_targets)]
        self.base_params = [[None] * n_bootstrap_samples for _ in range(n_targets)]

        self.base_train_errors = [[None] * n_bootstrap_samples for _ in range(n_targets)]
        self.meta_train_errors = [[None] * n_bootstrap_samples for _ in range(n_targets)]
        self.base_test_errors = [[None] * n_bootstrap_samples for _ in range(n_targets)]
        self.meta_test_errors = [[None] * n_bootstrap_samples for _ in range(n_targets)]
        # diagnostics containers (per target x bootstrap)
        self.parallel_norms = [[None] * n_bootstrap_samples for _ in range(n_targets)]
        self.perp_norms = [[None] * n_bootstrap_samples for _ in range(n_targets)]
        self.perp_over_par = [[None] * n_bootstrap_samples for _ in range(n_targets)]

    
        self.weights = None

    def train(self, obj_indices, tasks, test_sets, time_task=None, test_time=None, only_base = False, average_only=False,featurizer=None, bb_weights=None, save_path=False, log_res = True, verbose=True):

        if test_sets is None:
            raise ValueError("`test_sets` must be provided and cannot be None.")

        # Merge into full 4-tuple tasks
        full_tasks = []
        for (target_idx, X_train, y_train), (X_test, y_test) in zip(tasks, test_sets):
            full_tasks.append((X_train, y_train, X_test, y_test))

        print("Fitting base models...", flush=True)
        t = time.time()
        self.fit_base(full_tasks, bb_weights=bb_weights, featurizer=featurizer, obj_indices=obj_indices)
        print(f"Base fitted in {time.time()-t}s", flush=True)

        if only_base:
            return
        print("Fitting meta models...", flush=True)
        t = time.time()
        self.fit_meta(full_tasks, featurizer=featurizer, obj_indices=obj_indices)
        print(f"Meta fitted in {time.time()-t}s", flush=True)

        # fit success
        if time_task is not None and test_time is not None:
            t = time.time()
            err_tr = self.fit_success(time_task, test_time)
            if verbose:
                print("Success model fitted in", time.time() - t, "s", flush=True)
                print("Success train error:", err_tr, flush=True)


        self.train_times += 1
        #self.logrmse(full_tasks, average_only=False, save_path=save_path)
        if log_res:
            self.logrmse_csv(full_tasks, average_only=average_only, obj_indices=obj_indices, save_path=save_path)
            if save_path:
                self.log_model_metrics(save_path)

            #self.error_correlation_avg(save_path=save_path)
    
    def fit_base(self, tasks, bb_weights, featurizer=None, obj_indices=None):
        ns = [X.shape[0] for X, _, _, _ in tasks]

        if obj_indices is None:
            obj_indices = list(range(self.n_targets))

        # Sample BB weights only for obj targets
        self.weights = []
        for t in range(self.n_targets):
            n = ns[t]
            if t in obj_indices:
                idx = obj_indices.index(t)
                alpha = bb_weights[idx] if bb_weights else 0.03
                w = np.random.default_rng(self.seed).dirichlet(alpha=np.ones(n) * alpha, size=self.n_bootstrap).T
            else:
                w = np.ones((n, 1))
            self.weights.append(w)

        if mpi_is_master():
            job_args = []
            for t, full_task in enumerate(tasks):
                X_train, y_train, _, _ = full_task
                task_data = (
                    X_train.astype(np.float32, copy=False),  # preserve sparse
                    y_train.astype(np.float32, copy=False)
                )

                for b in range(self.n_bootstrap if t in obj_indices else 1):
                    w = self.weights[t][:, b] if t in obj_indices else self.weights[t].flatten()

                    config = {'random_state': self.seed, 'hierarchical': self.hierarchical}
                    job_args.append((
                        t, b,
                        pickle.dumps(task_data, protocol=pickle.HIGHEST_PROTOCOL),
                        pickle.dumps(w, protocol=pickle.HIGHEST_PROTOCOL),
                        pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
                    ))

            results = mpi_map_registered("fit_base_task", job_args)

            for t, b, error_train, error_test, coeff, params in results:
                if t not in obj_indices:
                    for bb in range(self.n_bootstrap):
                        self.base_train_errors[t][bb] = error_train
                        self.base_test_errors[t][bb] = error_test
                        self.base_coeff[t][bb] = coeff
                        self.base_params[t][bb] = params
                        self.meta_learners[t][bb].coeff = coeff
                        self.meta_learners[t][bb].params = params
                else:
                    self.base_train_errors[t][b] = error_train
                    self.base_test_errors[t][b] = error_test
                    self.base_coeff[t][b] = coeff
                    self.base_params[t][b] = params
                    self.meta_learners[t][b].coeff = coeff
                    self.meta_learners[t][b].params = params



    def fit_meta(self, tasks, featurizer=None, obj_indices=None):
        if obj_indices is None:
            obj_indices = list(range(self.n_targets))

        if mpi_is_master():
            job_args = []
            for t, full_task in enumerate(tasks):
                if t not in obj_indices:
                    continue

                X_train, y_train, _, _ = full_task
                task_data = (
                    X_train.astype(np.float32, copy=False), 
                    y_train.astype(np.float32, copy=False)
                )

                for b, learner in enumerate(self.meta_learners[t]):
                    w = self.weights[t][:, b]

                    other_coeffs = [self.base_coeff[ot][b] for ot in range(self.n_targets) if ot != t]
                    other_params = self.base_params[t][b]

                    config = {'random_state': self.seed, 'hierarchical': self.hierarchical}
                    job_args.append((
                        t, b,
                        pickle.dumps(task_data, protocol=pickle.HIGHEST_PROTOCOL),
                        pickle.dumps(w, protocol=pickle.HIGHEST_PROTOCOL),
                        pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL),
                        pickle.dumps(other_coeffs, protocol=pickle.HIGHEST_PROTOCOL),
                        pickle.dumps(other_params, protocol=pickle.HIGHEST_PROTOCOL)
                    ))

            results = mpi_map_registered("fit_meta_task", job_args)

            for t, b, error_train, error_test, coeff, params, diag_serialized in results:
                # unpack diagnostics from worker
                diagnostics = pickle.loads(diag_serialized) if diag_serialized is not None else {}
                par_norm = diagnostics.get('parallel_norm', None)
                perp_norm = diagnostics.get('perp_norm', None)

                self.meta_train_errors[t][b] = error_train
                self.meta_test_errors[t][b] = error_test
                self.meta_learners[t][b].coeff = coeff
                self.meta_learners[t][b].params = params

                # store norms and ratio (handle zero/None safely)
                try:
                    self.parallel_norms[t][b] = float(par_norm) if par_norm is not None else None
                except Exception:
                    self.parallel_norms[t][b] = None
                try:
                    self.perp_norms[t][b] = float(perp_norm) if perp_norm is not None else None
                except Exception:
                    self.perp_norms[t][b] = None

                # compute ratio if possible
                try:
                    pnorm = self.parallel_norms[t][b]
                    qnorm = self.perp_norms[t][b]
                    if pnorm is None or pnorm == 0 or qnorm is None:
                        self.perp_over_par[t][b] = None
                    else:
                        self.perp_over_par[t][b] = float(qnorm / pnorm)
                except Exception:
                    self.perp_over_par[t][b] = None




    def fit_success(self, task, test):

        # Logistic Regression with L2 regularization, compatible with sparse input
        base_model = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000, tol=1e-4)

        

        


        X_train, y_train = task
        X_test, y_test = test

        #print("TOtal success y_train", np.sum(y_train), flush=True)
        #print("Total success y_test", np.sum(y_test), flush=True)

        # Fit model
    
        

        if self.best_time_params is None or (self.train_times+1)%10==0:

            t = time.time()

            param_grid = {
            'C': np.logspace(-2, 2, 5)  # from 0.01 to 1000
            }


            cv_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            refit=True,
            verbose=0,
            n_jobs=1)
            
            cv_search.fit(X_train, y_train)

            self.best_time_params = cv_search.best_params_
            self.time_learner = cv_search.best_estimator_

            print(f'Grid Search done in {time.time()-t}', flush=True)
        
        else:

            base_model = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000, tol=1e-4, C=self.best_time_params['C'])
            base_model.fit(X_train, y_train)
            self.time_learner = base_model


        

        # Predict on train/test
        y_pred_train = self.time_learner.predict_proba(X_train)[:, 1]
        y_pred_test = self.time_learner.predict_proba(X_test)[:, 1]

        # Compute log loss or binary accuracy
        error_train_time = log_loss(y_train, y_pred_train)
        #error_test_time = log_loss(y_test, y_pred_test)

        return error_train_time#, error_test_time

    def success_predict(self, X):
        """
        Predict success probability on new data.
        """
        return self.time_learner.predict_proba(X)[:, 1]


    '''def predict(self, X, verbose=False):
        """
        Predict using all trained models. Output shape: (n_samples, n_targets, n_bootstrap)
        """
        t = time.time()

        n_samples = X.shape[0]
        preds = np.zeros((n_samples, self.n_targets, self.n_bootstrap))

        for t in range(self.n_targets):
            for b, learner in enumerate(self.meta_learners[t]):
                preds[:, t, b] = learner.predict(X)


        # predict time
        try:
            success_prob = self.success_predict(X)
        except Exception:
            success_prob = np.zeros_like(preds[:, 0, 0])

        if verbose:
            print(f"[MODEL] Model & Success prediction {time.time() - t}s", flush=True)

        return preds, success_prob'''
    
    def predict(self, X, obj_indices=None, verbose=False):
        t_start = time.time()
        n_samples = X.shape[0]

        if obj_indices is None:
            obj_indices = list(range(self.n_targets))

        preds = np.zeros((n_samples, len(obj_indices), self.n_bootstrap))

        for i, t in enumerate(obj_indices):
            for b, learner in enumerate(self.meta_learners[t]):
                preds[:, i, b] = learner.predict(X)

        try:
            success_prob = self.success_predict(X)
        except Exception:
            success_prob = np.zeros(n_samples)

        if verbose:
            print(f"[MODEL] Model & Success prediction {time.time() - t_start:.2f}s", flush=True)

        return preds, success_prob
    
    def predict_base(self, X, obj_indices=None, verbose=False):
        """
        Predict using base models only (not meta-learned models).
        Returns base model predictions in the same format as predict().
        """
        from sklearn.linear_model import Ridge
        
        t_start = time.time()
        n_samples = X.shape[0]

        if obj_indices is None:
            obj_indices = list(range(self.n_targets))

        preds = np.zeros((n_samples, len(obj_indices), self.n_bootstrap))

        for i, t in enumerate(obj_indices):
            for b in range(self.n_bootstrap):
                # Use base model coefficients and parameters to make predictions
                if self.base_coeff[t][b] is not None and self.base_params[t][b] is not None:
                    try:
                        # base_params is a list of parameter dicts, we need the first (and typically only) one
                        params_dict = self.base_params[t][b][0] if isinstance(self.base_params[t][b], list) else self.base_params[t][b]
                        
                        # Create Ridge model directly with base coefficients and parameters
                        model_temp = Ridge(**params_dict)
                        model_temp.coef_ = self.base_coeff[t][b]
                        model_temp.intercept_ = 0
                        preds[:, i, b] = model_temp.predict(X)
                    except Exception as e:
                        if verbose:
                            print(f"[WARN] Base prediction failed for target {t}, bootstrap {b}: {e}")
                            print(f"[DEBUG] base_params type: {type(self.base_params[t][b])}, value: {self.base_params[t][b]}")
                            print(f"[DEBUG] base_coeff type: {type(self.base_coeff[t][b])}, shape: {getattr(self.base_coeff[t][b], 'shape', 'no shape')}")
                        preds[:, i, b] = np.nan
                else:
                    # If no base model available, use NaN
                    preds[:, i, b] = np.nan

        try:
            success_prob = self.success_predict(X)
        except Exception:
            success_prob = np.zeros(n_samples)

        if verbose:
            print(f"[MODEL] Base model prediction {time.time() - t_start:.2f}s", flush=True)

        return preds, success_prob
    
    def log_model_metrics(self, save_path, top_k=50, obj_indices=None):
        """
        Write per-target-per-bootstrap diagnostics to CSV with the requested metrics:
        - cosine similarity meta vs same-task base
        - mean absolute change per feature (meta vs base)
        - concentration measure (Gini) on abs(meta coeffs)
        - concentration measure (Gini) on abs(base coeffs)  # NEW
        - norm_par, norm_perp, perp_over_par
        """
        print("Logging model sparsity metrics...", flush=True)
        if save_path is None:
            return
        csv_path = os.path.join(save_path, "model_sparsity_metrics.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not os.path.exists(csv_path)

        if obj_indices is None:
            obj_indices = list(range(self.n_targets))

        def gini_abs(vec):
            v = np.abs(np.asarray(vec)).ravel()
            if v.sum() < 1e-12:
                return 0.0
            sorted_v = np.sort(v)
            n = v.size
            cumw = np.cumsum(sorted_v)
            sum_v = cumw[-1]
            idx = np.arange(1, n + 1)
            return float((2.0 * np.sum(idx * sorted_v) / (n * sum_v)) - (n + 1.0) / n)

        def safe_cos(a, b):
            an = np.linalg.norm(a)
            bn = np.linalg.norm(b)
            if an < 1e-12 or bn < 1e-12:
                return float('nan')
            return float(np.dot(a, b) / (an * bn))
        
        def enp_abs(vec):
            v = np.abs(vec).ravel()
            if v.sum() < 1e-12:
                return 0.0
            p = v / v.sum()
            return float(1.0 / np.sum(p**2))


        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "train_iteration","target","bootstrap",
                    "cosine_meta_base","mean_abs_delta","meta_gini","base_gini",
                    "meta_enp","base_enp",
                    "avg_ess_target",                     
                    "norm_par","norm_perp","perp_over_par","topk_overlap_with_base", "size_feat"
                ])

            for t in obj_indices:
                target_label = self.target_names[t] if self.target_names else f"target_{t}"
                for b in range(self.n_bootstrap):
                    base_coeff = self.base_coeff[t][b]
                    meta_coeff = None
                    try:
                        meta_coeff = self.meta_learners[t][b].coeff if self.meta_learners[t][b] is not None else None
                    except Exception:
                        meta_coeff = None

                    if base_coeff is None or meta_coeff is None:
                        row = [self.train_times, target_label, b] + [None]*8  # Updated to 8 None values
                        writer.writerow(row)
                        continue

                    base_coeff = np.asarray(base_coeff).ravel()
                    meta_coeff = np.asarray(meta_coeff).ravel()

                    cos_mb = safe_cos(meta_coeff, base_coeff)
                    mean_abs_delta = float(np.mean(np.abs(meta_coeff - base_coeff)))
                    meta_g = gini_abs(meta_coeff)
                    base_g = gini_abs(base_coeff)  # NEW: Gini for base coefficients
                    meta_enp_val = enp_abs(meta_coeff)
                    base_enp_val = enp_abs(base_coeff)

                    norm_par = self.parallel_norms[t][b]
                    norm_perp = self.perp_norms[t][b]
                    perp_over_par = self.perp_over_par[t][b]

                    # top-k overlap with same base
                    k = min(top_k, base_coeff.size)
                    top_meta_idx = set(np.argsort(-np.abs(meta_coeff))[:k])
                    top_base_idx = set(np.argsort(-np.abs(base_coeff))[:k])
                    overlap = len(top_meta_idx & top_base_idx) / float(k)

                    W = self.weights[t]                     # shape = (n_samples, n_bootstrap)
                    ess_list = []

                    size_array = len(meta_coeff)
                    for b in range(W.shape[1]):
                        w = W[:, b]
                        denom = np.sum(w * w)
                        if denom < 1e-12:
                            ess = 0.0
                        else:
                            ess = (w.sum() ** 2) / denom
                        ess_list.append(ess)

                    avg_ess_target = float(np.mean(ess_list))

                    row = [
                        self.train_times, target_label, b,
                        cos_mb, mean_abs_delta, meta_g, base_g,
                        meta_enp_val, base_enp_val,
                        avg_ess_target,
                        norm_par, norm_perp, perp_over_par, overlap, size_array
                    ]

                    

                    writer.writerow(row)




    def logrmse_csv(self, tasks, average_only=False, obj_indices=None, save_path=None):
        if save_path is None:
            return

        csv_path = os.path.join(save_path, "mse_models.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        headers = ["train_iteration"]
        row = [self.train_times]

        for t, task in enumerate(tasks):
            
            X_train, y_train, X_test, y_test = task
            n_train = X_train.shape[0]

            base_train_errs = self.base_train_errors[t]

            if t not in obj_indices:
                headers.append(f"{target_label}_base_train_rmse_only")
                row.append(avg_train_base)
                continue

            meta_train_errs = self.meta_train_errors[t]

            avg_train_base = sum(base_train_errs) / len(base_train_errs)
            avg_train_meta = sum(meta_train_errs) / len(meta_train_errs)

            # Compute NSE on training set
            preds_all, _ = self.predict(X_train)  # shape: (n_samples, n_targets, n_bootstrap)
            y_preds_all_t = preds_all[:, t, :]
            y_pred = np.mean(y_preds_all_t, axis=1)
            y_var = np.var(y_preds_all_t, axis=1) + 1e-8
            nse_train = np.mean((y_train - y_pred) ** 2 / y_var)

            # Get name for the current target
            if hasattr(self, 'target_names') and self.target_names is not None:
                target_label = self.target_names[t]
            else:
                target_label = f"target_{t}"

            # Append RMSE train metrics only
            headers.extend([
                f"{target_label}_train_size",
                f"{target_label}_base_train_rmse",
                f"{target_label}_meta_train_rmse",
                f"{target_label}_train_nse",
            ])
            row.extend([n_train, avg_train_base, avg_train_meta, nse_train])

        # Write header if file doesn't exist
        write_header = not os.path.exists(csv_path)

        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(headers)
            writer.writerow(row)

    def evaluate_on_test_set(
    self,
    Y_test,
    model_pred,
    success_pred,
    selected_smiles,
    mask,
    target_names=None,
    save_path=None,
    obj_indices=None
):
        """
        Evaluate OOD prediction loss and calibration from externally supplied predictions.
        Used for newly acquired points.

        Parameters:
            Y_test: List[List[float]] of shape (n_samples, n_total_targets)
            model_pred: ndarray, shape (n_samples, len(obj_indices), n_bootstrap)
            success_pred: ndarray, shape (n_samples,)
            mask: boolean mask over selected_smiles
            obj_indices: list of int indices of predicted targets
        """
        y_array = np.array(Y_test)  # shape: (n_samples, n_total_targets)

        headers = ["train_iteration"]
        row = [self.train_times]
        nse_all = []

        if obj_indices is None:
            obj_indices = list(range(model_pred.shape[1]))

        for i, t in enumerate(obj_indices):
            y_true = y_array[:, t]
            y_preds = model_pred[:, i, :]  # index into model_pred, not full target axis

            y_mean = np.mean(y_preds, axis=1)
            y_var = np.var(y_preds, axis=1) + 1e-8

            mse = np.mean((y_true - y_mean) ** 2)
            nse = np.mean((y_true - y_mean) ** 2 / y_var)
            nse_all.append(nse)

            if mpi_is_master():
                print(f"[MODEL] OOD target {t} - MSE: {mse:.4f}, NSE: {nse:.4f}")

            label = self.target_names[t] if self.target_names else f"target_{t}"
            headers.extend([
                f"{label}_ood_mse",
                f"{label}_ood_nse",
            ])
            row.extend([mse, nse])

        # === Success model evaluation ===
        success_y_true = mask.astype(int)
        success_y_pred = success_pred
        try:
            success_loss = log_loss(success_y_true, success_y_pred)
        except ValueError:
            success_loss = np.nan

        headers.append("success_log_loss")
        row.append(success_loss)

        if mpi_is_master():
            print(f"[MODEL] Success log-loss: {success_loss:.4f}")

        # Save to CSV
        if mpi_is_master() and save_path is not None:
            csv_path = os.path.join(save_path, "ood_models.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(headers)
                writer.writerow(row)

        return nse_all



    def error_correlation_avg(self,  save_path=None):
        """
        Compute correlation between average meta-errors across tasks per bootstrap.
        """
        # Collect matrix of shape (n_targets, n_bootstrap)
        error_matrix = np.array(self.meta_test_errors)  # shape: (n_targets, n_bootstrap)

        # Compute correlation across tasks
        corr_matrix = np.corrcoef(error_matrix)

        # Optionally save
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, f"error_corr/error_corr_{self.train_times}.csv")
            np.savetxt(save_file, corr_matrix, delimiter=",")

        return corr_matrix


