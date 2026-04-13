from scipy.sparse import csr_matrix, vstack
import numpy as np
import pandas as pd
import random
import time
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, RidgeCV, SGDRegressor
from sklearn.model_selection import GridSearchCV

from minervachem.regressors import HierarchicalResidualModel


def centroid_by_points(points):
    """
    The function takes in arrays-like structures describing N points in m-dimensional space.
    Outputs the centroid point between the input points.
    :return np.array. Should be a numpy array of shape (1,m).
    """
    points = np.array([np.array(point) for point in points])
    centroid = (np.sum(points, axis=0))/len(points)

    return centroid


def delete_from_csr(mat, row_indices=(), col_indices=()):
    """
    row_indices: list of indices
    col_indices: list of indices
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = list(row_indices)
    cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:, mask]
    else:
        return mat


def cross_validate_parameters(x_train, y_train, sample_weight=None, alpha_range='a1', hierachical=False, featurizer=None):
    """
    Possible solvers: ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    Possible fit_intercept: [True, False]
    Tolerance (tol) can also be included as a varied parameter.
    [!!!] Currently cross-validation is only implemented for alpha search.
    :param x_train:
    :param y_train:
    :return:
    """
    # mine
    # ridge_reg = Ridge()
    # params_ridge = {'alpha': np.logspace(-6, 6, 26),
    #                 "fit_intercept": [False],
    #                 "solver": ['sparse_cg'],
    #                 "tol": [1e-5]}
    # ridge_grid_search = GridSearchCV(ridge_reg, param_grid=params_ridge, n_jobs=-1)
    # ridge_grid_search.fit(x_train, y_train)

    # minerva examples
    # base_model = Ridge(alpha=1e2, solver='sparse_cg', tol=1e-5, fit_intercept=False)
    # if alpha_range == 'a1':
    #     param_grid = {'alpha': 10. ** np.arange(-1, 9, 1 / 3)}
    # elif alpha_range == 'a2':
    #     param_grid = {'alpha': np.logspace(-6, 6, 26)}
    # else:
    #     print("Alpha range must be either 'a1' or 'a2'. Defaulting to 'a1'.")
    #     param_grid = {'alpha': 10. ** np.arange(-1, 9, 1 / 3)}
    # # print("[!!!] Warning: currently we are cross-validating Ridge model for alpha parameter only [!!!]")
    # cv_search = GridSearchCV(estimator=base_model,
    #                          param_grid=param_grid,
    #                          refit=True,
    #                          verbose=0, # used to be 1
    #                          n_jobs=-3)
    # cv_search.fit(x_train, y_train)
    # cv_res = pd.DataFrame.from_dict(cv_search.cv_results_)
    # bp = cv_search.best_params_

    # update 11/13/24 due to the discovery that non-alpha values are not being output !

    base_model = Ridge()
    if alpha_range == 'a1':
        param_grid = {'alpha': 10. ** np.arange(-1, 9, 1 / 3),
                      'fit_intercept': [False],
                      'solver': ['sparse_cg'],
                      'tol': [1e-5]}
        
    elif alpha_range == 'a_parallel':
        param_grid = {'alpha': list(10. ** np.arange(-1, 3.5, 1 / 2)),
                      'fit_intercept': [False],
                      'solver': ['cholesky'],
                      'tol': [1e-5]}
    elif alpha_range == 'a2':
        param_grid = {'alpha': np.logspace(-6, 6, 26),
                      'fit_intercept': [False],
                      'solver': ['sparse_cg'],
                      'tol': [1e-5]}
    elif alpha_range == 'a3':
        param_grid = {'alpha': list(np.logspace(-6, 6, 26))[1::2],
                      'fit_intercept': [False],
                      'solver': ['sparse_cg'],
                      'tol': [1e-5]}
    elif alpha_range == 'a4':
        param_grid = {'alpha': list(10. ** np.arange(-1, 3.5, 1 / 2)),
                      'fit_intercept': [False],
                      'solver': ['sparse_cg'],
                      'tol': [1e-5]}
    else:
        print("Alpha range can be 'a1','a2' or 'a3'. Defaulting to 'a1'.")
        param_grid = {'alpha': 10. ** np.arange(-1, 9, 1 / 3),
                      'fit_intercept': [False],
                      'solver': ['sparse_cg'],
                      'tol': [1e-5]}
    # print("[!!!] Warning: currently we are cross-validating Ridge model for alpha parameter only [!!!]")
    cv_search = GridSearchCV(estimator=base_model,
                             param_grid=param_grid,
                             refit=True,
                             verbose=0, # used to be 1
                             n_jobs=1) # CAREFUL HERE
    

    if hierachical:
        # Still not working HERE
        cv_search = HierarchicalResidualModel(cv_search,verbose=False)
        cv_search.fit(x_train, y_train, levels=featurizer.bit_sizes_)
    cv_search.fit(x_train, y_train, sample_weight=sample_weight)
    cv_res = pd.DataFrame.from_dict(cv_search.cv_results_)
    bp = cv_search.best_params_
    return bp, cv_res


def evaluate_regression_model(model_parameters, model_coefficients, x_test, y_test, print_on=True): # TODO update with MTynes analogous function style
    model_temp = Ridge(**model_parameters)
    model_temp.coef_ = model_coefficients
    model_temp.intercept_ = 0
    pred = model_temp.predict(x_test)

    mae = round(mean_absolute_error(y_test, pred), 2)
    rmse = round(np.sqrt(mean_squared_error(y_test, pred)), 2)
    r2 = round(r2_score(y_test, pred), 4)
    dataset_size = len(y_test)
    if print_on:
        print(f"MAE={mae},   RMSE={rmse},   R_2={r2},   Dataset size: {dataset_size}")
    errors = {'mae': mae,
              'rmse': rmse,
              'r2': r2}
    return errors


def print_evaluations(model_params, model_vectors, input_sets):
    """
    A function that prints out the evaluation results of the Ridge regression models for old tasks.
    :param model_params: list. A list of model parameters.
    :param model_vectors: list. A list of model vectors. Total length should be equal to the number of old tasks.
    :param input_sets: list of lists. Each list contains [xvalues_train, yvalues_train, xvalues_test, yvalues_test]
    for each of the old_tasks.
    :return: None.
    """
    n_ = len(input_sets)
    print(f'__________ON TEST SETS_________')
    for i in range(n_):
        print(f'[+++] TASK {i} {input_sets[i]} [+++]')
        _, _, _ = evaluate_regression_model(model_params[i], model_vectors[i], input_sets[i][2], input_sets[i][3])
    print(f'__________ON TRAINING SETS_________')
    for i in range(n_):
        print(f'[+++] TASK {i} {input_sets[i]} [+++]')
        _, _, _ = evaluate_regression_model(model_params[i], model_vectors[i], input_sets[i][0], input_sets[i][1])


def fit_with_prior(ridgemodel, sample_weight, hierarchical, x_values, y_values, prior=None, reval_final=False, a_range='a1', featurizer=None):
    """Fit a regularized model with a nonzero prior
    Prior should be a vector with a length of X"""
    #print("before assert")
    assert prior is not None, "A prior is not specified"
    #print("after assert")
    if type(x_values) == np.ndarray:
        #print("inside here???")
        new_y = y_values - np.dot(x_values, prior)
    else:
        #print('multiplication started', flush=True)
        #new_y = y_values - np.dot(x_values.toarray(), prior) DENSE!!!!
        new_y = y_values - x_values.dot(prior)
        #print('multiplication done', flush=True)
    if reval_final:
        params, _ = cross_validate_parameters(x_values, new_y, alpha_range=a_range, hierachical=hierarchical, sample_weight=sample_weight, featurizer=featurizer)
        ridgemodel = Ridge(**params)
        #print('reval done', flush=True)
    if hierarchical:
        ridgemodel.fit(x_values, new_y, levels=featurizer.bit_sizes_)
    else:
        #print('prior fit started', flush=True)
        ridgemodel.fit(x_values, new_y, sample_weight=sample_weight)
        #print('prior fit done', flush=True)
    # changes from 12/17/2024
    final_coef = np.copy(ridgemodel.coef_)
    final_alpha = ridgemodel.alpha
    # ridgemodel.coef_ += prior  # modifying underlying model's coefficients
    final_coef += prior
    return final_coef, final_alpha


def cosine_measure(vec1, vec2):
    if np.linalg.norm(vec1)==0 or np.linalg.norm(vec2)==0:
        return 2
    cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cosine


def dist_measure(vec1, vec2):
    dist = np.linalg.norm(vec1 - vec2)
    return dist


def split_data_per_task(all_fp_sparse, task_values, rstate=42, make_log=False, tsize=0.8, subset_fraction=1.0):
    """
    Splits data into training and test sets for a given task, with optional log transformation
    and dataset subsampling before splitting.

    The function is designed for sparse fingerprint data (CSR format) and corresponding property values
    (NumPy array). It first removes NaN values, optionally selects a subset of the data,
    and finally performs a train-test split.

    :param all_fp_sparse: scipy.sparse.csr_matrix
        A sparse matrix representing fingerprints.
    :param task_values: np.ndarray
        An array of corresponding task values.
    :param rstate: int, default=42
        Random seed for reproducibility.
    :param make_log: bool, default=False
        If True, applies log10 transformation to task values.
    :param tsize: float, default=0.8
        Fraction of the dataset to use for training after filtering and subsampling.
    :param subset_fraction: float, default=1.0
        Fraction of the total dataset to retain before train-test split (range: 0.0 < subset_fraction ≤ 1.0).
    :return: list
        A list containing:
        - xvalues_train (scipy.sparse.csr_matrix): Training set fingerprints
        - yvalues_train (np.ndarray): Training set task values
        - xvalues_test (scipy.sparse.csr_matrix): Test set fingerprints
        - yvalues_test (np.ndarray): Test set task values
    """
    # Apply log transformation if specified
    if make_log:
        task_values = np.log10(task_values)

    # Remove NaN values
    inds_to_del = np.argwhere(np.isnan(task_values))
    if np.any(inds_to_del):
        yvalues_task = np.delete(task_values, inds_to_del)
        xvalues_task = delete_from_csr(all_fp_sparse, row_indices=list(np.concatenate(inds_to_del)))
    else:
        yvalues_task = task_values
        xvalues_task = all_fp_sparse

    # Select a random subset of the dataset if subset_fraction is specified
    if 0.0 < subset_fraction < 1.0:
        subset_size = int(len(yvalues_task) * subset_fraction)
        subset_indices, _ = train_test_split(np.arange(0, len(yvalues_task)),
                                             train_size=subset_fraction,
                                             random_state=rstate)
        yvalues_task = yvalues_task[subset_indices]
        xvalues_task = xvalues_task[subset_indices]

    # Perform train-test split
    train_ind, test_ind = train_test_split(np.arange(0, len(yvalues_task)), train_size=tsize, random_state=rstate)
    yvalues_train, yvalues_test = yvalues_task[train_ind], yvalues_task[test_ind]
    xvalues_train, xvalues_test = xvalues_task[train_ind], xvalues_task[test_ind]

    return [xvalues_train, yvalues_train, xvalues_test, yvalues_test]

def run_default_ridge_regressions(input_sets, sample_weight=None, hierarchical=False, featurizer=None, a_range='a1'):
    """
    A function that runs the default Ridge regression model on classical X and Y training sets.
    Cross validation is turned on. [!!!] Currently cross-validating only for alpha.
    :param input_sets: list of lists, where each nested list contains x_values_train, y_values_train,
                                                                      x_values_test, y_values_test
    :param hierarchical: boolean parameter to choose HierarchicalResidualModel over Ridge (default)
    :return: list of coefficient vectors, list of model parameters.
    Number of elements in each list is defined by the number of elements in input_sets
    """
    #print('Starting solving individual regressions')
    model_parameters = []
    coef_vectors = []
    for i in range(len(input_sets)):
        t = time.time()
        best_par, cv_results = cross_validate_parameters(input_sets[i][0], input_sets[i][1], alpha_range=a_range, sample_weight=sample_weight,  hierachical=hierarchical, featurizer=featurizer) #
        #print(f"Ran cross validation in {time.time()-t}s")
        model = Ridge(**best_par)
        model_parameters.append(best_par)
        if hierarchical:
            hmodel = HierarchicalResidualModel(regressor=model, verbose=1)
            hmodel.fit(input_sets[i][0], input_sets[i][1],  sample_weight=sample_weight, levels=featurizer.bit_sizes_) # here missing levels featurizer
            coef_vectors.append(hmodel.coef_)
        else:
            t = time.time()
            model.fit(input_sets[i][0], input_sets[i][1], sample_weight=sample_weight)
            #print(f"Ran fit in {time.time()-t}s")
            coef_vectors.append(model.coef_)
        #print(f'Compltered regression {i+1} (out of {len(input_sets)})')

    return coef_vectors, model_parameters


def save_dict_to_file(dictionary, file_location, file_prefix='data', extension='txt'):
    """
    Saves a dictionary to a text file with the filename containing the current date.
    If the file already exists, a consecutive number is added to the filename.

    Args:
        dictionary (dict): The dictionary to be saved.
        file_location (str): The directory where the file will be saved.
        file_prefix (str, optional): The prefix for the filename. Defaults to 'data'.
        extension (str, optional): The file extension. Takes in 'txt' or 'pkl', defaults to 'txt'.
    """
    today = datetime.date.today().strftime('%m%d%Y')
    filename = f"{file_prefix}_{today}.{extension}"
    file_path = os.path.join(file_location, filename)

    if os.path.exists(file_path):
        # If the file already exists, add a consecutive number to the filename
        i = 1
        while os.path.exists(file_path):
            filename = f"{file_prefix}_{today}_{i}.{extension}"
            file_path = os.path.join(file_location, filename)
            i += 1

    if extension == 'txt':
        with open(file_path, 'w') as file:
            for key, value in dictionary.items():
                file.write(f"{key}: {value}\n")
    elif extension == 'pkl':
        temp_df = pd.DataFrame([dictionary])
        temp_df.to_pickle(file_path)

    print(f"Dictionary saved to {file_path}")


def find_orthonormal_basis_set(set_of_vectors):
    """
    :param set_of_vectors: np.array, each element is a np.array itself, all elements must be the same length
    :return: orthobasis
    """
    # check if vectors are linearly independent
    if np.linalg.det(set_of_vectors) == 0.0:
        print("Provided set of vectors is not linearly independent")
        return set_of_vectors

    Q, R = np.linalg.qr(set_of_vectors)
    return np.array(Q)


def gram_schmidt(set_of_vectors, row_vecs=True, thres=1e-12):
    """

    :param set_of_vectors: np.array, each element is a np.array itself, all elements must be the same length
    :return: orthobasis
    """
    if not row_vecs:
        set_of_vectors = set_of_vectors.T

    # check for non linearity
    # TODO

    def normalize(vector):
        if np.linalg.norm(vector) < thres:
            return vector*0
        return vector / np.linalg.norm(vector)

    def project(vector, vector_on):
        return np.dot(vector, vector_on)*vector_on

    def compute_residual(vector, projection):
        return normalize(vector - projection)

    e1 = normalize(set_of_vectors[0, :])
    basis = [e1]

    for i in range(1, set_of_vectors.shape[0]):
        sum_projections = np.sum(np.array([project(set_of_vectors[i], basis[k]) for k in range(len(basis))]), axis=0)
        e_i = compute_residual(set_of_vectors[i], sum_projections)
        basis.append(e_i)
    return basis

