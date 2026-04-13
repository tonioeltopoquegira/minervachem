import numpy as np
import pandas as pd
import os
from ...optimization.utils import ParetoFront

# Get the directory where this module is located
_module_dir = os.path.dirname(os.path.abspath(__file__))
_static_dir = os.path.join(os.path.dirname(os.path.dirname(_module_dir)), 'static')


def compute_pareto():

    df = pd.read_csv(os.path.join(_static_dir, 'qm9_processed.csv'))
    # Step 1: Filter valid rows (no missing values)
    property_cols = ['E_at', 'zpve', 'e_gap', 'C_v']
    df_valid = df.dropna(subset=property_cols)

    # Step 2: Extract Y matrix and SMILES
    Y_all = df_valid[property_cols].copy()
    Y_all['E_at'] = -Y_all['E_at']  # Reverse sign for minimization (if lower is better)
    Y_all_matrix = Y_all.to_numpy()
    smiles_all = df_valid['smiles'].to_numpy()

    # Step 3: Compute Pareto front
    def is_dominated(y, others):
        return np.any(np.all(others <= y, axis=1) & np.any(others < y, axis=1))

    pareto_mask = np.array([not is_dominated(Y_all_matrix[i], np.delete(Y_all_matrix, i, axis=0))
                            for i in range(Y_all_matrix.shape[0])])

    Y_pareto = Y_all_matrix[pareto_mask]
    smiles_pareto = smiles_all[pareto_mask]


    pareto_df = pd.DataFrame(Y_pareto, columns=['-E_at', 'zpve', 'e_gap', 'C_v'])
    pareto_df['smiles'] = smiles_pareto

    pareto_df.to_csv(os.path.join(_module_dir, 'qm9_pareto_set.csv'), index=False)



def get_pareto():

    actual_df = pd.read_csv(os.path.join(_module_dir, 'qm9_pareto_set.csv'))
    Y_actual = actual_df[['-E_at', 'zpve', 'e_gap', 'C_v']].to_numpy() #
    smiles_actual = actual_df['smiles'].tolist()

    # Generate random Xs (just placeholders)
    X_random = [np.zeros(1)] * len(Y_actual)  # 1D dummy vector


    actual_pf = ParetoFront()
    actual_pf.initialize_pareto(X_random, Y_actual, smiles_actual)

    return actual_pf
