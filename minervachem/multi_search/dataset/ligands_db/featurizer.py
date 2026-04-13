
from scipy.sparse import load_npz
from rdkit import Chem
import pandas as pd
import os
import numpy as np
import time


# Get the directory where this module is located
_module_dir = os.path.dirname(os.path.abspath(__file__))
_dataset_dir = os.path.join(os.path.dirname(_module_dir), 'ligands_db')

# For moment being assume I already have full featurizer...
df = pd.read_pickle(os.path.join(_dataset_dir, 'initial_ligands.pkl'))

X_featurized = load_npz(os.path.join(_dataset_dir, 'X_lig_feat.npz'))

def featurize(candidates_list):
    #mol = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in smiles]
    #init_x = featurizer.transform(mol)
    # Map from SMILES to index for fast lookup
    smiles_to_idx = {smi: idx for idx, smi in enumerate(df['lig_smiles'])}

    smiles_list = [cand['new_smiles'] for cand in candidates_list]

    indices = []
    for smi in smiles_list:
        if smi in smiles_to_idx:
            indices.append(smiles_to_idx[smi])
        else:
            raise ValueError(f"SMILES '{smi}' not found in DataFrame.")

    return X_featurized[indices]


def fit_featurizer(folder_path, featurizer, verbose):
    """
    Fit a featurizer using all successful entries in query_runs.csv.
    Returns: X_featurized matrix and a DataFrame with smiles/mol alignment.
    """
    t= time.time()
    if verbose:
        print('[FINGER] Fitting Fingerprint...', flush=True)
    file_path = os.path.join(folder_path, "query_runs.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"query_runs.csv not found in {folder_path}")

    df = pd.read_csv(file_path)
    smiles_col = 'new_smiles'
    if 'new_smiles' not in df.columns:
        smiles_col = 'smiles'


    # Only successful entries
    df = df[df['success'] == True].copy()

    def safe_mol(smi):
        mol = Chem.MolFromSmiles(smi)
        return Chem.AddHs(mol) if mol else None

    df['mol'] = df[smiles_col].map(safe_mol)
    df = df[df['mol'].notna()].copy()

    if df.empty:
        raise ValueError("No valid molecules found for featurization.")
    
    

    X = featurizer.fit_transform(df['mol'])
    if verbose:
        print(f'[FINGER] Fitted in {time.time()-t}s', flush=True)




def transform_featurizer(samples, featurizer, verbose=False):

    t = time.time()
    

    smiles_col = 'new_smiles' if 'new_smiles' in samples[0] else 'smiles'
    smiles = [s[smiles_col] for s in samples]

    def safe_mol(s):
        mol = Chem.MolFromSmiles(s)
        return Chem.AddHs(mol) if mol else None

    mols = []
    failed_idxs = []

    for idx, smi in enumerate(smiles):
        mol = safe_mol(smi)
        if mol is None:
            failed_idxs.append(idx)
            if verbose:
                print(f'[FINGER] Excluding not-parsable molecule {smi}')
        else:
            mols.append(mol)

    if mols:
        X = featurizer.transform(mols)
    else:
        X = None  # or raise a warning if all failed

    valid_mask = np.ones(len(samples), dtype=bool)
    valid_mask[failed_idxs] = False

    if verbose:
        print(f'[FINGER] Transformed all in {time.time()-t}s with {len(valid_mask) - np.sum(valid_mask)} not-parsable', flush=True)

    return X, valid_mask



def transform(sampled_smiles, functional=False, featurizer=None, prop=None, verbose=False):

    if functional:

        X, failed_idx = transform_featurizer(sampled_smiles, featurizer, verbose)
        sampled_smiles = [s for s, f in zip(sampled_smiles, failed_idx) if f]

        if prop is not None:
            prop = [p for p, f in zip(prop, failed_idx) if f]
            return X, sampled_smiles, prop
        else:
            return X, sampled_smiles

    else:
        return featurize(sampled_smiles)  

