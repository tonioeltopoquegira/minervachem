import pandas as pd
from scipy.sparse import load_npz
import os

# Get the directory where this module is located
_module_dir = os.path.dirname(os.path.abspath(__file__))

# For moment being assume I already have full featurizer...
df = pd.read_csv(os.path.join(_module_dir, 'qm9_processed.csv'))

X_featurized = load_npz(os.path.join(_module_dir, 'X_matrix.npz'))

def featurize(smiles_list):
    #mol = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in smiles]
    #init_x = featurizer.transform(mol)
    # Map from SMILES to index for fast lookup
    smiles_to_idx = {smi: idx for idx, smi in enumerate(df['smiles'])}

    indices = []
    for smi in smiles_list:
        if smi in smiles_to_idx:
            indices.append(smiles_to_idx[smi])
        else:
            raise ValueError(f"SMILES '{smi}' not found in DataFrame.")

    return X_featurized[indices]