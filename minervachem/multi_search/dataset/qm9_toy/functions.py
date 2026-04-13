import pandas as pd
import os

# Get the directory where this module is located
_module_dir = os.path.dirname(os.path.abspath(__file__))
_data_file = os.path.join(_module_dir, 'qm9_processed.csv')

df = pd.read_csv(_data_file)

# Compute means and stds for normalization
norm_stats = {
    'E_at': {
        'mean': df['E_at'].mean(),
        'std': df['E_at'].std()
    },
    'zpve': {
        'mean': df['zpve'].mean(),
        'std': df['zpve'].std()
    },
    'C_v': {
        'mean': df['C_v'].mean(),
        'std': df['C_v'].std()
    },
    'e_gap': {
        'mean': df['e_gap'].mean(),
        'std': df['e_gap'].std()
    }
}

normalization = False

def E_at(smiles, normalize=normalization):
    match = df[df['smiles'] == smiles]
    if not match.empty:
        value = -match.iloc[0]['E_at']  # Still apply your negation
        if normalize:
            value = (value - norm_stats['E_at']['mean']) / norm_stats['E_at']['std']
        return value
    else:
        return None


def zvpe(smiles, normalize=normalization):
    match = df[df['smiles'] == smiles]
    if not match.empty:
        value = match.iloc[0]['zpve']
        if normalize:
            value = (value - norm_stats['zpve']['mean']) / norm_stats['zpve']['std']
        return value
    else:
        return None


def C_v(smiles, normalize=normalization):
    match = df[df['smiles'] == smiles]
    if not match.empty:
        value = match.iloc[0]['C_v']
        if normalize:
            value = (value - norm_stats['C_v']['mean']) / norm_stats['C_v']['std']
        return value
    else:
        return None


def e_gap(smiles, normalize=normalization):
    match = df[df['smiles'] == smiles]
    if not match.empty:
        value = match.iloc[0]['e_gap']
        if normalize:
            value = (value - norm_stats['e_gap']['mean']) / norm_stats['e_gap']['std']
        return value
    else:
        return None
