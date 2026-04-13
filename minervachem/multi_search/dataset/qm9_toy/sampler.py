import numpy as np
import pandas as pd
import os

# Get the directory where this module is located
_module_dir = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(_module_dir, 'qm9_processed.csv'))

# currently not checking for sampling with or without replacement (Are the proposed values all different and new? 
# Might need to do it if our molecule generation is not vary enough)


def sample(n_samples: int, pareto=None, seed: int = 42) -> list:
    """
    Returns list of SMILES strings, either from random sampling or from ParetoFrontier.
    """
    rng = np.random.default_rng(seed)

    if pareto is not None:
        return pareto.functionalize(n_samples)
    else:
        indices = rng.choice(len(df), size=n_samples, replace=False)
        return df.iloc[indices]['smiles'].tolist()

