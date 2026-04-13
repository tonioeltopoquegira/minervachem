import numpy as np
import pandas as pd
from typing import List, Dict, Any, Iterable

def _to_iterable_list(x: Any) -> List[Any]:
    """Safely convert list-like things (np.ndarray, pd.Series) to Python list."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, (np.ndarray, pd.Series)):
        return list(x)
    # If it's a single scalar (string, dict...), wrap it
    return [x]

def smiles_to_dicts(smiles: Iterable[str], key: str = "new_smiles") -> List[Dict[str, str]]:
    """Wrap plain SMILES (iterable) into list of dicts with specified key."""
    out = []
    for s in _to_iterable_list(smiles):
        if isinstance(s, bytes):
            s = s.decode()
        out.append({key: s})
    return out

def ensure_smiles_dicts(smiles_like, prefer_key="new_smiles") -> List[Dict[str, Any]]:
    """
    Normalize input into a list of dicts with both 'smiles' and 'new_smiles' keys.
    - Strings become {"smiles": s, "new_smiles": s}.
    - Dicts get patched to always include both.
    - Other objects are stringified.
    """
    items = _to_iterable_list(smiles_like)
    if len(items) == 0:
        return []

    normalized = []
    for item in items:
        # bytes -> decode
        if isinstance(item, bytes):
            item = item.decode()

        if isinstance(item, str):
            normalized.append({"smiles": item, "new_smiles": item})
            continue

        if isinstance(item, dict):
            d = dict(item)  # shallow copy
            val = d.get(prefer_key) or d.get("smiles")
            if val is None:
                val = str(d)
            d["new_smiles"] = val
            d["smiles"] = val
            normalized.append(d)
            continue

        if hasattr(item, "get"):
            try:
                d = dict(item)
                val = d.get(prefer_key) or d.get("smiles") or str(d)
                d["new_smiles"] = val
                d["smiles"] = val
                normalized.append(d)
                continue
            except Exception:
                pass

        try:
            s = str(item)
            normalized.append({"smiles": s, "new_smiles": s})
        except Exception:
            continue

    return normalized


def mask_valid_adapter(mask_valid_fn, target_indices, all_properties, init_x, init_smiles, dataset=None):
    """
    Adapter that tries legacy mask_valid first; if it errors, falls back to dataset.prepare_batch.
    Returns: X_valid, Y_valid, smiles_valid_dicts, mask
    """
    import numpy as _np

    try:
        X_valid, Y_valid, smiles_valid, mask = mask_valid_fn(target_indices, all_properties, init_x, init_smiles)
    except Exception:
        if dataset is None:
            raise
        # ensure init_smiles passed as dicts if they are raw strings
        init_input = ensure_smiles_dicts(init_smiles, prefer_key="smiles")
        X_valid, Y_valid, smiles_valid = dataset.prepare_batch(
            smiles=init_input, all_properties=all_properties, target_indices=target_indices, verbose=False, fit=True
        )
        # construct mask relative to original init_input
        valid_new = [d.get('new_smiles', d.get('smiles')) for d in smiles_valid]
        orig_strings = [d.get('smiles') if isinstance(d, dict) and 'smiles' in d else (d if isinstance(d, str) else str(d))
                        for d in init_input]
        mask = _np.array([s in valid_new for s in orig_strings])

    # normalize smiles_valid into dicts with 'new_smiles'
    smiles_valid = ensure_smiles_dicts(smiles_valid, prefer_key="new_smiles")
    return X_valid, Y_valid, smiles_valid, mask
