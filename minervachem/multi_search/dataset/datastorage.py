from rdkit import Chem
from typing import List, Dict, Optional, Tuple
from scipy.sparse import vstack, csr_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from .ligands_db.featurizer import featurize
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import pairwise_distances
import scipy.sparse as sp

class Dataset:
    
    def __init__(self, seed: int, standardize: bool = False, featurizer=None):
        self.seed = seed
        self.standardize = standardize
        self.rng = np.random.default_rng(seed)

        self.featurizer = featurizer

        # Core storage
        self.smiles_list: List[str] = []     # All SMILES strings
        self.mols: List[Chem.Mol] = []       # Corresponding RDKit molecules
        self.X: List[csr_matrix] = []        # Featurized vectors (csr_matrix rows)
        self.Y: List[List[float]] = []       # Objective property values per molecule
        self.prop_names: List[str] = []      # Optional: track names of targets
        
        # Training data by target
        self.data: Dict[int, Dict[str, Tuple[csr_matrix, np.ndarray]]] = {}
        self.last_added_indices: Dict[int, Dict[str, List[int]]] = {}
        self.scalers: Dict[int, StandardScaler] = {}

        self.train_smiles: Dict[int, List[str]] = {}


        self.fitted: bool = False


    def _safe_mol(self, smi: str) -> Optional[Chem.Mol]:
        mol = Chem.MolFromSmiles(smi)
        return Chem.AddHs(mol) if mol else None


    def add_data(self, smiles: List[str], props: List[List[float]]):
        """Add molecules + properties to the canonical dataset."""
        for s, p in zip(smiles, props):
            mol = self._safe_mol(s)
            if mol is not None:
                self.smiles_list.append(s)
                self.mols.append(mol)
                self.Y.append(p)


    def fit_featurizer(self, sampled_cand: List[str] = None, verbose=False):
        """
        Fit the featurizer on either:
        - the provided list of SMILES (if given), or
        - the internal stored `self.mols` (default behavior).
        """

        if verbose:
            print(f'[FINGER] Fitting fingerprint...', flush=True)
        
        if not self.featurizer:
            raise ValueError("Featurizer not set.")
        
        t = time.time()

        if sampled_cand is not None:

            smiles = [s['new_smiles'] if 'new_smiles' in s.keys() else s['smiles'] for s in sampled_cand]
            mols = [self._safe_mol(s) for s in smiles]
            mols = [m for m in mols if m is not None]
            if not mols:
                raise ValueError("No valid molecules from provided SMILES.")
            X = self.featurizer.fit_transform(mols)
        else:
            if not self.mols and not self.smiles_list:
                raise ValueError("No molecules available to fit.")
            
        
            mols = [self._safe_mol(s) for s in self.smiles_list]
            self.mols = [m for m in mols if m is not None]

            if not self.mols:
                raise ValueError("No valid molecules from stored SMILES.")
            if verbose:
                print(f'Fitting with {len(self.mols)} molecules',flush=True )
            X = self.featurizer.fit_transform(self.mols)


        self.X = list(X)
        self.fitted = True
        if verbose:
            print(f'[FINGER] Fitted the fingerprint in {time.time()-t}s', flush=True)
        
        self.rebuild_training_data(verbose=verbose)



    def transform_smiles(self, sampled_cand: List[str], verbose: bool = False, fit=False, return_unseen=False):
        """
        Transform new molecules using the current fitted featurizer.
        If return_unseen=True, also return X_unseen (novel fragments mask).
        Returns:
            - X: np.ndarray, shape (n_mols, n_features)
            - sampled_cand: list of valid candidate dicts
            - valid_mask: np.ndarray, shape (n_mols,) boolean
            - X_unseen: np.ndarray, shape (n_mols, n_features) [if return_unseen]
        """
        str_sm = 'new_smiles' if 'new_smiles' in sampled_cand[0].keys() else 'smiles'
        if verbose:
            print('[FINGER] Transforming fingerprint...', flush=True)
        smiles = [s[str_sm] for s in sampled_cand]
        t = time.time()
        if not self.fitted:
            raise RuntimeError("Featurizer has not been fit.")
        mols = []
        valid_smiles = []
        valid_mask = np.ones(len(smiles), dtype=bool)
        for i, s in enumerate(smiles):
            mol = self._safe_mol(s)
            if mol is None:
                valid_mask[i] = False
                if verbose:
                    print(f"[FINGER] Skipping unparsable: {s}", flush=True)
            else:
                mols.append(mol)
                valid_smiles.append(s)
        if not mols:
            raise ValueError("No valid molecules for transformation.")
        if return_unseen:
            X, X_unseen = self.featurizer.transform(mols, return_unseen=True)
            sampled_cand = [s for s in sampled_cand if s[str_sm] in valid_smiles]
            if verbose:
                print(f'[FINGER] Transformed {len(valid_smiles)}/{len(smiles)} SMILES in {time.time()-t:.2f}s')
            return X, X_unseen, sampled_cand, valid_mask
        else:
            X = self.featurizer.transform(mols)
            sampled_cand = [s for s in sampled_cand if s[str_sm] in valid_smiles]
            if verbose:
                print(f'[FINGER] Transformed {len(valid_smiles)}/{len(smiles)} SMILES in {time.time()-t:.2f}s')
            return X, sampled_cand, valid_mask


    def get_last_batch(self, target_idx: int) -> Tuple[csr_matrix, np.ndarray]:
        """Returns the last added (X, y) batch from the training set for a specific target."""
        if target_idx not in self.last_added_indices or target_idx not in self.data:
            return None, None

        train_idx = self.last_added_indices[target_idx]['train']
        X_train_all, y_train_all = self.data[target_idx]['train']

        return (
            X_train_all[-len(train_idx):],
            y_train_all[-len(train_idx):]
        )

    def prepare_batch(
            self,
            smiles: List[str],
            all_properties: List[Dict],
            target_indices: List[int],
            verbose: bool = False,
            fit=True,
            test=False
        ) -> Tuple[csr_matrix, List[List[float]], List[str]]:
        """
        Filters and featurizes a batch of molecules using internal featurizer,
        excluding molecules that can't be parsed or have incomplete properties.
        """
        if not self.featurizer:
            raise ValueError("Featurizer not initialized.")

        mols = []
        smiles_valid = []
        property_valid = []

        unparsable_count = 0
        incomplete_count = 0

        for s, props in zip(smiles, all_properties):
            smi = s['new_smiles'] if 'new_smiles' in s else s['smiles']
            mol = self._safe_mol(smi)

            if mol is None:
                unparsable_count += 1
                if verbose:
                    print(f"[FILTER] Dropping unparsable molecule: {smi}")
                continue

            if not props or "result" not in props:
                incomplete_count += 1
                continue

            results = props["result"]
            if not all(
                t_idx < len(results) and results[t_idx] is not None and not np.isnan(results[t_idx])
                for t_idx in target_indices
            ):
                incomplete_count += 1
                continue

            # Keep
            mols.append(mol)
            smiles_valid.append(s)
            property_valid.append([results[t_idx] for t_idx in target_indices])

        if not mols:
            raise ValueError("No valid molecules after featurization + property masking.")
        
        if not self.fitted and fit and not test:
            self.fit_featurizer(sampled_cand=smiles_valid, verbose=verbose)
        
        if test:

            if verbose:
                print('[DATASET] Fallback to featurization through lookup', flush=True)

            X = featurize(smiles_valid)
            valid_smiles, valid_mask = smiles_valid, np.ones(len(smiles_valid))

        else:
            X, valid_smiles, valid_mask = self.transform_smiles(sampled_cand=smiles_valid, verbose=True)

        property_valid = [p for p, v in zip(property_valid, valid_mask) if v ]
        
        if verbose:
            print(f"[DATASET] {len(valid_smiles)}/{len(smiles)} molecules retained.", flush=True)
            print(f"[DATASET] {unparsable_count} unparsable, {incomplete_count} missing properties.", flush=True)

        return X, property_valid, valid_smiles



    def get_tasks(self) -> List[Tuple[int, csr_matrix, np.ndarray]]:
        """Return all target-specific (target_idx, X, y) triplets for training."""
        tasks = []
        for target_idx, splits in self.data.items():
            X_tr, y_tr = splits['train']
            tasks.append((target_idx, X_tr, y_tr))
        return tasks


    def get_valid_objective_vectors(self, target_indices: List[int]) -> List[List[float]]:
        """Return valid objective vectors across all targets (samples with no NaNs)."""
        if not all(tgt in self.data for tgt in target_indices):
            raise ValueError("Not all targets have data in the dataset.")

        all_targets_y = []
        for tgt in target_indices:
            y_train = self.data[tgt]['train'][1]
            all_targets_y.append(y_train)

        N = len(all_targets_y[0])
        if not all(len(y) == N for y in all_targets_y):
            raise ValueError("Mismatch in sample counts across targets.")

        valid_Y = []
        for i in range(N):
            y_i = [y[i] for y in all_targets_y]
            if all(v is not None and not np.isnan(v) for v in y_i):
                valid_Y.append(y_i)

        return valid_Y


    def add_multi_target_points(
        self,
        X: csr_matrix,
        Y_list: List[List[float]],
        target_indices: List[int],
        smiles: List[str]
    ):
        """Add a batch of X, Y rows and smiles/mols into the dataset."""
        Y_array = np.array(Y_list)  # shape (n_samples, n_targets)
        indices = np.arange(X.shape[0])
        self.rng.shuffle(indices)
        X_train = X[indices]

        # === Add SMILES and mols only ONCE ===
        shuffled_smiles = [smiles[i]['new_smiles'] for i in indices]
        self.smiles_list.extend(shuffled_smiles)
        self.mols.extend([self._safe_mol(s) for s in shuffled_smiles])

        for i, t_idx in enumerate(target_indices):
            y_train = Y_array[indices, i]

            # Store SMILES used for this target
            if t_idx not in self.train_smiles:
                self.train_smiles[t_idx] = [shuffled_smiles[j] for j in range(len(shuffled_smiles))]
            else:
                self.train_smiles[t_idx].extend([shuffled_smiles[j] for j in range(len(shuffled_smiles))])

            if self.standardize:
                scaler = self.scalers.get(t_idx, StandardScaler())
                y_train = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
                self.scalers[t_idx] = scaler

            if t_idx not in self.data:
                self.data[t_idx] = {
                    'train': (X_train, y_train)
                }
            else:
                Xtr_old, ytr_old = self.data[t_idx]['train']
                self.data[t_idx]['train'] = (
                    vstack([Xtr_old, X_train]),
                    np.concatenate([ytr_old, y_train])
                )

            self.last_added_indices[t_idx] = {'train': indices.tolist()}


    def rebuild_training_data(self, verbose=False):
        """Re-transform all stored SMILES using the current featurizer and rebuild training data."""
        if not self.fitted:
            raise RuntimeError("Featurizer must be fitted before rebuilding training data.")
        if verbose:
            print("[DATASET] Rebuilding training data from stored SMILES...", flush=True)

        for t_idx in self.train_smiles:
            smiles_list = self.train_smiles[t_idx]
            mols = [self._safe_mol(s) for s in smiles_list]
            mols = [m for m in mols if m is not None]

            print(f"[DEBUG] Rebuilding target {t_idx} with {len(mols)} mols")


            if not mols:
                raise ValueError(f"No valid molecules for target {t_idx}")

            X = self.featurizer.transform(mols)
            y = self.data[t_idx]['train'][1]

            self.data[t_idx]['train'] = (X, y)

            print(f"[DEBUG] Target {t_idx} rebuilt: X shape = {X.shape}, y shape = {y.shape}")


        if verbose:
            print("[DATASET] Training features updated to match featurizer.", flush=True)

    def get_unseen_fragments(self, mols):
        """
        Returns the unseen fragment matrix (X_unseen) for the given molecules,
        using the internal featurizer fitted on training data.
        Args:
            mols: list of RDKit Mol objects or SMILES strings
        Returns:
            X_unseen: np.ndarray, shape (n_mols, n_features)
        """
        # If input is SMILES, convert to RDKit Mol
        if isinstance(mols[0], str):
            from rdkit import Chem
            mols = [Chem.MolFromSmiles(s) for s in mols]
        # Use internal featurizer, must be fitted
        X_bits, X_unseen = self.featurizer.transform(mols, return_unseen=True)
        return X_unseen

    def filter_valid_molecule_dicts(self, dicts: List[Dict], target_indices: List[int], verbose: bool = False) -> List[Dict]:
        """
        Filters molecule dicts for valid SMILES, properties, and valence, using _safe_mol (adds hydrogens).
        This matches the logic used in training and should be used for manual test set filtering.
        """
        filtered = []
        for td in dicts:
            results = td["result"]
            smi = td.get('new_smiles', td.get('smiles'))
            mol = self._safe_mol(smi)
            if (mol is not None) and all((v is not None) and (not np.isnan(v)) for v in [results[t] for t in target_indices]):
                try:
                    Chem.SanitizeMol(mol)
                    filtered.append(td)
                except Exception:
                    if verbose:
                        print(f"[FILTER] Valence error for: {smi}", flush=True)
                    continue
            else:
                if verbose:
                    print(f"[FILTER] Dropping: {smi}", flush=True)
        return filtered




from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
import numpy as np

def cluster_centroids(sampled_smiles, X, n_clusters=None, ei=None, original_indices=None):
    if n_clusters is None:
        n_clusters = max(1, int(0.01 * X.shape[0]))

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=42)
    cluster_assignments = kmeans.fit_predict(X)

    representative_indices_local = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_assignments == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue

        if ei is not None:
            best_local_idx = cluster_indices[np.argmax(ei[cluster_indices])]
        else:
            dists = pairwise_distances(X[cluster_indices], kmeans.cluster_centers_[cluster_id].reshape(1, -1))
            best_local_idx = cluster_indices[np.argmin(dists)]

        representative_indices_local.append(best_local_idx)

    # Map back to original indices
    if original_indices is not None:
        representative_indices_global = [original_indices[i] for i in representative_indices_local]
    else:
        representative_indices_global = representative_indices_local

    centroids_smiles = [sampled_smiles[i] for i in representative_indices_local]
    centroids_ei = np.array([ei[i] for i in representative_indices_local]) if ei is not None else None


    return centroids_smiles, centroids_ei, np.array(representative_indices_global)
