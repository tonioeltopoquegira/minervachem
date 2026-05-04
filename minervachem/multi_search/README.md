# minerva-search

**A framework for meta-learning-enhanced multi-objective molecular**

---

This package provides a modular pipeline for fast and efficient multi-objective molecular search. It combines:

- **Meta-Learning**: Train meta-learners that quickly adapt to new property prediction tasks
- **Active Learning**: Intelligently select molecules to evaluate based on uncertainty-based acquisition functions (EI, PI, UCB). The uncertainty quantification is performed using Bayesian Bootstrapping.
- **Multi-Objective Optimization**: Maintain and improve Pareto fronts across multiple properties simultaneously
- **Parallel Evaluation**: Distributed structure generation and property evaluation via MPI

**Example Applications:**
- QM9 Dataset: Optimize electronic and thermal properties
- SCO Systems: Coordinate complex design for solvation and electronic properties

---

Instructions for a conda environment that enables to run the expertiments:

```bash

conda create -n minervachem_sco python=3.12 \
    fairchem-core \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    scipy \
    crest \
    'xtb>6.5' \
    ase \
    py3Dmol \
    openbabel \
    numba \
    pynauty \
    tqdm \
    mendeleev \
    tblite-python \
    mpi4py \
    lightgbm \
    'setuptools<74' \
    -c conda-forge -y

conda activate minervachem_sco

cd minervachem
pip install -r requirements.txt
pip install -e .

cd minervachem/multi_search/Architector
pip install -e .
cd ../../..

pip install rdkit pymoo lightgbm fairchem-core==2.1.0

python -c "import numpy; print('numpy:', numpy.__version__); import fairchem; import architector; import minervachem; print('All working!')"

```

### Running the QM9 experiment

QM9 evaluations correspond to a look-up in a table
```bash
# Quick test
mpirun -n 4 python demos/multi_search/experiment_qm9.py --seed 2 --n_generations 5

# Full QM9 run
mpirun -n 4 python demos/multi_search/experiment_qm9.py --n_generations 50 --n_samples 10 --seed 0

```
To run Architector experiment it is necessary to have access to an HPC system
```bash
# Full SCO run
mpirun -n 4 python demos/multi_search/experiment_sco.py --n_generations 50 --n_samples 10 --seed 0
```
