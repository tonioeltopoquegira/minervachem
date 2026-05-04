"""
Reproducibility experiment: QM9 multi-objective active learning.

This script reproduces the QM9 search experiments from the paper. It runs
the full Meta-Active-Multi-objective pipeline (`NDchemicalsearch` from
``minervachem.multi_search.pipeline``) with the four QM9 properties
``E_at, zvpe, e_gap, C_v``.

Run it under MPI with a master rank that orchestrates the search and
worker ranks that evaluate query functions in parallel:

    mpirun -n 4 python demos/multi_search/experiment_qm9.py --name meta_ego --seed 0

The script writes its outputs to::

    figs_logs/{name}_{seed}_E_at_zvpe_e_gap_C_v/

which is the directory layout the post-processing notebooks
(``post_process_qm9.ipynb``, ``post_process_qm9_multiple.ipynb``) expect.

Paper variants
--------------
The notebooks compare three variants per seed; reproduce them by varying
``--name`` together with the appropriate flags:

* ``--name meta_ego``                       (default: meta-learner + EGO acquisition)
* ``--name base_ego --base_only``           (single-task base learner only)
* ``--name meta_random --random_sample``    (meta-learner with random selection)

To produce an ensemble across seeds, run the same command for
``--seed 0 1 2 3 ...``.
"""

import argparse
import sys
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')


import os as _os
_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from minervachem.multi_search.pipeline import NDchemicalsearch
from minervachem.multi_search.utils_mpi import (
    mpi_is_master,
    mpi_worker_loop,
    stop_all_workers,
)


TARGET_NAMES = ["E_at", "zvpe", "e_gap", "C_v"]
TARGETS = [0, 1, 2, 3]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce the QM9 search experiment from the paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Identifiers --
    parser.add_argument(
        "--name",
        type=str,
        default="meta_ego",
        help="Experiment name. The output folder is "
        "figs_logs/{name}_{seed}_E_at_zvpe_e_gap_C_v/. "
        "Use 'meta_ego', 'base_ego', or 'meta_random' to match the paper.",
    )
    parser.add_argument("--seed", type=int, default=0)

    # -- Generation / training --
    parser.add_argument("--n_generations", type=int, default=100)
    parser.add_argument("--n_bootstrap", type=int, default=200)

    # -- Sampling / acquisition --
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--retain", type=int, default=100)
    parser.add_argument(
        "--acquisition",
        type=str,
        default="pi",
        choices=["pi", "ei", "ucb"],
        help="Acquisition function used by EGO.",
    )
    parser.add_argument(
        "--base_only",
        action="store_true",
        help="Use only base learners (no meta-learning). For 'base_ego' variant.",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Replace EGO selection with random selection. For 'meta_random' variant.",
    )
    parser.add_argument("--cluster_size", type=int, default=5)

    parser.add_argument(
        "--start_weights",
        type=float,
        nargs="+",
        default=[0.03, 0.03, 0.03, 0.03],
    )
    parser.add_argument(
        "--no_update_weights",
        dest="update_weights",
        action="store_false",
        help="Disable adaptive update of bootstrap-bagging weights.",
    )
    parser.set_defaults(update_weights=True)
    parser.add_argument(
        "--history_len",
        type=int,
        default=-1,
        help="Window length for calibration history (-1 = use all history).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if mpi_is_master():
        run_name = f"{args.name}_{args.seed}"
        print("=" * 70, flush=True)
        print(f"QM9 search experiment: {run_name}", flush=True)
        print("=" * 70, flush=True)
        print(
            f"  generations={args.n_generations}  bootstrap={args.n_bootstrap}\n"
            f"  sample_size={args.sample_size}    retain={args.retain}\n"
            f"  acquisition={args.acquisition}    base_only={args.base_only}\n"
            f"  random_sample={args.random_sample} cluster_size={args.cluster_size}\n"
            f"  update_weights={args.update_weights} history_len={args.history_len}\n"
            f"  start_weights={args.start_weights}",
            flush=True,
        )

        try:
            NDchemicalsearch(
                name=run_name,
                target_names=TARGET_NAMES,
                targets=TARGETS,
                n_generations=args.n_generations,
                n_bootstrap=args.n_bootstrap,
                acquisition=args.acquisition,
                base_only=args.base_only,
                random_sample=args.random_sample,
                cluster_size=args.cluster_size,
                sample_size=args.sample_size,
                retain=args.retain,
                seed=args.seed,
                history_len=args.history_len,
                start_weights=args.start_weights,
                update_weights=args.update_weights,
            )
        finally:
            # Always release the worker ranks even if the master crashes,
            # otherwise they hang forever in mpi_worker_loop().
            stop_all_workers()
    else:
        mpi_worker_loop()


if __name__ == "__main__":
    main()
