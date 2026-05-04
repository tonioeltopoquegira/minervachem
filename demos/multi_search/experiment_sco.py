"""
Reproducibility experiment: SCO (coordination-complex) multi-objective active learning.

This script reproduces the SCO search experiments from the paper. It runs
the full Meta-Active-Multi-objective pipeline (`NDchemicalsearch` from
``minervachem.multi_search.pipeline_sco``) over the 13 ligand-database
properties, optimizing the four targets ``[0, 7, 8, 9]`` =
``sco_kcal, octanol_gsolv_eV, octanol_hl_gap_eV, octanol_dipole`` by default.

Run it under MPI with a master rank that orchestrates the search and
worker ranks that build/evaluate coordination complexes via
Architector + xtb / CREST:

    mpirun -n 4 python demos/multi_search/experiment_sco.py \\
        --name meta_4_ensemble_wupdate --seed 0 --start_from_scratch

Outputs land in::

    figs_logs/{name}_{seed}_<obj_target_names>/

which is the directory layout the post-processing notebooks
(``post_process_sco.ipynb``, ``post_process_sco_multiple.ipynb``) expect.


Reproduce the four variants compared in the paper by varying ``--name``
and the appropriate flags:

* ``--name meta_4_ensemble_wupdate``                              (meta + adaptive weights)
* ``--name base_4_ensemble_wupdate     --base_only``              (single-task base + adaptive)
* ``--name meta_4_ensemble_woupdate    --no_update_weights``      (meta, no adaptive update)
* ``--name random_4_ensemble           --random_sample``          (meta + random selection)

For each variant, run with ``--seed 0 1 2 3 ...`` to build the ensemble.
"""

import argparse
import sys

import os as _os
_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from minervachem.multi_search.pipeline_sco import NDchemicalsearch
from minervachem.multi_search.utils_mpi import (
    mpi_is_master,
    mpi_worker_loop,
    stop_all_workers,
)



TRAIN_NAMES = [
    "sco_kcal",
    "water_gsolv_eV",   "water_hl_gap_eV",   "water_dipole",
    "acetone_gsolv_eV", "acetone_hl_gap_eV", "acetone_dipole",
    "octanol_gsolv_eV", "octanol_hl_gap_eV", "octanol_dipole",
    "hexane_gsolv_eV",  "hexane_hl_gap_eV",  "hexane_dipole",
]

# Default objective targets: sco_kcal, octanol_gsolv_eV,
# octanol_hl_gap_eV, octanol_dipole (indices into TRAIN_NAMES).
DEFAULT_OBJ_TARGETS = [0, 7, 8, 9]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce the SCO architector search experiment from the paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Identifiers --
    parser.add_argument(
        "--name",
        type=str,
        default="meta_4_ensemble_wupdate",
        help="Experiment name. Output folder is figs_logs/{name}_{seed}_<obj names>/. "
        "Use one of: meta_4_ensemble_wupdate, base_4_ensemble_wupdate, "
        "meta_4_ensemble_woupdate, random_4_ensemble (paper variants).",
    )
    parser.add_argument("--seed", type=int, default=0)

    # -- Targets --
    parser.add_argument(
        "--obj_targets",
        type=int,
        nargs="+",
        default=DEFAULT_OBJ_TARGETS,
        help="Indices into TRAIN_NAMES that the search optimizes.",
    )

    # -- Resume / restart --
    parser.add_argument(
        "--start_from_scratch",
        action="store_true",
        help="If set, re-sample the initial set and re-query. "
        "Otherwise the pipeline tries to resume from query_runs.csv.",
    )

    # -- Generation / training --
    parser.add_argument("--initial_set_size", type=int, default=500)
    parser.add_argument("--n_generations", type=int, default=30)
    parser.add_argument("--n_bootstrap", type=int, default=100)

    # -- Sampling / acquisition --
    parser.add_argument("--sample_size", type=int, default=40000)
    parser.add_argument("--retain", type=float, default=0.025)
    parser.add_argument(
        "--acquisition",
        type=str,
        default="pi",
        choices=["pi", "ei", "ucb"],
    )
    parser.add_argument(
        "--functional",
        type=int,
        default=2,
        help="Functionalization mode used by the parallel sampler.",
    )
    parser.add_argument("--cluster", type=int, default=None)
    parser.add_argument(
        "--base_only",
        action="store_true",
        help="Use only base learners (no meta-learning).",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Replace EGO selection with random selection.",
    )

    # -- Calibration / weighting --
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
        help="Disable the adaptive bootstrap-bagging weight update.",
    )
    parser.set_defaults(update_weights=True)
    parser.add_argument("--history_len", type=int, default=-1)
    parser.add_argument(
        "--lowbound",
        type=float,
        default=0.0001,
        help="Minimum value lower-bound for the alpha floor.",
    )

    # -- Featurizer --
    parser.add_argument("--max_len", type=int, default=5)

    # -- Misc --
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if mpi_is_master():
        run_name = f"{args.name}_{args.seed}"
        obj_names = [TRAIN_NAMES[i] for i in args.obj_targets]
        print("=" * 70, flush=True)
        print(f"SCO search experiment: {run_name}", flush=True)
        print("=" * 70, flush=True)
        print(
            f"  obj_targets={args.obj_targets} -> {obj_names}\n"
            f"  initial_set_size={args.initial_set_size} n_generations={args.n_generations}\n"
            f"  n_bootstrap={args.n_bootstrap}  sample_size={args.sample_size}\n"
            f"  retain={args.retain}  acquisition={args.acquisition}\n"
            f"  base_only={args.base_only}     random_sample={args.random_sample}\n"
            f"  update_weights={args.update_weights}  history_len={args.history_len}\n"
            f"  start_from_scratch={args.start_from_scratch}",
            flush=True,
        )

        try:
            NDchemicalsearch(
                name=run_name,
                seed=args.seed,
                train_names=TRAIN_NAMES,
                obj_targets=args.obj_targets,
                start_from_scratch=args.start_from_scratch,
                # Generation & training
                initial_set_size=args.initial_set_size,
                n_generations=args.n_generations,
                n_bootstrap=args.n_bootstrap,
                # Surrogates
                base_only=args.base_only,
                # Sampling
                sample_size=args.sample_size,
                retain=args.retain,
                acquisition=args.acquisition,
                random_sample=args.random_sample,
                functional=args.functional,
                cluster=args.cluster,
                # Calibration / alpha
                update_weights=args.update_weights,
                start_weights=args.start_weights,
                history_len=args.history_len,
                min_val_lowbound=args.lowbound,
                # Featurizer
                max_len=args.max_len,
                # Misc
                plot=args.plot,
                verbose=args.verbose,
                # Per-target objective limits used by visualization
                lim=[
                    (0, 50),
                    (-500, 50),
                    (-5.0, 50.0),
                    (-50.0, 5.0),
                ],
            )
        finally:
            stop_all_workers()

        # Match the original pipeline_sco.py main: drop a flag once done.
        with open("workflow_done.flag", "w") as f:
            f.write("done\n")
    else:
        mpi_worker_loop()


if __name__ == "__main__":
    main()
