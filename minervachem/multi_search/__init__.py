# Check for MPI availability at import time
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    import warnings
    warnings.warn(
        "mpi4py not installed. Multi-search functionality requires MPI. "
        "To use multi-search features, install minervachem with MPI support: "
        "pip install 'minervachem[mpi]' or install mpi4py separately with your system MPI.",
        ImportWarning,
        stacklevel=2
    )

# Import plot_utils directly (doesn't require MPI)
from . import plot_utils

# Core pipeline functions - will be lazily imported when needed
def __getattr__(name):
    """Lazy loading of multi_search components with helpful error messages."""
    if not HAS_MPI:
        raise ImportError(
            f"Cannot import '{name}' from minervachem.multi_search: "
            "mpi4py/MPI not available. Install with MPI support to use multi-search features.\n"
            "Options:\n"
            "  1. pip install 'minervachem[mpi]'\n"
            "  2. Install mpi4py separately: pip install mpi4py (requires system MPI)\n"
            "  3. Use conda: conda install mpi4py"
        )
    
    if name == "NDchemicalsearch":
        from .pipeline import NDchemicalsearch
        return NDchemicalsearch
    elif name == "NDchemicalsearch_SCO":
        from .pipeline_sco import NDchemicalsearch_SCO
        return NDchemicalsearch_SCO
    elif name == "Pipeline":
        from .pipeline import NDchemicalsearch as Pipeline
        return Pipeline
    elif name == "PipelineSCO":
        from .pipeline_sco import NDchemicalsearch_SCO as PipelineSCO
        return PipelineSCO
    elif name == "utils":
        from . import utils
        return utils
    elif name == "utils_mpi":
        from . import utils_mpi
        return utils_mpi
    elif name == "compat":
        from . import compat
        return compat
    elif name == "models_performance":
        from . import models_performance
        return models_performance
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "NDchemicalsearch",
    "NDchemicalsearch_SCO",
    "Pipeline",
    "PipelineSCO",
    "utils",
    "utils_mpi",
    "compat",
    "models_performance",
    "plot_utils",
    "HAS_MPI",
]
