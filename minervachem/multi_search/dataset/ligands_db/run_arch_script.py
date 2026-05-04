# run_arch_script.py
import sys
import json
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr

from arch_funct import arch

if __name__ == "__main__":
    try:
        inp = json.load(sys.stdin)
        ligand = inp["ligand"]
        seed = inp["seed"]

        # Capture stdout and stderr from arch()
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            out = arch(ligand, {'sco': True, 'solvents': ['water', 'acetone', 'octanol', 'hexane']}, seed=seed)

        if out is None:
            out = {
                "error": "arch() returned None",
                "success": False,
                "smiles": ligand.get("smiles", ""),
            }

        # Add captured debug output
        out['_debug_stdout'] = stdout_capture.getvalue()
        out['_debug_stderr'] = stderr_capture.getvalue()

        json.dump(out, sys.stdout)
    except Exception as e:
        error_output = {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "success": False,
            "smiles": inp.get("ligand", {}).get("smiles", "unknown") if 'inp' in locals() else "unknown",
        }
        json.dump(error_output, sys.stdout)
        sys.exit(1)
