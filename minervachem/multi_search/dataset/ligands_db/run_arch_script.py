# run_arch_script.py
import sys
import json
from arch_funct import arch

if __name__ == "__main__":
    inp = json.load(sys.stdin)
    ligand = inp["ligand"]
    seed = inp["seed"]
    out = arch(ligand, {'sco': True, 'solvents': ['water', 'acetone', 'octanol', 'hexane']}, seed=seed)
    
    json.dump(out, sys.stdout)
