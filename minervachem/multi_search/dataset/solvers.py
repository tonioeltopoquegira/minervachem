import sys
import numpy as np
import time
from ..utils_mpi import register_mpi_function
from .qm9_toy.functions import E_at, zvpe, e_gap, C_v
import csv
import subprocess
import json
import os
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import shutil


def sascore(candidate):
    m  = Chem.MolFromSmiles(candidate['new_smiles'])
    return sascorer.calculateScore(m)
    
def logp(candidate):
    m  = Chem.MolFromSmiles(candidate['new_smiles'])
    return MolLogP(m)

ARCH_FUNCT = [
    'sco_kcal', 
    'water_gsolv_eV', 'water_hl_gap_eV', 'water_dipole', 
    'acetone_gsolv_eV', 'acetone_hl_gap_eV', 'acetone_dipole',
    'octanol_gsolv_eV', 'octanol_hl_gap_eV', 'octanol_dipole',
    'hexane_gsolv_eV', 'hexane_hl_gap_eV', 'hexane_dipole'
]


FUNCTIONS = {
    'SA': sascore,
    'logp': logp,
}

@register_mpi_function("query")
def mpi_query(args):

    smiles, kwargs = args
    safe_kwargs = {k: v for k, v in kwargs.items() if k != "_initial_seed"}
    return query_lig(smiles, **safe_kwargs)

@register_mpi_function("query_offline")
def query_offline(candidate, target=-1):
    """
    Offline query function for pre-defined properties.
    Supports both plain SMILES strings and dicts with 'new_smiles'.
    """
    functions = [E_at, zvpe, e_gap, C_v]

    # handle input type
    if isinstance(candidate, dict):
        smiles = candidate.get("new_smiles", candidate.get("smiles"))
    else:
        smiles = candidate

    if target == -1:
        target = np.arange(len(functions))
    if isinstance(target, int):
        target = [target]

    results = []
    for t in target:
        try:
            results.append(functions[t](smiles))
        except Exception:
            results.append(None)

    return {
        "result": results,
        "success": all(r is not None for r in results),
        "smiles": smiles,
        "new_smiles": smiles,  
    }



def query(candidate, target=-1):
    
    functions = [E_at,  zvpe, e_gap, C_v]

    #print(candidate)

    if target == -1:
        target = np.arange(len(functions))

    target = [target] if isinstance(target, int) else target

    res = []
    for t in target:
        res.append(functions[t](candidate))
        
    return {"result": res, "success": True, "smiles": candidate}


def query_lig(candidate, target=-1, seed=42, timeout=2000):

    if target == -1:
        target_from_arch = ARCH_FUNCT
    else:
        target_from_arch = target

    # Compute custom targets: those not in target_from_arch
    custom_targets = [t for t in target if t not in ARCH_FUNCT]
    for t in custom_targets:
        if t not in FUNCTIONS:
            raise ValueError(f"[QUERY ERROR] Custom target '{t}' has no associated function defined.")

    # Collect corresponding functions
    res = [FUNCTIONS[t] for t in custom_targets]

    # CHECK 
    #candidate['functionalizations'] = ''

    # Prepare input data
    input_data = {'ligand': candidate, 'seed': seed}
    print(f'Running... {input_data}', flush=True)

    try:
        t = time.time()
        # Get absolute path to the evaluation script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, 'ligands_db', 'run_arch_script.py')

        # Use the current Python executable (from conda environment)
        python_exe = sys.executable

        proc = subprocess.run(
            [python_exe, script_path],
            input=json.dumps(input_data),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(script_path)
        )

        # # Debug logging 
        # print(f"\n[RUN] Finished {input_data['ligand']['smiles']} in {time.time() - t:.2f}s", flush=True)
        # if proc.stderr.strip():
        #     print(f"[STDERR]:\n{proc.stderr.strip()}", flush=True)

        # Validate non-empty stdout
        if not proc.stdout.strip():
            print(f"[ERROR] Empty stdout from subprocess. Python: {python_exe}, Script: {script_path}", flush=True)
            if proc.returncode != 0:
                print(f"[ERROR] Subprocess returned code {proc.returncode}", flush=True)
            raise ValueError("Empty stdout received from subprocess.")

        # Try to decode output
        try:
            out_dict = json.loads(proc.stdout)
        except json.JSONDecodeError as je:
            # print("JSON decode error:", je, flush=True)
            # print(f"Raw stdout: {proc.stdout[:200]}", flush=True)
            out_dict = {
                "result": [None] * len(target_from_arch),
                "success": False,
                "error": f"JSONDecodeError: {str(je)}",
                "running_time": time.time() - t,
                "_timeout": False
            }
        else:
            out_dict['_timeout'] = False
            out_dict.setdefault('success', True)  # default to True if not set
            out_dict.setdefault('running_time', time.time() - t)

            # Print captured debug output
            if '_debug_stdout' in out_dict and out_dict['_debug_stdout'].strip():
                print(f"[ARCH_DEBUG]\n{out_dict['_debug_stdout']}", flush=True)
            if '_debug_stderr' in out_dict and out_dict['_debug_stderr'].strip():
                print(f"[ARCH_DEBUG_STDERR]\n{out_dict['_debug_stderr']}", flush=True)

            # Log if subprocess returned error
            if not out_dict.get('success', True):
                print(f"[ARCH_ERROR] {out_dict.get('error', 'Unknown error')}", flush=True)

    except subprocess.TimeoutExpired as e:
        #print(f"TIMEOUT for {input_data['ligand']['smiles']} after {timeout}s", flush=True)
        #print("STDOUT (partial):", e.stdout, flush=True)
        #print("STDERR (partial):", e.stderr, flush=True)

        out_dict = {
            "result": [None] * len(target_from_arch),
            "success": False,
            "error": "Timeout exceeded",
            "running_time": timeout,
            "_timeout": True
        }

    except Exception as e:
        print(f"Unexpected error for {input_data['ligand']['smiles']}: {e}", flush=True)
        out_dict = {
            'error': str(e),
            'success': False,
            'running_time': -1.0,
            '_timeout': False
        }

    for tar in target_from_arch:
        try:
            res.append(out_dict[tar])
        except Exception:
            res.append(None)

    success = out_dict.get('error', '') == ''

    print(f"Success: {success}, {candidate} in {time.time()-t}s w/ {res}", flush=True)

    return {
        "result": res,
        "success": success,
        "seed": seed,
        "smiles": candidate['smiles'],
        "coordList": candidate['coordList'],
        "functionalizations": candidate.get('functionalizations', []),
        "running_time": out_dict.get('running_time', None),
        "error": out_dict.get('error', ''),
        "_timeout": out_dict['_timeout'],
        'high_spin_mol2_omol' : out_dict.get('high_spin_mol2_omol', ''),
        'low_spin_mol2_omol' : out_dict.get('low_spin_mol2_omol', ''),
        'architector_uff_mol2' : out_dict.get('architector_uff_mol2', '')
    }

if __name__ == '__main__':
    import os
    print("Speed and Success Test\n")

    # Define test dataset
    test_data = [
    ('Cc1ccccc1[O-]', [9, 6, 3, 20, 12, 0]),
    ('C[Si](C)(C)C(P1CC1c1ccccc1)[Si](C)(C)C', [5]),
    ('CCCCN1C=CC(=N1)c1cccc([c-]1)C1=NN(CCCC)C=C1', [16, 8, 14]),
    ('C1CCC(CC1)P(Oc1cccc2cccnc12)C1CCCCC1', [6, 16]),
    ('CC1CCC(C)P1c1cc(cc(c1[O-])C(C)(C)C)C(C)(C)C', [6, 13]),
    ('Brc1ccc([O-])c(c1)C1=NNC=C1', [9, 5]),
    ('Nc1[nH+]c(CN2CCN(CCN(CCN(CC2)Cc2cccc(N)n2)Cc2cccc(N)n2)Cc2cccc(N)n2)ccc1', [14, 5, 8, 11, 24, 40]),
    ('CC(C)P(=N)(C(C)C)C(C)C', [4]),
    ('O1N=C(c2ccccn2)c2ccccc12', [8, 1]),
    ('Cc1ccc(CCCCC2=C(CCCCc3ccc(C)cc3)C(=Pc3c(cc(cc3C(C)(C)C)C(C)(C)C)C(C)(C)C)C2=Pc2c(cc(cc2C(C)(C)C)C(C)(C)C)C(C)(C)C)cc1', [43, 23]),
    ('Oc1ccccc1C(=O)NN=Cc1ccc2ccc(C=NNC(=O)c3ccccc3O)c([O-])c2c1O', [23, 20, 32]),
    ('Cc1cc(c2ccc(F)cc2)c(N=C2c3cccc4cccc(C2=Nc2c(cc(C)cc2c2ccc(F)cc2)c34)c(c1)c1ccc(F)cc1', [12, 24]),
    ('Cc1cc(c2cc(F)cc(F)c2)c(N=C2c3cccc4cccc(C2=Nc2c(cc(C)cc2c2cc(F)cc(F)c2)c2cc(F)cc(F)c2)c34)c(c1)c1cc(F)cc(F)c1', [25, 13]),
    ('Cc1cc(c(N=C2c3cccc4cccc(C2=Nc2c(cc(C)cc2c2c(F)c(F)c(F)c(F)c2F)c2c(F)c(F)c(F)c(F)c2F)c34)c(c1)c1c(F)c(F)c(F)c(F)c1F)c1c(F)c(F)c(F)c(F)c1F', [17, 5]),
    ('CC(N[Si](C)(C)C([Si](C)(C)NC(C)c1ccccc1)[Si](C)(C)NC(C)c1ccccc1)c1ccccc1', [2, 10, 22]),
    ('C[Si](C)(NC1CCc2ccccc12)C([Si](C)(C)NC1CCc2ccccc12)[Si](C)(C)NC1CCc2ccccc12', [3, 17, 30]),
    ('CN(C)N=C1C2CCC3(CS(=O)(=O)N=C13)C2(C)C', [13, 3]),
    ('C=Cc1ccccc1P(c1ccccc1)c1ccccc1P(c1ccccc1)c1ccccc1', [21, 8]),
    ('CC(C)=Cc1ccccc1P(c1ccccc1)c1ccccc1P(c1ccccc1)c1ccccc1', [23, 10]),
    ('OC(=O)c1ccc(cc1)P(Cc1cccc(CP(c2ccc(cc2)C(O)=O)c2ccc(cc2)C(O)=O)[c-]1)c1ccc(cc1)C(O)=O', [9, 36, 17]),
    ('FC(F)(F)c1cc(cc(c1)C(F)(F)F)C#N', [15]),
    ('CC(C)(C)NC(=O)C(=[C-]c1ccccc1)c1ccccc1', [6, 8]),
    ('O=C(N1CCCC1)C(=[C-]c1ccccc1)c1ccccc1', [0, 8]),
    ('CC(C)N1C=CN([CH-]1)c1ccccc1N1N=C(C)C=C1C', [15, 7]),
    ('CCSC(=O)C(CC)=[C-]CC', [4, 8]),
    ('CC(C)=NN=C([S-])SCc1ccccc1', [3, 6]),
    ('CC1=NC(=[NH+]N1)C', [2]),
    ('CC(=NN=C([S-])SCc1ccccc1)C1=CC=CS1', [2, 5]),
    ('Cc1cc(C=Nc2ccccc2)c([O-])c(c1)C12CC3CC(CC(C3)C1)C2', [13, 5]),
    ('Brc1ccc([O-])c(C=NCC[O-])c1', [5, 8, 11]),
    ('CN(C)C([S-])=NN=Cc1cccc(C)n1', [6, 1, 4]),
    ('[CH-]=C(OP(Oc1ccccc1)Oc1ccccc1)c1ccccc1', [3, 0]),
    ('CC(C)(C)c1cc(COc2ccncc2)cc(c1)C(C)(C)C', [12]),
    ('O=C1N(c2ccncc2)C(=O)c2ccc(c3cccc1c23)N(=O)=O', [6]),
    ('O=C1N(c2cccnc2)C(=O)c2ccc(c3cccc1c23)N(=O)=O', [7]),
    ('CC(C)c1cc(nn1Bn1nc(cc1C(C)C)C(C)(C)C)C(C)(C)C', [6, 10]),
    ('COc1ccc([O-])c(C=NNC(N)=S)c1', [13, 6, 9]),
    ('OC(=O)c1ccnc(c1)c1cc(cc(n1)c1cc(ccn1)C(O)=O)C(=O)[O-]', [14, 20, 6]),
    ('NC(=S)NN=Cc1cc(Br)ccc1[O-]', [2, 13, 4]),
    ('NC(=S)NN=Cc1cc(ccc1[O-])N(=O)=O', [2, 12, 4]),
    ('ClC1=C([N-]N=C1c1ccccn1)c1ccccn1', [17, 3]),
    ('BrC1=C([N-]N=C1c1ccccn1)c1ccccn1', [3, 17])
]



    # File path
    csv_filename = "test_speed_query.csv"

    # Write header if file does not exist
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["smiles", "coordList", "result", "time_taken"])

    # Loop with immediate appends
    for idx, (smiles, coords) in enumerate(test_data):
        candidate = {'smiles': smiles, 'coordList': coords}

        print(f"Testing molecule {idx + 1}/{len(test_data)}")
        t0 = time.time()

        try:
            output = query_lig(candidate)
        except Exception as e:
            output = f"Error: {str(e)}"

        elapsed = round(time.time() - t0, 3)

        print(f"Result: {output}")
        print(f"Time: {elapsed:.3f}s\n")

        # Append row to CSV
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([smiles, coords, output, elapsed])

    print("All results appended to", csv_filename)