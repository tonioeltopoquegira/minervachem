import os
import warnings
import time
script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['OMOL_MODEL_PATH'] = os.path.join(script_dir, 'esen_sm_conserving_all.pt')
from architector import build_complex, convert_io_molecule, CalcExecutor
from ase import units
import numpy as np


def arch(lig, task_dict={
    'sco':True,
    'solvents':['water','acetone','octanol','hexane'],
    },
    debug=False,
    debug_time=False,
    seed = 42):
    

    time_init = time.time()
    good = True
    error = ''
    output_dict = dict()
    # print(f"[ARCH] Starting for {lig.get('smiles', 'unknown')}", flush=True)
    if task_dict.get('sco', False):
        arch_out = dict()
        nlig = int(6/len(lig['coordList']))
        architector_input = {
            'core':{'metal':'Fe','coreType':'octahedral'},
            'ligands':[lig]*nlig,
            'parameters':{
                'metal_ox':2,
                'metal_spin':4,
                'assemble_method':'UFF',
                'full_method':'UFF',
                'seed':seed},
        }
        arch_mol = None
        try:
            # print(f"[ARCH] Building complex with {len(lig['coordList'])} ligands", flush=True)
            # print(f"[ARCH] Input: metal=Fe, nlig={nlig}, coordList={lig['coordList']}", flush=True)
            t_cal = time.time()
            arch_out = build_complex(architector_input)
            # print(f"[ARCH] build_complex completed in {time.time()-t_cal:.2f}s", flush=True)
            # print(f"[ARCH] arch_out type: {type(arch_out)}, len: {len(arch_out) if arch_out else 0}", flush=True)

            if arch_out and len(arch_out) > 0:
                # print(f"[ARCH] arch_out keys: {list(arch_out.keys())}", flush=True)
                key = list(arch_out.keys())[0]
                arch_mol = convert_io_molecule(arch_out[key]['mol2string'])
                output_dict['architector_uff_mol2'] = arch_out[key]['mol2string']
                # print(f"[ARCH] Successfully converted structure", flush=True)
            else:
                good = False
                error += "Failed to build complex - empty output. "
                # print(f"[ARCH] arch_out is empty or None!", flush=True)
                # print(f"[ARCH] arch_out value: {arch_out}", flush=True)
        except Exception as e:
            good = False
            error += str(e)
            # print(f"[ARCH] Exception during complex building: {type(e).__name__}: {e}", flush=True)
            # import traceback
            # print(f"[ARCH] Traceback:\n{traceback.format_exc()}", flush=True)

        if len(arch_out) == 0:
            good = False
            error += 'Architector produced no output.'
        else:
            # print(f"[ARCH] Starting LS/HS calculations", flush=True)
            mol = convert_io_molecule(arch_mol)
            mol.uhf = 0
            # print(f"[ARCH] Running low-spin OMOL calculation...", flush=True)
            t_cal = time.time()
            low_spin = CalcExecutor(mol, method='omol', relax=True, fmax=0.05)
            # print(f"[ARCH] LS completed in {time.time()-t_cal:.2f}s, successful={low_spin.successful}", flush=True)

            if low_spin.successful:
                output_dict['low_spin_mol2_omol'] = low_spin.mol.write_mol2('ls',writestring=True)
                # print(f"[ARCH] LS energy: {low_spin.energy:.4f}", flush=True)
            else:
                error += 'low spin omol failed'
                # print(f"[ARCH] LS failed", flush=True)

            # print(f"[ARCH] Running high-spin OMOL calculation...", flush=True)
            mol = convert_io_molecule(arch_mol)
            t_cal = time.time()
            high_spin = CalcExecutor(mol, method='omol', relax=True, fmax=0.05)
            # print(f"[ARCH] HS completed in {time.time()-t_cal:.2f}s, successful={high_spin.successful}", flush=True)

            if high_spin.successful:
                output_dict['high_spin_mol2_omol'] = high_spin.mol.write_mol2('hs',writestring=True)
                # print(f"[ARCH] HS energy: {high_spin.energy:.4f}", flush=True)
            else:
                error += 'high spin omol failed'
                # print(f"[ARCH] HS failed", flush=True)
            if low_spin.successful and high_spin.successful:
                sco = np.abs((high_spin.energy - low_spin.energy) / (units.kcal/units.mol))
                output_dict['sco_kcal'] = sco
                # print(f"[ARCH] SCO energy: {sco:.4f} kcal/mol", flush=True)

    if good and len(task_dict.get('solvents',[])) > 0:
        # print(f"[ARCH] Starting solvation calculations for {len(task_dict.get('solvents',[]))} solvents", flush=True)
        for solv in task_dict['solvents']:
            # print(f"[ARCH] Running XTB for {solv}...", flush=True)

            mol = CalcExecutor(arch_mol,
                method='GFN2-xTB', xtb_solvent=solv,
                relax=False, store_results=True)

            if mol.successful:
                output_dict['{}_gsolv_eV'.format(solv)] = mol.results['gsolv_eV']
                output_dict['{}_hl_gap_eV'.format(solv)] = mol.results['hl_gap_eV']
                output_dict['{}_dipole'.format(solv)] = - np.linalg.norm(mol.results['dipole']).item() # Here minus for maximization!
                # print(f"[ARCH] {solv} success: gsolv={mol.results['gsolv_eV']:.4f}, gap={mol.results['hl_gap_eV']:.4f}", flush=True)
            else:
                error += '{} solvent failed XTB'.format(solv)
                # print(f"[ARCH] {solv} failed", flush=True)
    if debug and error=='':
        print(f' [RANK rank - {lig['smiles']}] Success in {time.time()-time_init}', flush=True)
    elif debug and error !='':
        print(f' [RANK rank - {lig['smiles']}] Failed in {time.time()-time_init}', flush=True)
    
    output_dict['error'] = error
    output_dict['running_time'] = time.time()-time_init
    
    return output_dict