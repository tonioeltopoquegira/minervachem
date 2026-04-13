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
            if debug_time:
                print(f'[RANK rank - {lig['smiles']}] Building Complex', flush=True)
            t_cal = time.time()
            arch_out = build_complex(architector_input)
            if debug_time:
                print(f'[RANK rank - {lig['smiles']}] Building Complex in {time.time()-t_cal}', flush=True)

            if arch_out:
                key = list(arch_out.keys())[0]
                arch_mol = convert_io_molecule(arch_out[key]['mol2string'])
                output_dict['architector_uff_mol2'] = arch_out[key]['mol2string']
            else:
                good = False
                error += "Failed to build complex. "
        except Exception as e:
            good = False
            error += str(e)

        if len(arch_out) == 0:
            good = False
            error += 'Architector produced no output.'
        else:
            mol = convert_io_molecule(arch_mol)
            mol.uhf = 0
            if debug_time:
                print(f'[RANK rank - {lig['smiles']}]  Relaxing LS Complex', flush=True)
            t_cal = time.time()
            low_spin = CalcExecutor(mol, method='omol', relax=True, fmax=0.05)
            if debug_time:
                print(f'[RANK rank - {lig['smiles']}] LS complex in {time.time()-t_cal}', flush=True)

            

            if low_spin.successful:
                output_dict['low_spin_mol2_omol'] = low_spin.mol.write_mol2('ls',writestring=True)
            else:
                error += 'low spin omol failed'
            if debug_time:
                print(f'[RANK rank - {lig['smiles']}] Relaxing HS Complex', flush=True)
            
            mol = convert_io_molecule(arch_mol)
            t_cal = time.time()
            high_spin = CalcExecutor(mol, method='omol', relax=True, fmax=0.05)
            if debug_time:
                print(f'[RANK rank - {lig['smiles']}] HS complex in {time.time()-t_cal}', flush=True)

            
            if high_spin.successful:
                output_dict['high_spin_mol2_omol'] = high_spin.mol.write_mol2('hs',writestring=True)
            else:
                error += 'high spin omol failed'
            if low_spin.successful and high_spin.successful:
                output_dict['sco_kcal'] = np.abs((high_spin.energy - low_spin.energy) / (units.kcal/units.mol)) # absolut value minimizaion !

    if good and len(task_dict.get('solvents',[])) > 0:
        for solv in task_dict['solvents']:
            if debug:
                #print('Evaluating {} solvent'.format(solv), flush=True)
                pass
            
            mol = CalcExecutor(arch_mol,
                method='GFN2-xTB', xtb_solvent=solv,
                relax=False, store_results=True)

            if mol.successful:
                output_dict['{}_gsolv_eV'.format(solv)] = mol.results['gsolv_eV']
                output_dict['{}_hl_gap_eV'.format(solv)] = mol.results['hl_gap_eV']
                output_dict['{}_dipole'.format(solv)] = - np.linalg.norm(mol.results['dipole']).item() # Here minus for maximization!
            else:
                error += '{} solvent failed XTB'.format(solv)
    if debug and error=='':
        print(f' [RANK rank - {lig['smiles']}] Success in {time.time()-time_init}', flush=True)
    elif debug and error !='':
        print(f' [RANK rank - {lig['smiles']}] Failed in {time.time()-time_init}', flush=True)
    
    output_dict['error'] = error
    output_dict['running_time'] = time.time()-time_init
    
    return output_dict