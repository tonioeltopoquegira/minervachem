from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer




def sascore(candidate):
    m  = Chem.MolFromSmiles(candidate['smiles'])
    return sascorer.calculateScore(m)
    
def logp(candidate):
    m  = Chem.MolFromSmiles(candidate['smiles'])
    return MolLogP(m)
