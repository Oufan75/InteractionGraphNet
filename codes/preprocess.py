import numpy as np
from rdkit import Chem
import pickle

def generate_complex(receptor, ligand_file, finalpath, cutoff=8):
    # get the pocket of protein
    protein = Chem.MolFromPDBFile(receptor)
    
    pocket_file = pocketpath + '/' + ligand_file.split('/')[-1].replace('.sdf', '_pocket.pdb')
    ligand = Chem.MolFromMolFile(ligand_file)
    pocket = Chem.MolFromPDBFile(pocket_file)

    # write the ligand and pocket to pickle object
    # ComplexFileName = ''.join(['./ign_input/', ligand_file.split('/')[-1].strip('.sdf')])
    ComplexFileName = ''.join([finalpath + '/', ligand_file.split('/')[-1][:-4]])
    with open(ComplexFileName, 'wb') as ComplexFile:
        pickle.dump([ligand, pocket], ComplexFile)

