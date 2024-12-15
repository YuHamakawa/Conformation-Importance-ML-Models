import os

import pandas as pd
from rdkit import Chem


def get_unique_elements(file_path, smiles_col='smiles'):
    """
    Function to read SMILES strings from a specified CSV file and obtain a set of unique elements.
    unique_elements = {'Si', 'C', 'P', 'F', 'N', 'Cl', 'I', 'S', 'Br', 'O'}
    The goal is to extract compounds containing only HCNOFS.
    
    Parameters:
    - file_path (str): Path to the CSV file.
    - smiles_col (str): Name of the column containing SMILES strings (default is 'smiles').
    
    Returns:
    - set: Set of unique elements.
    """
    df = pd.read_csv(file_path, usecols=[smiles_col])
    unique_elements = set()

    def get_elements_from_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Return a set of elements contained in the molecule
            return {atom.GetSymbol() for atom in mol.GetAtoms()}
        else:
            return set()

    # Add elements contained in each SMILES to the set to keep them unique
    for smiles in df[smiles_col]:
        unique_elements.update(get_elements_from_smiles(smiles))

    return unique_elements


def filter_elements_csv(data_path, save_dir):
    '''Extracts specific columns and removes rows based on conditions:
       - csid 10674 and 8659 are deleted
       - Rows with SMILES containing elements other than H, C, N, O, F, S are deleted
    '''
    use_columns = ['smiles', 'mpC', 'csid']
    all_elements = {
        'H', 'C', 'N', 'O', 'F', 'Br', 'I', 'Si', 'P', 'Cl', 'S', 'B'
    }
    allowed_elements = {'H', 'C', 'N', 'O', 'S'}
    df = pd.read_csv(data_path, usecols=use_columns)

    # Count the number of compounds containing each element
    element_counts = {
        element: df['smiles'].apply(lambda smiles: element in smiles).sum()
        for element in all_elements
    }
    for element, count in element_counts.items():
        print(f"Number of compounds containing {element}: {count}")

    # delete csid 10674 (could not generate mol because of too many rotatable bonds),
    # 8659 (could not calculate energy using mmff94 because mmff94 does not support Boron)
    df = df[~df['csid'].isin([10674, 8659])]

    # Filter SMILES column to include only specified elements
    def contains_only_allowed_elements(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
            return elements.issubset(allowed_elements)
        return False

    # Filtered
    df = df[df['smiles'].apply(contains_only_allowed_elements)]

    df.to_csv(f'{save_dir}/mp_{df.shape[0]}.csv', index=False)


def sort_name_delete_mol_sdf(input_sdf, output_sdf, ref_path):
    '''Sort SDF file by name (csid) and delete compounds 
    that are not in the reference CSV file.
    '''
    # Read the reference CSV file and get the list of 'csid' column
    ref_df = pd.read_csv(ref_path)
    ref_csid = set(ref_df['csid'].astype(str))  # Store numbers as strings

    # Read the SDF file
    suppl = Chem.SDMolSupplier(input_sdf, removeHs=False)

    # Store only valid molecules in a list and keep those whose _Name prefix exists in the reference list
    mols = []
    for mol in suppl:
        if mol is not None:
            # Split _Name by "_" and treat the first element as csid
            mol_csid = mol.GetProp('_Name').split('_')[0]
            if mol_csid in ref_csid:
                mols.append(mol)

    print(f'Number of valid conformers: {len(mols)}')

    # Extract molecule names, split xx_yy, and sort
    mols_sorted = sorted(
        mols, key=lambda x: tuple(map(int,
                                      x.GetProp('_Name').split('_'))))

    # Save sorted molecules to a new SDF
    writer = Chem.SDWriter(output_sdf)
    for mol in mols_sorted:
        writer.write(mol)
    writer.close()


if __name__ == '__main__':

    DATA_PATH = 'xxx'
    unique_elements = get_unique_elements(DATA_PATH)
    print('smiles_rotatable_bonds_6.csv contains:', unique_elements)

    DATA_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    os.makedirs(SAVE_DIR, exist_ok=True)
    # filter_elements_csv(DATA_PATH, SAVE_DIR)

    INPUT_SDF = 'xxx'
    SAVE_DIR = 'xxx'
    REF_PATH = 'xxx'
    os.makedirs(SAVE_DIR, exist_ok=True)
    OUTPUT_SDF = f'{SAVE_DIR}/conformers.sdf'
    sort_name_delete_mol_sdf(INPUT_SDF, OUTPUT_SDF, REF_PATH)
