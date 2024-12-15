import os

import numpy as np
import pandas as pd
from ase import Atoms
from dscribe.descriptors import MBTR
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


def calc_morse_mbtr_descs_aptc(sdf_path, save_dir):
    ''' calculate morse and mbtr descriptors and save as csv
    concat correct labels, morse descs, mbtr descs
    
    Args:
        sdf_path: str, path to sdf file generated geometry by RDKit with activity label
        save_dir: str, directory path to save the csv file
    '''

    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mols = [mol for mol in supplier if mol is not None]

    ### Get MMFF energy, activity label and cid from _name ###
    props_data = []
    for mol in tqdm(mols, desc='Extracting prop from sdf'):
        props = mol.GetPropsAsDict()
        name = mol.GetProp('_Name')
        props['conf_id'] = name
        props['cid'] = name.split('_')[0] + '_' + name.split('_')[1]
        props_data.append(props)
    props_df = pd.DataFrame(props_data)

    ### calculate 3D-MORSE descriptors in Mordred module ###
    # temporary Calculator for checking descriptor names
    calc_temp = Calculator(descriptors, ignore_3D=False)
    # extract descriptors containing "Mor" meaning 3D-MORSE
    morse_descriptors = [
        desc for desc in calc_temp.descriptors if "Mor" in str(desc)
    ]
    # calsulate 3D-MORSE descriptors
    morse_calculator = Calculator(morse_descriptors, ignore_3D=False)
    morse_df = morse_calculator.pandas(mols, nproc=30)
    print('Morse descriptors:', morse_df.shape)

    ### calculate MBTR descriptors in DScribe module ###
    # MBTR settings
    k1 = {
        "geometry": {
            "function": "atomic_number"
        },
        "grid": {
            "min": 0,
            "max": 8,
            "n": 10,
            "sigma": 0.1
        },
    }
    k2 = {
        "geometry": {
            "function": "inverse_distance"
        },
        "grid": {
            "min": 0,
            "max": 4,
            "n": 10,
            "sigma": 0.1
        },
        "weighting": {
            "function": "exponential",
            "scale": 0.5,
            "cutoff": 1e-3
        },
    }
    k3 = {
        "geometry": {
            "function": "cosine"
        },
        "grid": {
            "min": -1,
            "max": 4,
            "n": 10,
            "sigma": 0.1
        },
        "weighting": {
            "function": "exponential",
            "scale": 0.5,
            "cutoff": 1e-3
        },
    }
    mbtr = MBTR(species=['H', 'C', 'N', 'O', 'F', 'Br', 'I'],
                k1=k1,
                k2=k2,
                k3=k3,
                periodic=False,
                normalization="l2_each")
    mbtr_descriptors = []
    for mol in tqdm(mols, desc='Calculating MBTR descriptors'):
        # convert RDKit Mol to ASE Atoms
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

        atoms = Atoms(symbols=symbols, positions=positions)

        mbtr_descriptor = mbtr.create(atoms)
        mbtr_descriptors.append(mbtr_descriptor)

    mbtr_descriptors_array = np.array(mbtr_descriptors)
    mbtr_df = pd.DataFrame(
        mbtr_descriptors_array,
        columns=[f'MBTR_{i}' for i in range(mbtr_descriptors_array.shape[1])])
    print('MBTR descriptors:', mbtr_df.shape)

    descs_df = pd.concat([props_df, morse_df, mbtr_df], axis=1)

    # Save the final dataframe to CSV
    os.makedirs(save_dir, exist_ok=True)
    descs_df.to_csv(f'{save_dir}/morse_mbtr_descs.csv', index=False)


def rename_concat_sdf(sdf_path_train, sdf_path_test, data_path, save_path):
    '''APTC dataset are separated into train and test sdf,
    so concat both sdf. Also rename mol._Name to 'tr_xx_yy' or 'te_xx_yy' where xx is index, yy is conformer id.
    and then, add the correct labels to the sdf file.
    
    Args:
        sdf_path_train: str, path to the training sdf file
        sdf_path_test: str, path to the testing sdf file
        data_path: str, path to the csv file contain correct labels
        save_path: str, path to save the concatenated sdf file
    '''

    data_df = pd.read_csv(data_path, usecols=['cid', 'ACTIVITY'])
    data_df = data_df.groupby('cid').mean().reset_index()

    # Load train and test molecules
    supplier_train = Chem.SDMolSupplier(sdf_path_train, removeHs=False)
    supplier_test = Chem.SDMolSupplier(sdf_path_test, removeHs=False)

    mols_train = [mol for mol in supplier_train if mol is not None]
    mols_test = [mol for mol in supplier_test if mol is not None]

    # Rename molecules
    for mol in mols_train:
        mol_name = mol.GetProp('_Name')
        mol.SetProp('_Name', f'tr_{mol_name}')

        idx = 'tr_' + mol_name.split('_')[0]
        activity = data_df[data_df['cid'] == idx]['ACTIVITY'].values[0]
        mol.SetProp('ACTIVITY', str(activity))

    for mol in mols_test:
        mol_name = mol.GetProp('_Name')
        mol.SetProp('_Name', f'te_{mol_name}')

        idx = 'te_' + mol_name.split('_')[0]
        activity = data_df[data_df['cid'] == idx]['ACTIVITY'].values[0]
        mol.SetProp('ACTIVITY', str(activity))

    # Concatenate molecules
    all_mols = mols_train + mols_test

    # Write to new sdf file
    writer = Chem.SDWriter(save_path)
    for mol in all_mols:
        writer.write(mol)
    writer.close()


if __name__ == "__main__":
    SAVE_DIR = 'xxx'
    os.makedirs(SAVE_DIR, exist_ok=True)

    SDF_PATH_TRAIN = 'xxx'
    SDF_PATH_TEST = 'xxx'
    DATA_PATH = 'xxx'
    SAVE_PATH = f'{SAVE_DIR}/aptc1_catalyst_conformers.sdf'
    rename_concat_sdf(SDF_PATH_TRAIN, SDF_PATH_TEST, DATA_PATH, SAVE_PATH)

    SDF_PATH_TRAIN = 'xxx'
    SDF_PATH_TEST = 'xxx'
    DATA_PATH = 'xxx'
    SAVE_PATH = f'{SAVE_DIR}/aptc2_catalyst_conformers.sdf'
    rename_concat_sdf(SDF_PATH_TRAIN, SDF_PATH_TEST, DATA_PATH, SAVE_PATH)

    SDF_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    calc_morse_mbtr_descs_aptc(SDF_PATH, SAVE_DIR)

    SDF_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    calc_morse_mbtr_descs_aptc(SDF_PATH, SAVE_DIR)
