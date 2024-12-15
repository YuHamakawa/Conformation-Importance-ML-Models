import os

import numpy as np
import pandas as pd
from ase import Atoms
from dscribe.descriptors import MBTR
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


def calc_morse_mbtr_descs(sdf_path, data_path, save_dir, rmsd_path):
    ''' calculate morse and mbtr descriptors and save as csv
    concat correct labels, moe descs, morse descs, mbtr descs
    
    Args:
        sdf_path: str, path to sdf file contain oprimized geometry with MOPAC PM6 and MOE descs
        data_path: str, path to csv file contain correct labels
        save_dir: str, directory path to save the csv file
        rmsd_path: str, path to csv file contain RMSD values to ground truth conformation
    '''

    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mols = [mol for mol in supplier if mol is not None]

    ### Get molecule properties and cid from _name ###
    props_data = []
    for mol in tqdm(mols, desc='Extracting MOE data'):
        props = mol.GetPropsAsDict()
        props['cid'] = mol.GetProp('_Name')
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
    mbtr = MBTR(species=['H', 'C', 'N', 'O'],
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

        # generate ASE Atoms object
        atoms = Atoms(symbols=symbols, positions=positions)

        # calculate MBTR descriptors
        mbtr_descriptor = mbtr.create(atoms)
        mbtr_descriptors.append(mbtr_descriptor)

    # output results as numpy array or csv
    mbtr_descriptors_array = np.array(mbtr_descriptors)
    mbtr_df = pd.DataFrame(
        mbtr_descriptors_array,
        columns=[f'MBTR_{i}' for i in range(mbtr_descriptors_array.shape[1])])

    descs_df = pd.concat([props_df, morse_df, mbtr_df], axis=1)

    ### concat correct labels, delete missing value, save ###
    labels_df = pd.read_csv(data_path)
    selected_columns = [
        'cid', 'dipoleMoment', 'homo', 'gap', 'lumo', 'energy', 'enthalpy'
    ]
    labels_df = labels_df[selected_columns]

    # Merge descriptors and labels on 'cid'
    merge_df = pd.merge(descs_df, labels_df, on='cid', how='inner')
    # Drop rows with missing values in the 'enthalpy' column
    merge_df.dropna(subset=['enthalpy'], inplace=True)

    # merge rmsd
    rmsd_df = pd.read_csv(rmsd_path).drop(columns=['total_energy'])
    merged_rmsd_df = pd.merge(merge_df,
                              rmsd_df,
                              left_on='cid',
                              right_on='mol2d_idx',
                              how='inner').drop(columns=['mol2d_idx'])

    # Save the final dataframe to CSV
    os.makedirs(save_dir, exist_ok=True)
    merged_rmsd_df.to_csv(f'{save_dir}/moe_morse_mbtr_descs.csv', index=False)


def calc_morse_mbtr_descs_ground_truth_conformation(sdf_path, data_path,
                                                    ref_path, save_dir):
    '''calculate morse and mbtr descriptors to ground-truth conformation
    1. read sdf file contain 100k compounds
    2. calc morse and mbtr descriptors
    3. read correct labels and moe descs
    4. read reference data and delete some cid from correct labels
    
    Args:
        sdf_path: str, path to sdf file contain 100k compounds
        data_path: str, path to csv file contain MOE descs & correct labels
        ref_path: str, path to csv file contain cid. Need to delete some cid from ground-truth data, because it have much more data than aggregated data
        save_dir: str, directory path to save the csv file
    '''

    supplier = Chem.SDMolSupplier(sdf_path, removeHs=True)
    mols = [mol for mol in supplier if mol is not None]
    mols = [Chem.AddHs(mol, addCoords=True) for mol in mols]

    ### Get molecule cid from _name ###
    props_data = []
    for mol in tqdm(mols, desc='Extracting MOE data'):
        props = mol.GetPropsAsDict()
        props['cid'] = mol.GetProp('_Name')
        props_data.append(props)
    cid_df = pd.DataFrame(props_data)

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
    mbtr = MBTR(species=['H', 'C', 'N', 'O'],
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

        # generate ASE Atoms object
        atoms = Atoms(symbols=symbols, positions=positions)

        # caculate MBTR descriptors
        mbtr_descriptor = mbtr.create(atoms)
        mbtr_descriptors.append(mbtr_descriptor)

    # output results as numpy array or csv
    mbtr_descriptors_array = np.array(mbtr_descriptors)
    mbtr_df = pd.DataFrame(
        mbtr_descriptors_array,
        columns=[f'MBTR_{i}' for i in range(mbtr_descriptors_array.shape[1])])

    morse_mbtr_descs = pd.concat([cid_df, morse_df, mbtr_df], axis=1)

    moe_df = pd.read_csv(data_path, sep='\t').drop(columns=['CanonicalSMILES'])
    morse_mbtr_descs['cid'] = morse_mbtr_descs['cid'].astype(int)
    # Merge descriptors on 'cid'
    descs_df = pd.merge(moe_df, morse_mbtr_descs, on='cid', how='inner')

    ref_cid_df = pd.read_csv(ref_path, sep='\t', usecols=['cid'])
    # Filter out rows in descs_df that have 'cid' not present in ref_cid_df
    descs_df = descs_df[descs_df['cid'].isin(ref_cid_df['cid'])]

    print(descs_df.shape)
    print(descs_df.columns.to_list())

    # Save the final dataframe to TSV
    os.makedirs(save_dir, exist_ok=True)
    # Save the first 100 rows to TSV
    descs_df.head(100).to_csv(f'{save_dir}/ground_truth_100.tsv',
                              sep='\t',
                              index=False)
    descs_df.to_csv(f'{save_dir}/ground_truth.tsv', sep='\t', index=False)


if __name__ == "__main__":
    SDF_PATH = 'xxx'
    DATA_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    RMSD_PATH = 'xxx'
    calc_morse_mbtr_descs(SDF_PATH, DATA_PATH, SAVE_DIR, RMSD_PATH)

    SDF_PATH = 'xxx'
    DATA_PATH = 'xxx'
    REF_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    calc_morse_mbtr_descs_ground_truth_conformation(SDF_PATH, DATA_PATH,
                                                    REF_PATH, SAVE_DIR)
