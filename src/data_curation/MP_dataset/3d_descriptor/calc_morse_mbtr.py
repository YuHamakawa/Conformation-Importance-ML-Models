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

    # morse need Hs
    mols = [Chem.AddHs(mol, addCoords=True) for mol in mols]

    ### Get MMFF energy, activity label and cid from _name ###
    props_data = []
    for mol in tqdm(mols, desc='Extracting prop from sdf'):
        props = mol.GetPropsAsDict()
        name = mol.GetProp('_Name')
        props['csid_conf'] = name
        props['csid'] = name.split('_')[0]
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
    mbtr = MBTR(species=['H', 'C', 'N', 'O', 'S'],
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

    descs_df.to_csv(f'{save_dir}/morse_mbtr_descs.csv', index=False)


def concat_3d_descs(label_data_path, moe_data_path, pmapper_data_path,
                    pmapper_rowname_path, morse_mbtr_data_path, save_dir):
    '''concat 3d descriptors, label, and then save as csv
    csid, label, smiles, mpC from label_data_path
    moe descriptors from moe_data_path
    pmapper descriptors from pmapper_data_path
    morse and mbtr descriptors from morse_mbtr_data_path
    
    pmapper_rowname_path contain 'csid_conf'
    '''
    label_df = pd.read_csv(label_data_path)  # columns = smiles,mpC,csid

    moe_df = pd.read_csv(
        moe_data_path)  # columns = mol,csid_conf,MMFF_Energy, 117 descs...
    moe_df.drop(columns=['mol'], inplace=True)

    pmapper_df = pd.read_csv(
        pmapper_data_path)  # columns = csid,pmapper_0,..., pmapper_574
    pmapper_rowname_df = pd.read_csv(pmapper_rowname_path,
                                     header=None)  # columns = csid_conf
    pmapper_df['csid_conf'] = pmapper_rowname_df[0]
    pmapper_df = pmapper_df[
        ['csid_conf'] +
        [col for col in pmapper_df.columns if col != 'csid_conf']]
    pmapper_df.drop(columns=['csid'], inplace=True)

    morse_mbtr_df = pd.read_csv(
        morse_mbtr_data_path
    )  # columns = MMFF_Energy,csid_conf,csid,Mor01,...,Mor32p,MBTR_0,...,MBTR_949
    morse_mbtr_df.drop(columns=['MMFF_Energy'], inplace=True)

    merged_df = pd.merge(moe_df, pmapper_df, on='csid_conf')
    merged_df = pd.merge(merged_df, morse_mbtr_df, on='csid_conf')
    final_df = pd.merge(label_df, merged_df, on='csid')

    final_df.sort_values(by='csid', inplace=True)
    # columns = smiles,mpC,csid,csid_conf,MMFF_Energy,117 descs,575 pmapper descs,32 morse descs,950 mbtr descs
    final_df.to_csv(f'{save_dir}/3d_descriptors.csv', index=False)


if __name__ == "__main__":
    SDF_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    os.makedirs(SAVE_DIR, exist_ok=True)
    # calc_morse_mbtr_descs_aptc(SDF_PATH, SAVE_DIR)

    LABEL_DATA_PATH = 'xxx'
    MOE_DATA_PATH = 'xxx'
    PMAPPER_DATA_PATH = 'xxx'
    PMAPPER_ROWNAME_PATH = 'xxx'
    MORSE_MBTR_DATA_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    concat_3d_descs(LABEL_DATA_PATH, MOE_DATA_PATH, PMAPPER_DATA_PATH,
                    PMAPPER_ROWNAME_PATH, MORSE_MBTR_DATA_PATH, SAVE_DIR)
