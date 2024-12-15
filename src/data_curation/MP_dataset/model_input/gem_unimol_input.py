'''
Making dataset for GEM, unimol
'''
import os
import sys
from itertools import islice

import numpy as np
import pandas as pd
from rdkit import Chem
from step7_calc_aggregation import *
from tqdm import tqdm


def make_unimol_input(sdf_path: str, save_path: str):
    '''
    extract correct coords and label from SDF file
    save as csv
    '''

    # Load the SDF file
    supplier = Chem.SDMolSupplier(sdf_path)

    # Initialize lists to store the data
    data: dict[str, list] = {
        'csid_conf': [],
        'csid': [],
        'mpC': [],
        'atoms': [],
        'coordinates': [],
    }

    # Loop over all molecules in the SDF file
    for mol in tqdm(supplier):  # for mol in islice(supplier, 100):
        # Get the first conformer
        conf = mol.GetConformer()

        # Get atoms and coordinates
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coordinates = [
            list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())
        ]

        # Get the title (csid_conf)
        csid_conf = mol.GetProp('_Name')
        csid = int(csid_conf.split('_')[0])

        mpC = float(mol.GetProp('mpC'))

        # Add the data to the lists
        data['csid_conf'].append(csid_conf)
        data['csid'].append(csid)
        data['mpC'].append(mpC)
        data['atoms'].append(atoms)
        data['coordinates'].append(coordinates)

    # Convert the data dictionary to a DataFrame
    coods_df = pd.DataFrame(data)

    coods_df.sort_values(by='csid', inplace=True)
    # Save the merged DataFrame as a CSV file
    coods_df.to_csv(save_path, index=False)


def make_gem_input(sdf_path, data_path, save_dir, test_mode):
    '''
    extract certain conformation from SDF file for GEM input
    methods: 
        - random
        - most energetically favored
        - RMSD nearest to correct coordinates
        - RMSD farthest to correct
        - non-aggregated
    
    Args:
        sdf_path: path to SDF file (after PM6 optimization)
        data_path: path to csv file containing cid, properties, energy, etc.
        save_dir: path to save the output files
    '''

    if test_mode:
        df_data = pd.read_csv(
            data_path,
            usecols=['mpC', 'csid', 'csid_conf', 'MMFF_Energy'],
            nrows=100)
    else:
        df_data = pd.read_csv(
            data_path, usecols=['mpC', 'csid', 'csid_conf', 'MMFF_Energy'])

    # In this data, cid = compound id, conf_id = conformer id

    # In this data, cid = conformer id, comp_id = compound id

    idx = 'csid'
    p_value = 'p_value'
    print('Calc. BoltzProb from existing energy data.')
    df_data[p_value] = energy_to_boltzmann_prob(df_data[['MMFF_Energy', idx]],
                                                T=298,
                                                unit='kcal/mol')

    print('Calc. aggregated descs.')
    # most energitically favord one.
    desc_max = agg_max_existing_prob_descs(df_data, idx, p_value)
    # randomly select one conformer
    desc_random = agg_random_descs(df_data, idx, p_value)

    # read sdf and convert to mol object
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mols = [mol for mol in supplier if mol is not None]

    # get 'conf_id' list
    all_conf_id_list = df_data['csid_conf'].tolist()
    desc_max_conf_id_list = desc_max['csid_conf'].tolist()
    desc_random_conf_id_list = desc_random['csid_conf'].tolist()

    # get properties dict
    properties_dict = df_data.set_index('csid_conf')[['mpC']].to_dict('index')

    # open sdwriter
    all_writer = Chem.SDWriter(os.path.join(save_dir, 'non_aggregation.sdf'))
    global_min_writer = Chem.SDWriter(
        os.path.join(save_dir, 'global_minimum.sdf'))
    random_writer = Chem.SDWriter(os.path.join(save_dir, 'random.sdf'))

    all_count = 0
    global_min_count = 0
    random_count = 0

    for mol in mols:
        # first, delete mol property
        prop_names = list(mol.GetPropNames())
        for prop in prop_names:
            mol.ClearProp(prop)

        mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else ''

        if mol_name in all_conf_id_list:
            props = properties_dict.get(mol_name, {})
            for prop_name, prop_value in props.items():
                mol.SetProp(prop_name, str(prop_value))
            all_writer.write(mol)
            all_count += 1

            if mol_name in desc_max_conf_id_list:
                global_min_writer.write(mol)
                global_min_count += 1
            if mol_name in desc_random_conf_id_list:
                random_writer.write(mol)
                random_count += 1

    all_writer.close()
    global_min_writer.close()
    random_writer.close()

    print(
        f'All: {all_count}, Global Min: {global_min_count}, Random: {random_count}'
    )


if __name__ == '__main__':
    SDF_PATH = 'xxx'
    DATA_PATH = 'xxx'
    SAVE_DIR_GEM = 'xxx'
    os.makedirs(SAVE_DIR_GEM, exist_ok=True)
    make_gem_input(SDF_PATH, DATA_PATH, SAVE_DIR_GEM, test_mode=False)

    SAVE_DIR_UNIMOL = 'xxx'
    os.makedirs(SAVE_DIR_UNIMOL, exist_ok=True)
    sdf_list = ['global_minimum.sdf', 'random.sdf', 'non_aggregation.sdf']
    save_name_list = [
        'global_minimum.csv', 'random.csv', 'non_aggregation.csv'
    ]
    for sdf, save_name in zip(sdf_list, save_name_list):
        make_unimol_input(os.path.join(SAVE_DIR_GEM, sdf),
                          os.path.join(SAVE_DIR_UNIMOL, save_name))
