'''
Making dataset for GEM
'''
import os
import sys
from itertools import islice

import numpy as np
import pandas as pd
from calc_aggregation import *
from rdkit import Chem
from tqdm import tqdm


def extract_coords_correct(data_path_sdf: str, data_path_props: str,
                           save_dir: str):
    '''
    extract correct coords from SDF file
    add objective variables
    save as csv
    '''

    # Load the SDF file
    supplier = Chem.SDMolSupplier(data_path_sdf)

    # Initialize lists to store the data
    data = {
        'cid': [],
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

        # Get the title (cid)
        cid = int(mol.GetProp('_Name'))

        # Add the data to the lists
        data['cid'].append(cid)
        data['atoms'].append(atoms)
        data['coordinates'].append(coordinates)

    # Convert the data dictionary to a DataFrame
    coods_df = pd.DataFrame(data)

    # Load the CSV file
    target_df = pd.read_csv(data_path_props,
                            usecols=[
                                'cid', 'dipoleMoment', 'homo', 'gap', 'lumo',
                                'energy', 'enthalpy'
                            ],
                            sep='\t')

    # Merge the two DataFrames on the 'cid' column
    merged_df = pd.merge(coods_df, target_df, on='cid')
    merged_df.sort_values(by='cid', inplace=True)
    # Save the merged DataFrame as a CSV file
    # merged_df.to_csv(
    #     f'{save_dir}/correct_coords_props_{merged_df.shape[0]}.csv',
    #     index=False)


def extract_coords_agg(data_path_coords,
                       data_path_props,
                       save_dir,
                       test_mode=False):

    idx = 'cid'
    p_value = 'p_value'
    df_data = pd.read_csv(data_path_coords)
    if test_mode:
        df_data = df_data.head(100)

    df_data[idx] = df_data[idx].apply(lambda x: x.split('_')[0])
    print('Calc. BoltzProb from existing energy data.')
    df_data[p_value] = energy_to_boltzmann_prob(df_data[['total_energy', idx]],
                                                T=298,
                                                unit='kcal/mol')

    print('Calc. aggregated descs.')
    # most energitically favord one.
    desc_max = agg_max_existing_prob_descs(df_data, idx, p_value)
    # randomly select one conformer
    desc_random = agg_random_descs(df_data, idx, p_value)

    # select RMSD nearest & farthest to correct molecule coordinates
    desc_rmsd_min = agg_rmsd_descs(df_data, idx, p_value, transform='min')
    desc_rmsd_max = agg_rmsd_descs(df_data, idx, p_value, transform='max')

    print('Merge coords and properties')
    # Load the CSV file
    target_df = pd.read_csv(data_path_props,
                            usecols=[
                                'cid', 'dipoleMoment', 'homo', 'gap', 'lumo',
                                'energy', 'enthalpy'
                            ],
                            sep='\t')

    desc_max.reset_index(level=0, inplace=True)
    desc_max = pd.merge(desc_max.astype({'cid': 'int64'}), target_df, on='cid')
    desc_max.sort_values(by=idx, inplace=True)

    desc_random.reset_index(level=0, inplace=True)
    desc_random = pd.merge(desc_random.astype({'cid': 'int64'}),
                           target_df,
                           on='cid')
    desc_random.sort_values(by=idx, inplace=True)

    desc_rmsd_min.reset_index(level=0, inplace=True)
    desc_rmsd_min = pd.merge(desc_rmsd_min.astype({'cid': 'int64'}),
                             target_df,
                             on='cid')
    desc_rmsd_min.sort_values(by=idx, inplace=True)

    desc_rmsd_max.reset_index(level=0, inplace=True)
    desc_rmsd_max = pd.merge(desc_rmsd_max.astype({'cid': 'int64'}),
                             target_df,
                             on='cid')
    desc_rmsd_max.sort_values(by=idx, inplace=True)

    print('Confirming that all shapes are the same:', desc_max.shape,
          desc_random.shape, desc_rmsd_max.shape, desc_rmsd_min.shape)

    print('Mean of rmsd:', desc_max['rmsd'].mean(), desc_random['rmsd'].mean(),
          desc_rmsd_max['rmsd'].mean(), desc_rmsd_min['rmsd'].mean())

    # print('Saving aggregated descriptors to CSV')
    # desc_max.to_csv(os.path.join(save_dir, f'1conf_{desc_max.shape[0]}.tsv'),
    #                 sep='\t',
    #                 index=False)
    # desc_random.to_csv(os.path.join(save_dir,
    #                                 f'random_{desc_random.shape[0]}.tsv'),
    #                    sep='\t',
    #                    index=False)
    # desc_rmsd_min.to_csv(os.path.join(
    #     save_dir, f'rmsd_min_{desc_rmsd_min.shape[0]}.tsv'),
    #                      sep='\t',
    #                      index=False)
    # desc_rmsd_max.to_csv(os.path.join(
    #     save_dir, f'rmsd_max_{desc_rmsd_max.shape[0]}.tsv'),
    #                      sep='\t',
    #                      index=False)


def extract_conformation(sdf_path, data_path, save_dir, test_mode):
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
        df_data = pd.read_csv(data_path,
                              usecols=[
                                  'cid', 'rmsd', 'total_energy',
                                  'dipoleMoment', 'homo', 'gap', 'lumo',
                                  'energy', 'enthalpy'
                              ],
                              nrows=100)
    else:
        df_data = pd.read_csv(data_path,
                              usecols=[
                                  'cid', 'rmsd', 'total_energy',
                                  'dipoleMoment', 'homo', 'gap', 'lumo',
                                  'energy', 'enthalpy'
                              ])

    # In this data, cid = conformer id, comp_id = compound id
    idx = 'comp_id'
    p_value = 'p_value'

    df_data[idx] = df_data['cid'].apply(lambda x: x.split('_')[0])
    print('Calc. BoltzProb from existing energy data.')
    df_data[p_value] = energy_to_boltzmann_prob(df_data[['total_energy', idx]],
                                                T=298,
                                                unit='kcal/mol')

    print('Calc. aggregated descs.')
    # most energitically favord one.
    desc_max = agg_max_existing_prob_descs(df_data, idx, p_value)
    # randomly select one conformer
    desc_random = agg_random_descs(df_data, idx, p_value)
    # select RMSD nearest & farthest to correct molecule coordinates
    desc_rmsd_min = agg_rmsd_descs(df_data, idx, p_value, transform='min')
    desc_rmsd_max = agg_rmsd_descs(df_data, idx, p_value, transform='max')

    # read sdf and convert to mol object
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mols = [mol for mol in supplier if mol is not None]

    # get 'cid' list
    all_cid_list = df_data['cid'].tolist()
    desc_max_cid_list = desc_max['cid'].tolist()
    desc_random_cid_list = desc_random['cid'].tolist()
    desc_rmsd_min_cid_list = desc_rmsd_min['cid'].tolist()
    desc_rmsd_max_cid_list = desc_rmsd_max['cid'].tolist()

    # get properties dict
    properties_dict = df_data.set_index('cid')[[
        'dipoleMoment', 'homo', 'gap', 'lumo', 'energy', 'enthalpy'
    ]].to_dict('index')

    # open sdwriter
    all_writer = Chem.SDWriter(os.path.join(save_dir, 'non_aggregation.sdf'))
    global_min_writer = Chem.SDWriter(
        os.path.join(save_dir, 'global_minimum.sdf'))
    random_writer = Chem.SDWriter(os.path.join(save_dir, 'random.sdf'))
    rmsd_min_writer = Chem.SDWriter(os.path.join(save_dir, 'rmsd_min.sdf'))
    rmsd_max_writer = Chem.SDWriter(os.path.join(save_dir, 'rmsd_max.sdf'))

    # initialize counter
    all_count = 0
    global_min_count = 0
    random_count = 0
    rmsd_min_count = 0
    rmsd_max_count = 0

    for mol in mols:
        # first, delete mol property
        prop_names = list(mol.GetPropNames())
        for prop in prop_names:
            mol.ClearProp(prop)

        mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else ''

        if mol_name in all_cid_list:
            # get properties
            props = properties_dict.get(mol_name, {})
            # add propertirs to mol
            for prop_name, prop_value in props.items():
                mol.SetProp(prop_name, str(prop_value))
            all_writer.write(mol)
            all_count += 1

            if mol_name in desc_max_cid_list:
                global_min_writer.write(mol)
                global_min_count += 1
            if mol_name in desc_random_cid_list:
                random_writer.write(mol)
                random_count += 1
            if mol_name in desc_rmsd_min_cid_list:
                rmsd_min_writer.write(mol)
                rmsd_min_count += 1
            if mol_name in desc_rmsd_max_cid_list:
                rmsd_max_writer.write(mol)
                rmsd_max_count += 1

    all_writer.close()
    global_min_writer.close()
    random_writer.close()
    rmsd_min_writer.close()
    rmsd_max_writer.close()

    print(
        f'All: {all_count}, Global Min: {global_min_count}, Random: {random_count}, RMSD Min: {rmsd_min_count}, RMSD Max: {rmsd_max_count}'
    )


def extract_ground_truth_conformation(sdf_path, data_path, save_dir,
                                      test_mode):
    if test_mode:
        sdf_path = 'xxx'

    df_ecfp = pd.read_csv(data_path,
                          sep='\t',
                          usecols=[
                              'cid', 'dipoleMoment', 'homo', 'gap', 'lumo',
                              'energy', 'enthalpy'
                          ])
    # delete nan
    if df_ecfp['enthalpy'].isna().sum() > 0:
        missing = df_ecfp[df_ecfp['enthalpy'].isna()]
        print(f'some elements have missing value\n{missing}')
        df_ecfp.dropna(subset=['enthalpy'], inplace=True)

    properties_dict = df_ecfp.set_index('cid')[[
        'dipoleMoment', 'homo', 'gap', 'lumo', 'energy', 'enthalpy'
    ]].to_dict('index')

    cid_list = df_ecfp['cid'].tolist()

    # read sdf and convert to mol object
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mols = [mol for mol in supplier if mol is not None]

    writer = Chem.SDWriter(os.path.join(save_dir, 'ground_truth.sdf'))
    count = 0
    for mol in mols:
        # first, delete mol properintty
        prop_names = list(mol.GetPropNames())
        for prop in prop_names:
            mol.ClearProp(prop)

        mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else ''

        if int(mol_name) in cid_list:
            # df_dataからプロパティを取得
            props = properties_dict.get(int(mol_name), {})
            # プロパティをmolに追加
            for prop_name, prop_value in props.items():
                mol.SetProp(prop_name, str(prop_value))
            writer.write(mol)
            count += 1
    writer.close()
    print(f'Ground truth: {count}')


if __name__ == '__main__':

    SDF_PATH = 'xxx'
    DATA_PATH = 'xxx'
    SAVE_DIR_GEM = 'xxx'
    os.makedirs(SAVE_DIR_GEM, exist_ok=True)
    extract_conformation(SDF_PATH, DATA_PATH, SAVE_DIR_GEM, test_mode=False)

    SDF_PATH = 'xxx'
    DATA_PATH = 'xxx'
    os.makedirs(SAVE_DIR_GEM, exist_ok=True)
    extract_ground_truth_conformation(SDF_PATH,
                                      DATA_PATH,
                                      SAVE_DIR_GEM,
                                      test_mode=False)
