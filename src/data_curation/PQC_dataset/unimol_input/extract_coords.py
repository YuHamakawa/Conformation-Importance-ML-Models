'''
Making dataset for Unimol
'''
import os
import sys
from itertools import islice

import numpy as np
import pandas as pd
from ..3d_descriptor.calc_aggregation import *
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
    merged_df.to_csv(
        f'{save_dir}/correct_coords_props_{merged_df.shape[0]}.csv',
        index=False)


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

    print('Saving aggregated descriptors to CSV')
    desc_max.to_csv(os.path.join(save_dir, f'1conf_{desc_max.shape[0]}.tsv'),
                    sep='\t',
                    index=False)
    desc_random.to_csv(os.path.join(save_dir,
                                    f'random_{desc_random.shape[0]}.tsv'),
                       sep='\t',
                       index=False)
    desc_rmsd_min.to_csv(os.path.join(
        save_dir, f'rmsd_min_{desc_rmsd_min.shape[0]}.tsv'),
                         sep='\t',
                         index=False)
    desc_rmsd_max.to_csv(os.path.join(
        save_dir, f'rmsd_max_{desc_rmsd_max.shape[0]}.tsv'),
                         sep='\t',
                         index=False)


if __name__ == '__main__':
    DATA_PATH_SDF = 'xxx'
    DATA_PATH_PROPS = 'xxx'
    SAVE_DIR = 'xxx'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # extract_coords_correct(DATA_PATH_SDF, DATA_PATH_PROPS, SAVE_DIR)
    DATA_PATH_COORDS = 'xxx'
    extract_coords_agg(DATA_PATH_COORDS,
                       DATA_PATH_PROPS,
                       SAVE_DIR,
                       test_mode=False)
