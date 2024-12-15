import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.append('3D-MIL-QSSR/miqssr')
from utils import calc_pm6_descr


def calc_pmapper(data_path, save_dir, ref_path):
    """
    Calculate pmapper descriptors for generated conformers.
    
    Args:
        data_path (str): Path to the input data file containing conformers.
        save_dir (str): Directory where the output files will be saved.
        ref_path (str): Path to the reference CSV file, as data_path contains too many compounds.
    """
    smarts_file = os.path.join(save_dir, 'smarts_features.txt')

    bags_dict_pm6 = calc_pm6_descr(conf_file=data_path,
                                   smarts_features=smarts_file,
                                   ncpu=10,
                                   num_descr=[3],
                                   path=save_dir)

    csid_list = []
    bags_list = []
    for csid, bags in bags_dict_pm6.items():
        csid_list.append(csid)
        bags_list.append(bags)

    # Convert bags_list to a DataFrame
    bags_df = pd.DataFrame({'csid': csid_list, 'bags': bags_list})
    # read reference tsv file containing 'csid' column
    ref_csid_df = pd.read_csv(ref_path, usecols=['csid'])
    bags_df = bags_df[bags_df['csid'].isin(ref_csid_df['csid'])]
    bags_df.to_pickle(os.path.join(save_dir, 'pmapper_descs.pkl'))

    # Expand bags_df to separate each row into 1028-dimensional vectors
    expanded_df = bags_df.explode('bags', ignore_index=True)
    # Treat each element as a 1028-dimensional vector and convert from list to np.array
    expanded_df['bags'] = expanded_df['bags'].apply(
        lambda x: np.array(x) if isinstance(x, list) else x)
    pmapper_dim = expanded_df['bags'].iloc[0].shape[-1]
    # Split the 'bags' column into pmapper_dim columns, prefixing each column with 'pmapper_'
    bags_expanded = pd.DataFrame(
        expanded_df['bags'].tolist(),
        columns=[f'pmapper_{i}' for i in range(pmapper_dim)])
    # Combine with 'csid' column
    expanded_df = pd.concat([expanded_df[['csid']], bags_expanded], axis=1)
    # Filter out rows in expanded_df that have 'csid' not present in ref_csid_df
    pmapper_df = expanded_df[expanded_df['csid'].isin(ref_csid_df['csid'])]
    pmapper_df.to_csv(os.path.join(save_dir, 'pmapper_descs.csv'), index=False)


if __name__ == '__main__':
    # generated conformers sorted by name(csid)
    DATA_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    REF_PATH = 'xxx'
    os.makedirs(SAVE_DIR, exist_ok=True)
    calc_pmapper(DATA_PATH, SAVE_DIR, REF_PATH)
