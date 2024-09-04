import os

import pandas as pd


def preprocess_moe_descs(data_path_1: str,
                         data_path_2: str,
                         save_dir: str,
                         test_mode: bool = False):
    """Preprocesses MOE descriptors and returns the preprocessed data.

    Args:
        data_path_1 (str): The file path of the first data file.
        data_path_2 (str): The file path of the second data file.
        save_dir (str): The directory to save the preprocessed data.
        test_mode (bool, optional): Whether to run in test mode. Defaults to False.

    Returns:
        None

    """
    name_2d = 'cid'
    idx_descs = pd.read_csv(data_path_1)
    if test_mode:
        idx_descs = idx_descs.head(100)  # for small-data testing
    idx_descs.rename(columns={'mol2d_idx': 'cid'}, inplace=True)
    descs_columns = [name_2d] + idx_descs.columns[18:-1].to_list()
    descs = idx_descs.loc[:, descs_columns]
    drop_list = [
        'dipoleX',
        'dipoleY',
        'dipoleZ',
        'E_rele',
        'E_rnb',
        'E_rsol',
        'E_rvdw',
        'pmiX',
        'pmiY',
        'pmiZ',
    ]
    descs.drop(columns=drop_list, inplace=True)

    # read pm6 data
    pm6 = pd.read_csv(data_path_2)
    pm6 = pm6.loc[:, [
        'pubchem.cid', 'pubchem.PM6.properties.total dipole moment',
        'pubchem.PM6.properties.energy.alpha.homo',
        'pubchem.PM6.properties.energy.alpha.gap',
        'pubchem.PM6.properties.energy.alpha.lumo',
        'pubchem.PM6.properties.energy.total',
        'pubchem.PM6.properties.enthalpy',
        'pubchem.PM6.openbabel.Canonical SMILES'
    ]]

    pm6.rename(columns={
        'pubchem.PM6.properties.total dipole moment': 'dipoleMoment',
        'pubchem.PM6.properties.energy.alpha.homo': 'homo',
        'pubchem.PM6.properties.energy.alpha.gap': 'gap',
        'pubchem.PM6.properties.energy.alpha.lumo': 'lumo',
        'pubchem.PM6.properties.energy.total': 'energy',
        'pubchem.PM6.properties.enthalpy': 'enthalpy',
        'pubchem.PM6.openbabel.Canonical SMILES': 'CanonicalSMILES'
    },
               inplace=True)

    descs['compound_id'] = descs['cid'].apply(lambda x: x.split('_')[0])
    descs['compound_id'] = descs['compound_id'].astype(int)
    pm6['pubchem.cid'] = pm6['pubchem.cid'].astype(int)
    descs = pd.merge(descs,
                     pm6[[
                         'pubchem.cid', 'dipoleMoment', 'homo', 'gap', 'lumo',
                         'energy', 'enthalpy'
                     ]],
                     left_on='compound_id',
                     right_on='pubchem.cid',
                     how='left')

    descs.drop(columns=['pubchem.cid', 'compound_id'], inplace=True)
    descs.to_csv(os.path.join(save_dir, 'descs_no_agg.csv'), index=False)


if __name__ == '__main__':
    DATA_PATH_1 = 'xxx'
    DATA_PATH_2 = 'xxx'
    SAVE_DIR = 'xxx'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    preprocess_moe_descs(DATA_PATH_1, DATA_PATH_2, SAVE_DIR, test_mode=False)
