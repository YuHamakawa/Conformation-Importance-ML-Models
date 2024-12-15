'''
make data for MolCLR fine-tuning
data need cid, smiles, label
'''
import os

import pandas as pd


def make_molclr_input_pqc_data(data_path, save_dir):
    df_ecfp = pd.read_csv(data_path,
                          sep='\t',
                          usecols=[
                              'cid', 'smiles', 'dipoleMoment', 'homo', 'gap',
                              'lumo', 'energy', 'enthalpy'
                          ])

    # sort by cid
    df_ecfp.sort_values('cid', inplace=True)
    # delete nan
    if df_ecfp['enthalpy'].isna().sum() > 0:
        missing = df_ecfp[df_ecfp['enthalpy'].isna()]
        print(f'some elements have missing value\n{missing}')
        df_ecfp.dropna(subset=['enthalpy'], inplace=True)

    df_ecfp.to_csv(os.path.join(save_dir, 'molclr_input_data.csv'),
                   index=False)


if __name__ == "__main__":
    DATA_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    os.makedirs(SAVE_DIR, exist_ok=True)
    make_molclr_input_pqc_data(DATA_PATH, SAVE_DIR)
