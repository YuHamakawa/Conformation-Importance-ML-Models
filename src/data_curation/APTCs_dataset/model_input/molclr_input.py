'''
make data for MolCLR fine-tuning
data need cid, smiles, label
'''
import os

import pandas as pd


def make_molclr_input_pqc_data(data_path, save_dir):
    df_ecfp = pd.read_csv(data_path,
                          sep='\t',
                          usecols=['cid', 'CATALYST', 'ACTIVITY'])

    df_ecfp.rename(columns={'CATALYST': 'smiles'}, inplace=True)

    df_ecfp.to_csv(os.path.join(save_dir, 'molclr_input_data.csv'),
                   index=False)


if __name__ == "__main__":
    DATA_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    os.makedirs(SAVE_DIR, exist_ok=True)
    make_molclr_input_pqc_data(DATA_PATH, SAVE_DIR)

    DATA_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    os.makedirs(SAVE_DIR, exist_ok=True)
    make_molclr_input_pqc_data(DATA_PATH, SAVE_DIR)
