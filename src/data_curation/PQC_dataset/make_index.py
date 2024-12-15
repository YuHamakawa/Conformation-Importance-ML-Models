import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def make_index(data_path: str, save_dir: str, y_name: str, seed: int) -> None:
    """
    Make index for 5-fold cross-validation.

    Args:
        data_path (str): The path to the data file.
        save_dir (str): The directory to save the index files.
        y_name (str): The name of the target variable.

    Returns:
        None
    """
    save_dir = f'{save_dir}/seed_{seed}'
    os.makedirs(save_dir, exist_ok=True)

    # Load the data
    data = pd.read_csv(data_path, sep='\t')

    # Remove rows with missing values in the target variable
    data = data.dropna(subset=[y_name])

    # Get the unique CIDs
    cids = data['cid'].unique()

    # Split the CIDs into 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (train_index, test_index) in enumerate(kf.split(cids)):
        train_cid = cids[train_index]
        test_cid = cids[test_index]

        # Save the index files
        np.save(os.path.join(save_dir, f'train_{i}.npy'), train_cid)
        np.save(os.path.join(save_dir, f'test_{i}.npy'), test_cid)


if __name__ == "__main__":
    DATA_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    Y_NAME = 'enthalpy'  # only enthalpy has missing values
    seeds = [32, 42, 52]

    for seed in seeds:
        make_index(DATA_PATH, SAVE_DIR, Y_NAME, seed)
