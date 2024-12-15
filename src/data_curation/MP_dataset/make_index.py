import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def make_index(data_path: str, save_dir: str, seed: int) -> None:
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
    data = pd.read_csv(data_path)

    # Get the unique csid
    csids = data['csid'].unique()

    # Split the CIDs into 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (train_index, test_index) in enumerate(kf.split(csids)):
        train_csid = csids[train_index]
        test_csid = csids[test_index]

        # Save the index files
        np.save(os.path.join(save_dir, f'train_{i}.npy'), train_csid)
        np.save(os.path.join(save_dir, f'test_{i}.npy'), test_csid)


if __name__ == "__main__":
    DATA_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    seeds = [12, 22, 32, 42, 52]

    for seed in seeds:
        make_index(DATA_PATH, SAVE_DIR, seed)
