"""
This script calculates the number of rotatable bonds in a molecule using RDKit library.
It reads a CSV file containing molecular structures represented by SMILES strings,
calculates the rotatable bond count for each molecule in parallel using multiprocessing,
and saves the results to a new CSV file.
It also provides a function to extract the top 100k rows from the CSV file.
"""

import os
from multiprocessing import Pool, cpu_count
from typing import List, Optional

import pandas as pd
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import NumRotatableBonds
from tqdm import tqdm


def calculate_rotatable_bonds(smiles: str) -> Optional[int]:
    """
    Calculate the number of rotatable bonds in a molecule.

    Args:
        smiles (str): The SMILES representation of the molecule.

    Returns:
        Optional[int]: The number of rotatable bonds in the molecule, or None if the molecule is invalid.
    """
    mol = MolFromSmiles(smiles)
    if mol is not None:
        rot_bond_count = NumRotatableBonds(mol)
        return rot_bond_count
    return None


def calculate_rotatable_bonds_and_save(data_path: str, save_path: str) -> None:
    """
    Calculate the number of rotatable bonds for each molecule and save the results to a CSV file.

    Args:
        data_path (str): The path to the input CSV file containing molecular structures represented by SMILES strings.
        save_path (str): The path to save the output CSV file.

    Returns:
        None
    """
    # Automatically set the number of processes
    num_processes = min(cpu_count() or 1, (os.cpu_count() or 1) * 2)

    # Read the CSV file as a Pandas DataFrame
    df = pd.read_csv(data_path)

    # Create a list from the SMILES column
    smiles_list = df['pubchem.Isomeric SMILES'].tolist()

    # Create a process pool and start parallel processing
    with Pool(num_processes) as pool:
        # Display a progress bar
        rotatable_bond_counts: List[Optional[int]] = list(
            tqdm(pool.imap(calculate_rotatable_bonds, smiles_list),
                 total=len(smiles_list)))

    # Add the number of rotatable bonds as a new column to the DataFrame
    df['RotatableBonds'] = rotatable_bond_counts

    # Sort by the "RotatableBonds" column
    df = df.sort_values(by='RotatableBonds')

    # Save the results to a new CSV file
    df.to_csv(save_path, index=False)


def extract_top_100k(data_path: str, save_path: str) -> None:
    """
    Extract the top 100k rows from the CSV file.

    Args:
        data_path (str): The path to the input CSV file.
        save_path (str): The path to save the output CSV file.

    Returns:
        None
    """
    df = pd.read_csv(data_path)
    df = df.tail(100000)
    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    DATA_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    SAVE_PATH = os.path.join(SAVE_DIR, 'CHON500noSalt_rotatable.csv')
    calculate_rotatable_bonds_and_save(DATA_PATH, SAVE_PATH)

    DATA_PATH = 'xxx'
    SAVE_PATH = os.path.join(SAVE_DIR, 'CHON500noSalt_rotatable_top100k.csv')
    extract_top_100k(DATA_PATH, SAVE_PATH)
