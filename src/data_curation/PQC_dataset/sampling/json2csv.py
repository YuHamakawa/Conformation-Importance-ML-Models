"""
This script reads and processes multiple JSON files in a directory,
and merges the data into a single DataFrame.
"""

import json
import os
from multiprocessing import Pool
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm


def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read and parse a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def process_json_file(file_path: str) -> pd.DataFrame:
    """
    Process a JSON file and return a pandas DataFrame.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        pandas.DataFrame: The normalized JSON data as a DataFrame.
    """
    json_data = read_json_file(file_path)
    return pd.json_normalize(json_data)


def find_json_files(root_dir: str) -> List[str]:
    """
    Recursively search for JSON files in the given root directory.

    Args:
        root_dir (str): The root directory to start the search from.

    Returns:
        List[str]: The absolute path of each JSON file found.
    """
    json_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


def count_json_files(root_dir: str) -> int:
    """
    Count the number of JSON files in the given root directory.

    Args:
        root_dir (str): The root directory to search for JSON files.

    Returns:
        int: The number of JSON files found.
    """
    return len(find_json_files(root_dir))


def process_directory(root_dir: str) -> pd.DataFrame:
    """
    Process all JSON files in the specified directory and return a merged DataFrame.

    Args:
        root_dir (str): The root directory containing the JSON files.

    Returns:
        pandas.DataFrame: The merged DataFrame containing the data from all JSON files.
    """
    with Pool() as pool:
        results = list(
            tqdm(pool.imap(process_json_file,
                           find_json_files(root_dir),
                           chunksize=10),
                 desc="Processing JSON Files",
                 total=count_json_files(root_dir)))

    merged_dataframe = pd.concat(results, ignore_index=True)
    return merged_dataframe


if __name__ == '__main__':
    # Specify the directory path
    DIRECTORY_PATH = 'xxx'
    result_dataframe = process_directory(DIRECTORY_PATH)
    SAVE_DIR = 'xxx'

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Save the result DataFrame as a CSV file
    SAVE_PATH = os.path.join(SAVE_DIR, 'CHON500noSalt_1million_1.csv')
    result_dataframe.to_csv(SAVE_PATH, index=False)

    # Specify the directory path
    DIRECTORY_PATH = 'xxx'
    result_dataframe = process_directory(DIRECTORY_PATH)
    # Save the result DataFrame as a CSV file
    SAVE_PATH = os.path.join(SAVE_DIR, 'CHON500noSalt_1million_2.csv')
    result_dataframe.to_csv(SAVE_PATH, index=False)
