import os
from functools import partial

import numpy as np
import pandas as pd
import scipy.constants as sc
from tqdm import tqdm

tqdm.pandas()

import scipy.constants as sc


def energy_to_boltzmann_prob(df_energy_idx: pd.DataFrame, unit: str,
                             T: float) -> pd.Series:
    """
    Calculate the probability of existence based on the Boltzmann distribution.

    Args:
        df_energy_idx (pd.DataFrame): DataFrame with energy and molecule index columns.
        unit (str): Energy unit ('ev', 'J', 'kcal/mol').
        T (float): Temperature in Kelvin.

    Returns:
        pd.Series: Probability values for each molecule index.
    """
    ene_col, idx_col = df_energy_idx.columns

    func_logBdist = partial(boltzmann_distribution,
                            unit=unit,
                            T=T,
                            logarithm=True)
    logQ = df_energy_idx[ene_col].progress_apply(func_logBdist)
    cat_inf = pd.concat([df_energy_idx, logQ], axis=1)
    cat_inf.columns = [ene_col, idx_col, 'logQ']  # rename for columns
    p_values = cat_inf.groupby(idx_col)['logQ'].transform(exponential_prob)
    return p_values


def boltzmann_distribution(energy: np.ndarray,
                           unit: str = 'energy_of_formation',
                           T: float = 298.15,
                           logarithm: bool = True) -> np.ndarray:
    """
    Calculate the Boltzmann distribution for a given energy array.

    Parameters:
    energy (np.ndarray): Array of energy values.
    unit (str, optional): Unit of energy. Default is 'energy_of_formation'.
    T (float, optional): Temperature in Kelvin. Default is 298.15.
    logarithm (bool, optional): If True, calculate the logarithm of the distribution.
                                If False, calculate the exponential of the distribution. 
                                Default is True.

    Returns:
    np.ndarray: Array of distribution values.

    Raises:
    ValueError: If the energy unit is not defined.

    """
    match unit:
        case 'ev':
            k = sc.k / sc.elementary_charge  # /1.60218e-19
        case 'J':
            k = sc.k
        case 'kcal/mol':
            k = sc.R * 0.001 * 0.239
        case _:
            raise ValueError(f'energy unit {unit} is not defined.')

    #energy = energy.astype(float)
    if logarithm:
        distributions = -energy / k / T
    else:
        distributions = np.exp(-energy / k / T)

    return distributions


def exponential_prob(log_values: pd.Series) -> pd.Series:
    """
    Calculates the exponential probability for each value in the given series.

    Args:
        log_values (pd.Series): The input series of logarithmic values.

    Returns:
        pd.Series: The series of exponential probabilities.

    Raises:
        ValueError: If the index of the series is not unique.
    """
    prob = pd.Series(index=log_values.index, name='probability')
    if len(np.unique(log_values.index)) != len(log_values):
        raise ValueError('Index must be unique for calculating B-dist weights')
    for idx in log_values.index:
        x = log_values[idx]
        log_contri = log_values - x
        denomitor = np.exp(log_contri)  # array
        v = 1. / np.sum(denomitor)
        prob.at[idx] = v
    return prob


def agg_bolzmann_weighted_descs(df_data: pd.DataFrame, idx: str,
                                p_value: str) -> pd.DataFrame:
    """
    Calculates the Boltzmann-weighted descriptors based on the given data.

    Args:
        df_data (pd.DataFrame): The input data containing descriptors and p-values.
        idx (str): The column name for the index.
        p_value (str): The column name for the p-value.

    Returns:
        pd.DataFrame: The calculated Boltzmann-weighted descriptors.

    Raises:
        KeyError: If the specified column names are not found in the input data.
    """
    # Drop unnecessary columns
    df_descs = df_data.drop(columns=['cid', p_value])
    # Multiply descriptors by p-values
    desc_weight = df_descs.astype(float).mul(df_data[p_value], axis='index')
    # Concatenate descriptors with index column
    desc_weight = pd.concat([desc_weight, df_data[idx]], axis=1)
    # Sum the weighted descriptors by index
    desc_weight_sum = desc_weight.groupby(idx).sum()
    return desc_weight_sum


def agg_mean_weight_descs(df_data: pd.DataFrame, idx: str,
                          p_value: str) -> pd.DataFrame:
    """
    Calculates the mean-weighted descriptors based on the given data.

    Args:
        df_data (pd.DataFrame): The input data containing descriptors and p-values.
        idx (str): The column name for the index.
        p_value (str): The column name for the p-value.

    Returns:
        pd.DataFrame: The calculated mean-weighted descriptors.

    Raises:
        KeyError: If the specified column names are not found in the input data.
    """
    # Drop unnecessary columns
    df_descs = df_data.drop(columns=[p_value])
    # Calculate mean-weighted descriptors
    desc_mean_weighted = df_descs.groupby(idx).mean()
    return desc_mean_weighted


def agg_max_existing_prob_descs(df_data: pd.DataFrame, idx: str,
                                p_value: str) -> pd.DataFrame:
    """
    Selects the descriptors with the maximum probability value for each index.

    Args:
        df_data (pd.DataFrame): The input data containing descriptors and p-values.
        idx (str): The column name for the index.
        p_value (str): The column name for the p-value.

    Returns:
        pd.DataFrame: The selected descriptors with the maximum probability value for each index.

    Raises:
        KeyError: If the specified column names are not found in the input data.

    Note:
        select most energitically favord one, but not global minimum maybe...
        if some p-value same, select multiple rows.
    """
    # Drop unnecessary columns
    df_descs = df_data.drop(columns=[p_value])
    # Select descriptors with the maximum probability value for each index
    idx_max = df_data.groupby(idx)[p_value].transform(
        'max') == df_data[p_value]
    desc_max = df_descs[idx_max]
    desc_max.set_index(idx, inplace=True)
    # Delete duplicates
    duplicate_counts = desc_max.index.value_counts()
    # Print indices with duplicate values
    print('Duplicate of desc_max',
          duplicate_counts[duplicate_counts >= 2].index.tolist())
    desc_max = desc_max[~desc_max.index.duplicated(keep='first')]
    return desc_max


def agg_random_descs(df_data: pd.DataFrame, idx: str,
                     p_value: str) -> pd.DataFrame:
    """
    Selects a random descriptor for each index.

    Args:
        df_data (pd.DataFrame): The input data containing descriptors and p-values.
        idx (str): The column name for the index.
        p_value (str): The column name for the p-value.

    Returns:
        pd.DataFrame: The randomly selected descriptors for each index.
    """
    # Drop unnecessary columns
    df_descs = df_data.drop(columns=[p_value])
    # Select a random descriptor for each index
    desc_random = df_descs.groupby(idx).apply(
        lambda x: x.sample(n=1, random_state=42))
    desc_random.set_index(idx, inplace=True)
    return desc_random


def agg_rmsd_descs(df_data: pd.DataFrame,
                   idx: str,
                   p_value: str,
                   transform: str = 'max') -> pd.DataFrame:
    """
    Selects RMSD descriptors based on the specified transform.

    Args:
        df_data (pd.DataFrame): The input data containing descriptors and p-values.
        idx (str): The column name for the index.
        p_value (str): The column name for the p-value.
        transform (str, optional): The transform to apply for selecting RMSD descriptors. Defaults to 'max'.

    Returns:
        pd.DataFrame: The selected RMSD descriptors.
    """
    # select RMSD nearest & farthest to correct molecule coordinates
    # Drop unnecessary columns
    df_data = df_data.drop(columns=[p_value])
    # Select RMSD descriptors based on the specified transform
    idx_rmsd = df_data.groupby(idx)['rmsd'].transform(
        transform) == df_data['rmsd']
    desc_rmsd = df_data[idx_rmsd]
    desc_rmsd.set_index(idx, inplace=True)
    # delte duplicates
    duplicate_counts = desc_rmsd.index.value_counts()
    # Print indices with duplicate values
    print(f'Duplicate of desc_rmsd_{transform}',
          duplicate_counts[duplicate_counts >= 2].index.tolist())
    desc_rmsd = desc_rmsd[~desc_rmsd.index.duplicated(keep='first')]
    return desc_rmsd


def calcualte_aggregated_descriptors(data_path: str, save_dir: str) -> None:
    """
    Calculate and save aggregated descriptors.

    Args:
        data_path (str): The path to the data file.
        save_dir (str): The directory to save the aggregated descriptors.
    """
    idx = 'cid'
    p_value = 'p_value'

    df_data = pd.read_csv(data_path)
    # df_data = pd.read_csv(data_path, nrows=100)  # for small-data testing

    df_data[idx] = df_data[idx].apply(lambda x: x.split('_')[0])
    print('Calc. BoltzProb from existing energy data.')
    df_data[p_value] = energy_to_boltzmann_prob(df_data[['total_energy', idx]],
                                                T=298,
                                                unit='kcal/mol')

    # select descriptors want to aggregate
    descs_columns = [idx, p_value] + [
        'RotatableBonds', 'rmsd', 'total_energy'
    ] + df_data.loc[:, 'ASA':'vsurf_Wp8'].columns.to_list()
    df_data = df_data.loc[:, descs_columns]

    # Calculate and save statistics of conformer probabilities.
    # p_value_describe = df_data.loc[:,['cid', 'p_value']].groupby('cid').describe()
    # p_value_describe.to_csv(os.path.join(save_dir, 'p_value_describe.tsv'), sep='\t')

    print('Calc. aggregated descs.')
    # Bolzmann weight
    desc_weight_mean = agg_bolzmann_weighted_descs(df_data, idx, p_value)
    # equal weight
    desc_eq_mean = agg_mean_weight_descs(df_data, idx, p_value)
    # most energitically favord one.
    desc_max = agg_max_existing_prob_descs(df_data, idx, p_value)
    # randomly select one conformer
    desc_random = agg_random_descs(df_data, idx, p_value)

    # select RMSD nearest & farthest to correct molecule coordinates
    desc_rmsd_min = agg_rmsd_descs(df_data, idx, p_value, transform='min')
    desc_rmsd_max = agg_rmsd_descs(df_data, idx, p_value, transform='max')

    print('Confirming that all shapes are the same:', desc_weight_mean.shape,
          desc_eq_mean.shape, desc_max.shape, desc_random.shape,
          desc_rmsd_max.shape, desc_rmsd_min.shape)

    print('Saving aggregated descriptors to CSV')
    desc_weight_mean.to_csv(os.path.join(save_dir, 'HFweight.tsv'), sep='\t')
    desc_eq_mean.to_csv(os.path.join(save_dir, 'eq_weight.tsv'), sep='\t')
    desc_max.to_csv(os.path.join(save_dir, '1conf.tsv'), sep='\t')
    desc_random.to_csv(os.path.join(save_dir, 'random.tsv'), sep='\t')
    desc_rmsd_min.to_csv(os.path.join(save_dir, 'rmsd_min.tsv'), sep='\t')
    desc_rmsd_max.to_csv(os.path.join(save_dir, 'rmsd_max.tsv'), sep='\t')


if __name__ == '__main__':
    DATA_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    calcualte_aggregated_descriptors(DATA_PATH, SAVE_DIR)
