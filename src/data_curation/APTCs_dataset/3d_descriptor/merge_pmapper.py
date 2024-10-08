import os
from functools import partial

import numpy as np
import pandas as pd
import scipy.constants as sc
from tqdm import tqdm

tqdm.pandas()

import scipy.constants as sc


def concat_pmapper_aptc(save_dir, dataset='aptc-1'):
    '''
    Read the representations generated by Pmapper
    Read the activity
    Read the energy
    Read the conf_id
    Merge everything together
    '''
    ### read pmapper ###
    # pmapper is already calulated in this path
    data_path = '3D-MIL-QSSR/output/240603_catalyst'
    x_train = np.load(f'{data_path}/{dataset}/train/descriptors/x_train.npy',
                      allow_pickle=True)
    x_test = np.load(f'{data_path}/{dataset}/test/descriptors/x_test.npy',
                     allow_pickle=True)
    # 3d to 2d array for concat
    pmapper_train = np.concatenate(
        [sub_array.reshape(-1, sub_array.shape[-1]) for sub_array in x_train])
    pmapper_test = np.concatenate(
        [sub_array.reshape(-1, sub_array.shape[-1]) for sub_array in x_test])
    ###################

    ### read activity ###
    if dataset == 'aptc-1':
        training_dataset_path = '3D-MIL-QSSR/datasets/aptc-1/aptc_1_training_datatest.csv'
        test_dataset_path = '3D-MIL-QSSR/datasets/aptc-1/aptc_1_test_datatest.csv'
    elif dataset == 'aptc-2':
        training_dataset_path = '3D-MIL-QSSR/datasets/aptc-2/aptc_2_training_datatest.csv'
        test_dataset_path = '3D-MIL-QSSR/datasets/aptc-2/aptc_2_test_datatest.csv'
    train_data = pd.read_csv(training_dataset_path, usecols=['ACTIVITY'])
    test_data = pd.read_csv(test_dataset_path, usecols=['ACTIVITY'])
    ###################

    ### read energy ###
    if dataset == 'aptc-1':
        energy_train_path = '2d3d/dataset/APTC/step2/aptc1_train.csv'
        energy_test_path = '2d3d/dataset/APTC/step2/aptc1_test.csv'
    elif dataset == 'aptc-2':
        energy_train_path = '2d3d/dataset/APTC/step2/aptc2_train.csv'
        energy_test_path = '2d3d/dataset/APTC/step2/aptc2_test.csv'
    energy_train = pd.read_csv(energy_train_path,
                               usecols=['cid', 'MMFF_Energy'])
    energy_test = pd.read_csv(energy_test_path, usecols=['cid', 'MMFF_Energy'])

    ###################

    ### read conf_id ###
    conf_id_train = pd.read_csv(
        f'{data_path}/{dataset}/train/descriptors/catalyst_descriptors.rownames',
        header=None)
    conf_id_test = pd.read_csv(
        f'{data_path}/{dataset}/test/descriptors/catalyst_descriptors.rownames',
        header=None)
    # Add 'comp_id' column to the dataframes
    conf_id_train['comp_id'] = conf_id_train[0].str.split('_').str[0].astype(
        int)
    conf_id_test['comp_id'] = conf_id_test[0].str.split('_').str[0].astype(int)
    conf_id_train['conf_id'] = conf_id_train[0].str.split('_').str[1].astype(
        int)
    conf_id_test['conf_id'] = conf_id_test[0].str.split('_').str[1].astype(int)
    # Sort by 'comp_id' column in descending order
    conf_id_train = conf_id_train.sort_values('comp_id')
    conf_id_test = conf_id_test.sort_values('comp_id')
    conf_id_train.rename(columns={0: 'comp_conf_id'}, inplace=True)
    conf_id_test.rename(columns={0: 'comp_conf_id'}, inplace=True)
    ###################

    ### merge ###
    # merge conf id and activity
    merge_train = train_data.reset_index().merge(
        conf_id_train, left_on='index',
        right_on='comp_id').drop(columns='index')
    merge_test = test_data.reset_index().merge(
        conf_id_test, left_on='index',
        right_on='comp_id').drop(columns='index')
    # merge pmapper
    pmapper_train_df = pd.DataFrame(pmapper_train)
    pmapper_test_df = pd.DataFrame(pmapper_test)
    merge_train = pd.concat(
        [merge_train.reset_index(drop=True), pmapper_train_df], axis=1)
    merge_test = pd.concat(
        [merge_test.reset_index(drop=True), pmapper_test_df], axis=1)
    # merge energy
    merge_train = merge_train.merge(energy_train,
                                    left_on='comp_conf_id',
                                    right_on='cid').drop(columns='cid')
    merge_test = merge_test.merge(energy_test,
                                  left_on='comp_conf_id',
                                  right_on='cid').drop(columns='cid')
    ###################

    ### save ###
    save_dir = f'{save_dir}/{dataset}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    merge_train.to_csv(f'{save_dir}/pmapper_train.csv', index=False)
    merge_test.to_csv(f'{save_dir}/pmapper_test.csv', index=False)
    ###################


if __name__ == '__main__':
    SAVE_DIR = 'xxx'
    datasets = ['aptc-1', 'aptc-2']
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for dataset in datasets:
        concat_pmapper_aptc(SAVE_DIR, dataset=dataset)
