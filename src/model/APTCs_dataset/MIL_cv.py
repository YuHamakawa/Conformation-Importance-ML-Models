import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from descriptor_calculation.pmapper_3d import (convert_pkl_to_sdf,
                                               select_min_energy)
from estimators.attention_nets import AttentionNetRegressor
from estimators.mi_nets import BagNetRegressor, InstanceNetRegressor
from estimators.wrappers import (BagWrapperMLPRegressor,
                                 InstanceWrapperMLPRegressor)
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import (calc_catalyst_descr, calc_pm6_descr, calc_reaction_descr,
                   concat_react_cat_descr, extract_cat_descr,
                   gen_catalyst_confs, read_input_data, scale_data)

logging.basicConfig(format='%(message)s', level=logging.NOTSET)


def calcMetrics(true, pred):
    '''culculate R2, RMSE(root mean square error), MAE(mean absolute error)
    
    Args: 
        true : true label
        pred : model prediction
    Returns: 
        r2 : r2_score
        mse : mean_squared_error
        mae : mean_absolute_error
    '''
    r2 = r2_score(true, pred)
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    return r2, mse, mae


def select_dataset(desc_set, agg, dataset):
    '''
    Function to select which dataset to use for analysis
    desc_set: '2d', 'moe', 'pmapper', 'unimol', 'unirep'
    dataset: 'aptc1', 'aptc2'
    '''
    base_dir = '2d3d/dataset/APTC/step7'
    if desc_set == 'moe' or desc_set == 'pmapper':
        if agg == 'Boltzman weight':
            df = pd.read_csv(f'{base_dir}/{desc_set}/{dataset}/HFweight.tsv',
                             sep='\t')
        elif agg == 'mean':
            df = pd.read_csv(f'{base_dir}/{desc_set}/{dataset}/eq_weight.tsv',
                             sep='\t')
        elif agg == 'global minimum':
            df = pd.read_csv(f'{base_dir}/{desc_set}/{dataset}/1conf.tsv',
                             sep='\t')
        elif agg == 'random':
            df = pd.read_csv(f'{base_dir}/{desc_set}/{dataset}/random.tsv',
                             sep='\t')
        elif agg == 'random_12':
            df = pd.read_csv(f'{base_dir}/{desc_set}/{dataset}/random_12.tsv',
                             sep='\t')
        elif agg == 'random_22':
            df = pd.read_csv(f'{base_dir}/{desc_set}/{dataset}/random_22.tsv',
                             sep='\t')
        elif agg == 'random_32':
            df = pd.read_csv(f'{base_dir}/{desc_set}/{dataset}/random_32.tsv',
                             sep='\t')
        elif agg == 'random_52':
            df = pd.read_csv(f'{base_dir}/{desc_set}/{dataset}/random_52.tsv',
                             sep='\t')
        elif agg == 'no_agg':
            df = pd.read_csv(
                f'{base_dir}/{desc_set}/{dataset}/descs_no_agg.csv')
        else:
            raise ValueError(f'agg is invalid for desc_sets {desc_set}.')
    elif desc_set == '2d':
        if agg == 'ecfp_bit' or agg == 'fcfp_bit' or agg == 'ecfp_count' or agg == 'fcfp_count' or agg == 'pharm_2d':
            df = pd.read_csv(
                f'{base_dir}/{agg}/{dataset}/descs_{desc_set}.tsv', sep='\t')
        else:
            raise ValueError(f'agg is invalid for desc_sets {desc_set}.')
    elif desc_set == 'unirep':
        if agg == 'global minimum':
            df = pd.read_csv(f'{base_dir}/{desc_set}/{dataset}/1conf.csv')
        elif agg == 'no_agg':
            df = pd.read_csv(
                f'{base_dir}/{desc_set}/{dataset}/descs_no_agg.csv')
        else:
            raise ValueError(f'agg is invalid for desc_sets {desc_set}.')

    return df


def split_x_y(desc_tr, desc_te, desc_set, dataset, agg):
    '''
    Function to split the Train data and Test data into features and target variables.
    '''

    if desc_set == 'moe':
        # All descs are 117 descs
        desc_list = desc_tr.loc[:, 'ASA':'vsurf_Wp8'].columns.to_list()
        y_name = 'ACTIVITY'
    elif desc_set == 'pmapper':
        if dataset == 'aptc1':
            desc_list = desc_tr.loc[:, '0':'1201'].columns.to_list()
            y_name = 'ACTIVITY'
        elif dataset == 'aptc2':
            desc_list = desc_tr.loc[:, '0':'1555'].columns.to_list()
            y_name = 'ACTIVITY'
    elif desc_set == '2d':
        if agg == 'pharm_2d':
            desc_list = desc_tr.loc[:, '0':'2047'].columns.to_list()
        else:
            desc_list = desc_tr.loc[:, '0':'1023'].columns.to_list()
        y_name = 'ACTIVITY'
    elif desc_set == 'unirep':
        desc_list = desc_tr.loc[:, '0':'511'].columns.to_list()
        y_name = 'target'
    else:
        raise ValueError(f'desc_set is invalid. {desc_set}')

    if agg == 'no_agg':
        data_tr = desc_tr.groupby(by='cid').apply(
            lambda x: (x.loc[:, desc_list].to_numpy(), x[y_name].mean()))
        data_tr = pd.DataFrame(data_tr.tolist(),
                               index=data_tr.index,
                               columns=['Features', y_name]).reset_index()
        x_tr = data_tr['Features'].to_numpy()
        y_tr = data_tr[y_name].to_numpy()

        data_te = desc_te.groupby(by='cid').apply(
            lambda x: (x.loc[:, desc_list].to_numpy(), x[y_name].mean()))
        data_te = pd.DataFrame(data_te.tolist(),
                               index=data_te.index,
                               columns=['Features', y_name]).reset_index()
        x_te = data_te['Features'].to_numpy()
        y_te = data_te[y_name].to_numpy()
    else:
        x_tr = desc_tr.loc[:, desc_list].to_numpy().reshape(
            desc_tr.shape[0], 1, -1)
        x_te = desc_te.loc[:, desc_list].to_numpy().reshape(
            desc_te.shape[0], 1, -1)
        y_tr = desc_tr.loc[:, y_name].to_numpy()
        y_te = desc_te.loc[:, y_name].to_numpy()

    x_train_scaled, x_test_scaled = scale_data(x_tr, x_te)

    return x_train_scaled, x_test_scaled, y_tr, y_te


def save_pred(save_dir, tr_cid, te_cid, y_tr, y_te, tr_pred, te_pred):
    '''
    Function to save the predictions of the model for each fold as a dataframe
    '''
    # create df and save train_cid, y_pred_train, y_train
    pred_tr = pd.DataFrame({
        'cid': tr_cid,
        'predicted': tr_pred,
        'actual': y_tr
    })
    pred_tr.to_csv(os.path.join(save_dir, 'train_set_predictions.csv'),
                   index=False)
    # create df and save test_cid, y_pred_test, y_test
    pred_te = pd.DataFrame({
        'cid': te_cid,
        'predicted': te_pred,
        'actual': y_te
    })
    pred_te.to_csv(os.path.join(save_dir, 'test_set_predictions.csv'),
                   index=False)

    # calculate metrics
    r2_train, mse_train, mae_train = calcMetrics(y_tr, tr_pred)
    r2_test, mse_test, mae_test = calcMetrics(y_te, te_pred)

    # add results to a dataframe
    result_df = pd.DataFrame(
        {
            'R2 Train': r2_train,
            'R2 Test': r2_test,
            'MSE Train': mse_train,
            'MSE Test': mse_test,
            'MAE Train': mae_train,
            'MAE Test': mae_test
        },
        index=[0])

    result_df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)

    return result_df


def build_model(x_train_scaled,
                x_test_scaled,
                activity_train,
                algorithm,
                test_mode=False):
    '''
    build model and predict
    '''
    n_epoch = 500
    lr = 0.001
    weight_decay = 0.0001
    batch_size = 128

    if test_mode:
        n_epoch = 1

    # train model
    logging.info(f'Model training ...')

    algorithms = {
        'BagWrapperMLPRegressor': BagWrapperMLPRegressor,
        'InstanceWrapperMLPRegressor': InstanceWrapperMLPRegressor,
        'BagNetRegressor': BagNetRegressor,
        'InstanceNetRegressor': InstanceNetRegressor,
        'AttentionNetRegressor': AttentionNetRegressor
    }
    #
    init_cuda = torch.cuda.is_available()
    #
    n_dim = [x_test_scaled[0].shape[-1]] + [256, 128, 64]

    if algorithm == 'AttentionNetRegressor':
        det_ndim = n_dim
        net = algorithms[algorithm](ndim=n_dim,
                                    det_ndim=det_ndim,
                                    init_cuda=init_cuda)
        net.fit(
            x_train_scaled,
            activity_train,
            n_epoch=n_epoch,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            #use for gumbel softmax that is culculated in attention_nets.py
            dropout=1,
            #verbose=True
        )

    else:
        net = algorithms[algorithm](ndim=n_dim,
                                    pool='mean',
                                    init_cuda=init_cuda)
        net.fit(
            x_train_scaled,
            activity_train,
            n_epoch=n_epoch,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            #verbose=True
        )

    tr_pred = net.predict(x_train_scaled).ravel()
    te_pred = net.predict(x_test_scaled).ravel()

    return tr_pred, te_pred


def run_cv(save_dir,
           desc_set,
           agg,
           dataset,
           algorithm,
           test_mode,
           cv_seed=None):
    '''
    Function to perform cross-validation:
    1. Read the data.
    2. Load the pre-defined indices for train-test split.
    3. Split the data into features and target variables.
    4. Build the model and save the results for each fold.
    5. Combine and save the results from all folds.
    
    Args:
        cv_seed: Specify which seed to use for cross-validation in APTC1.
    '''
    df = select_dataset(desc_set, agg, dataset)

    index_dir = '2d3d/dataset/APTC/step7/index'

    if dataset == 'aptc1':
        cv_split = 5
    elif dataset == 'aptc2':
        cv_split = 40
    if test_mode:
        cv_split = 2

    # Create a dataframe to record the results of all folds
    results_df = pd.DataFrame(columns=[
        'R2 Train', 'R2 Test', 'MSE Train', 'MSE Test', 'MAE Train', 'MAE Test'
    ])
    for fold in range(cv_split):
        save_dir_each_fold = f'{save_dir}/fold_{fold}'
        os.makedirs(save_dir_each_fold, exist_ok=True)
        if dataset == 'aptc1':
            tr_cid = np.load(
                f'{index_dir}/{dataset}/{cv_seed}/train_{fold}.npy')
            te_cid = np.load(
                f'{index_dir}/{dataset}/{cv_seed}/test_{fold}.npy')
        elif dataset == 'aptc2':
            tr_cid = np.load(f'{index_dir}/{dataset}/train_{fold}.npy')
            te_cid = np.load(f'{index_dir}/{dataset}/test_{fold}.npy')

        desc_tr = df[df['cid'].isin(tr_cid)]
        desc_te = df[df['cid'].isin(te_cid)]
        x_tr, x_te, y_tr, y_te = split_x_y(desc_tr, desc_te, desc_set, dataset,
                                           agg)
        tr_pred, te_pred = build_model(x_tr, x_te, y_tr, algorithm, test_mode)

        df_append = save_pred(save_dir_each_fold, tr_cid, te_cid, y_tr, y_te,
                              tr_pred, te_pred)
        results_df = pd.concat([results_df, df_append],
                               axis=0,
                               ignore_index=True)

    results_df.to_csv(f'{save_dir}/results.csv', index=False)

    if dataset == 'aptc1':
        # calculate mean and std for all folds
        mean_std_df = pd.DataFrame({
            f'Mean_{desc_set}_{dataset}_{cv_seed}_{agg}_{algorithm}':
            results_df.mean(),
            f'Std_{desc_set}_{dataset}_{cv_seed}_{agg}_{algorithm}':
            results_df.std()
        }).T
    elif dataset == 'aptc2':
        mean_std_df = pd.DataFrame({
            f'Mean_{desc_set}_{dataset}_{agg}_{algorithm}':
            results_df.mean(),
            f'Std_{desc_set}_{dataset}_{agg}_{algorithm}':
            results_df.std()
        }).T
    mean_std_df.to_csv(f'{save_dir}/mean_std.csv', index=True)
    return mean_std_df


def unite_all_csv(root_dir, filename='mean_std.csv'):
    '''
    Function to combine all the calculated results.
    '''
    root_path = Path(root_dir)
    csv_files = root_path.rglob(filename)
    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.rename(columns={'Unnamed: 0': 'Name'}, inplace=True)
    combined_df.to_csv(f'{root_path}/all_mean_std.csv', index=False)
    return combined_df


if __name__ == '__main__':
    SAVE_DIR = '3D-MIL-QSSR/output/240806_aptc_MIL'
    TEST_MODE = False
    datasets = ['aptc1', 'aptc2']
    desc_sets = ['2d', 'moe', 'pmapper', 'unirep']

    aggs_moe_pmapper = [
        'Boltzman weight', 'mean', 'global minimum', 'random', 'no_agg',
        'random_12', 'random_22', 'random_32', 'random_52'
    ]
    aggs_2d = ['ecfp_bit', 'ecfp_count', 'fcfp_bit', 'fcfp_count', 'pharm_2d']
    aggs_unirep = ['global minimum', 'no_agg']

    APTC1_seeds = ['seed_12', 'seed_22', 'seed_32', 'seed_42', 'seed_52']

    algorithms = [
        'InstanceWrapperMLPRegressor', 'BagWrapperMLPRegressor',
        'InstanceNetRegressor', 'BagNetRegressor', 'AttentionNetRegressor'
    ]

    desc_sets = ['2d']
    aggs_2d = ['pharm_2d']

    if TEST_MODE:
        SAVE_DIR = f'{SAVE_DIR}_test'
        # desc_sets = ['moe']
        # datasets = ['aptc2']
        APTC1_seeds = ['seed_12']
        aggs_moe_pmapper = ['mean', 'no_agg']
        algorithms = ['InstanceNetRegressor', 'BagNetRegressor']

    # unite_all_csv(SAVE_DIR, filename='mean_std.csv')

    if len(sys.argv) == 2:
        match int(sys.argv[1]):
            case 1:
                desc_sets = ['2d']
            case 2:
                desc_sets = ['moe']
            case 3:
                desc_sets = ['pmapper']
            case 4:
                desc_sets = ['unirep']
    '''
    Directory hierarchy
    APTC-1:
    desc_set -> aptc1 -> seed -> algorithm -> agg -> 5fold -> pred for each fold

    APTC-2:
    desc_set -> aptc2 -> algorithm -> agg -> 40fold(LOO) -> pred for each fold
    '''

    for desc_set in desc_sets:
        for dataset in datasets:
            if dataset == 'aptc1':
                for seed in APTC1_seeds:
                    for algorithm in algorithms:
                        if desc_set == 'moe' or desc_set == 'pmapper':
                            for agg in aggs_moe_pmapper:
                                save_dir_each = f'{SAVE_DIR}/{desc_set}/{dataset}/{seed}/{algorithm}/{agg}'
                                os.makedirs(save_dir_each, exist_ok=True)
                                run_cv(save_dir_each,
                                       desc_set,
                                       agg,
                                       dataset,
                                       algorithm,
                                       TEST_MODE,
                                       cv_seed=seed)
                        elif desc_set == '2d':
                            for agg in aggs_2d:
                                save_dir_each = f'{SAVE_DIR}/{desc_set}/{dataset}/{seed}/{algorithm}/{agg}'
                                os.makedirs(save_dir_each, exist_ok=True)
                                run_cv(save_dir_each,
                                       desc_set,
                                       agg,
                                       dataset,
                                       algorithm,
                                       TEST_MODE,
                                       cv_seed=seed)
                        elif desc_set == 'unirep':
                            for agg in aggs_unirep:
                                save_dir_each = f'{SAVE_DIR}/{desc_set}/{dataset}/{seed}/{algorithm}/{agg}'
                                os.makedirs(save_dir_each, exist_ok=True)
                                run_cv(save_dir_each,
                                       desc_set,
                                       agg,
                                       dataset,
                                       algorithm,
                                       TEST_MODE,
                                       cv_seed=seed)

            elif dataset == 'aptc2':
                for algorithm in algorithms:
                    if desc_set == 'moe' or desc_set == 'pmapper':
                        for agg in aggs_moe_pmapper:
                            save_dir_each = f'{SAVE_DIR}/{desc_set}/{dataset}/{algorithm}/{agg}'
                            os.makedirs(save_dir_each, exist_ok=True)
                            run_cv(save_dir_each, desc_set, agg, dataset,
                                   algorithm, TEST_MODE)

                    elif desc_set == '2d':
                        for agg in aggs_2d:
                            save_dir_each = f'{SAVE_DIR}/{desc_set}/{dataset}/{algorithm}/{agg}'
                            os.makedirs(save_dir_each, exist_ok=True)
                            run_cv(save_dir_each, desc_set, agg, dataset,
                                   algorithm, TEST_MODE)
                    elif desc_set == 'unirep':
                        for agg in aggs_unirep:
                            save_dir_each = f'{SAVE_DIR}/{desc_set}/{dataset}/{algorithm}/{agg}'
                            os.makedirs(save_dir_each, exist_ok=True)
                            run_cv(save_dir_each, desc_set, agg, dataset,
                                   algorithm, TEST_MODE)
