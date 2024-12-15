import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from estimators.attention_nets import AttentionNetRegressor
from estimators.mi_nets import BagNetRegressor, InstanceNetRegressor
from estimators.wrappers import (BagWrapperMLPRegressor,
                                 InstanceWrapperMLPRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import scale_data

logger = logging.getLogger(__name__)


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


def select_dataset(desc_set, agg, test_mode):
    '''
    Function to select which dataset to use for analysis.
    desc_set: '2d', 'moe', 'pmapper', 'morse', 'mbtr'
    '''
    if desc_set == 'moe' or desc_set == 'pmapper' or desc_set == 'morse' or desc_set == 'mbtr':
        base_dir = 'xxx'
        if agg == 'Boltzman weight':
            df = pd.read_csv(f'{base_dir}/boltzmann_weight.tsv', sep='\t')
        elif agg == 'mean':
            df = pd.read_csv(f'{base_dir}/mean.tsv', sep='\t')
        elif agg == 'global minimum':
            df = pd.read_csv(f'{base_dir}/global_minimum.tsv', sep='\t')
        elif agg == 'random':
            df = pd.read_csv(f'{base_dir}/random.tsv', sep='\t')
        elif agg == 'no_agg':
            df = pd.read_csv(f'{base_dir}/non_aggregation.csv')
        else:
            raise ValueError(f'agg is invalid for desc_sets {desc_set}.')
    elif desc_set == '2d':
        base_dir = 'xxx'
        if agg == 'ecfp4_count':
            df = pd.read_csv(f'{base_dir}/ecfp4_count_195.tsv', sep='\t')
        else:
            raise ValueError(f'agg is invalid for desc_sets {desc_set}.')

    # if test_mode:
    #     df = df.sample(n=10, random_state=0)
    return df


def split_x_y(desc_tr, desc_te, desc_set, agg):
    '''
    Function to split the given train and test data into features and target variable.
    '''

    Y_NAME = 'mpC'

    if desc_set == 'moe':  # 117 descs
        desc_list = desc_tr.loc[:, 'ASA':'vsurf_Wp8'].columns.to_list()
    elif desc_set == 'pmapper':  # 575 descs
        desc_list = desc_tr.loc[:, 'pmapper_0':'pmapper_574'].columns.to_list()
    elif desc_set == 'morse':  # 160 descs
        desc_list = desc_tr.loc[:, 'Mor01':'Mor32p'].columns.to_list()
    elif desc_set == 'mbtr':  # 950 descs
        desc_list = desc_tr.loc[:, 'MBTR_0':'MBTR_949'].columns.to_list()
    elif desc_set == '2d':  # ecfp4 count 2048 dim
        desc_list = desc_tr.loc[:, '0':'2047'].columns.to_list()
    else:
        raise ValueError(f'desc_set is invalid. {desc_set}')

    if agg == 'no_agg':
        data_tr = desc_tr.groupby(by='csid').apply(
            lambda x: (x.loc[:, desc_list].to_numpy(), x[Y_NAME].mean()))
        data_tr = pd.DataFrame(data_tr.tolist(),
                               index=data_tr.index,
                               columns=['Features', Y_NAME]).reset_index()
        x_tr = data_tr['Features'].to_numpy()
        y_tr = data_tr[Y_NAME].to_numpy()

        data_te = desc_te.groupby(by='csid').apply(
            lambda x: (x.loc[:, desc_list].to_numpy(), x[Y_NAME].mean()))
        data_te = pd.DataFrame(data_te.tolist(),
                               index=data_te.index,
                               columns=['Features', Y_NAME]).reset_index()
        x_te = data_te['Features'].to_numpy()
        y_te = data_te[Y_NAME].to_numpy()
    else:
        x_tr = desc_tr.loc[:, desc_list].to_numpy().reshape(
            desc_tr.shape[0], 1, -1)
        x_te = desc_te.loc[:, desc_list].to_numpy().reshape(
            desc_te.shape[0], 1, -1)
        y_tr = desc_tr.loc[:, Y_NAME].to_numpy()
        y_te = desc_te.loc[:, Y_NAME].to_numpy()

    x_train_scaled, x_test_scaled = scale_data(x_tr, x_te)

    return x_train_scaled, x_test_scaled, y_tr, y_te


def save_pred(save_dir, tr_cid, te_cid, y_tr, y_te, tr_pred, te_pred):
    '''
    Function to save the model's prediction results for each fold as a dataframe.
    '''
    # create df and save train_cid, y_pred_train, y_train
    pred_tr = pd.DataFrame({
        'csid': tr_cid,
        'predicted': tr_pred,
        'actual': y_tr
    })
    pred_tr.to_csv(os.path.join(save_dir, 'train_set_predictions.csv'),
                   index=False)
    # create df and save test_cid, y_pred_test, y_test
    pred_te = pd.DataFrame({
        'csid': te_cid,
        'predicted': te_pred,
        'actual': y_te
    })
    pred_te.to_csv(os.path.join(save_dir, 'test_set_predictions.csv'),
                   index=False)

    # calculate metrics
    r2_train, mse_train, mae_train = calcMetrics(y_tr, tr_pred)
    r2_test, mse_test, mae_test = calcMetrics(y_te, te_pred)

    # 結果をデータフレームに追加
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
    logger.info(f'Model training ...')

    algorithms = {
        'BagWrapperMLPRegressor': BagWrapperMLPRegressor,
        'InstanceWrapperMLPRegressor': InstanceWrapperMLPRegressor,
        'BagNetRegressor': BagNetRegressor,
        'InstanceNetRegressor': InstanceNetRegressor,
        'AttentionNetRegressor': AttentionNetRegressor
    }
    #
    init_cuda = torch.cuda.is_available()
    logger.info(f'Init_cuda: {init_cuda}')
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


def run_cv(save_dir, desc_set, agg, algorithm, test_mode, cv_seed=None):
    '''
    Function to perform cross-validation.
    1. Load the data.
    2. Load pre-prepared indices to split into train and test sets.
    3. Split into features and target variable.
    4. Build the model for each fold and save the results.
    5. Combine and save the results for all folds.
    
    Args:
        cv_seed: Specify which seed's CV to perform on APTC1.
    '''

    logger.info(f'Running {desc_set} {agg} {algorithm} ...')

    df = select_dataset(desc_set, agg, test_mode)

    NUM_FOLDS = 5
    if test_mode:
        NUM_FOLDS = 2

    results_df = pd.DataFrame(columns=[
        'R2 Train', 'R2 Test', 'MSE Train', 'MSE Test', 'MAE Train', 'MAE Test'
    ])
    for fold in range(NUM_FOLDS):
        logger.info(f'fold = {fold+1} / {NUM_FOLDS}')
        save_dir_each_fold = f'{save_dir}/fold_{fold}'
        os.makedirs(save_dir_each_fold, exist_ok=True)

        INDEX_DIR = 'xxx'
        tr_csid = np.load(f'{INDEX_DIR}/seed_{cv_seed}/train_{fold}.npy')
        te_csid = np.load(f'{INDEX_DIR}/seed_{cv_seed}/test_{fold}.npy')
        desc_tr = df[df['csid'].isin(tr_csid)]
        desc_te = df[df['csid'].isin(te_csid)]
        x_tr, x_te, y_tr, y_te = split_x_y(desc_tr, desc_te, desc_set, agg)
        tr_pred, te_pred = build_model(x_tr, x_te, y_tr, algorithm, test_mode)

        df_append = save_pred(save_dir_each_fold, tr_csid.sort(),
                              te_csid.sort(), y_tr, y_te, tr_pred, te_pred)

        results_df = pd.concat([results_df, df_append],
                               axis=0,
                               ignore_index=True)

    results_df.to_csv(f'{save_dir}/results.csv', index=False)

    mean_std_df = pd.DataFrame({
        f'Mean_{cv_seed}_{desc_set}_{agg}':
        results_df.mean(),
        f'Std_{cv_seed}_{desc_set}_{agg}':
        results_df.std()
    }).T
    mean_std_df.to_csv(f'{save_dir}/mean_std.csv', index=True)
    return mean_std_df


if __name__ == '__main__':
    SAVE_DIR = 'xxx'
    TEST_MODE = False
    seeds = [12, 22, 32, 42, 52]
    algorithms = [
        'InstanceWrapperMLPRegressor', 'BagWrapperMLPRegressor',
        'InstanceNetRegressor', 'BagNetRegressor', 'AttentionNetRegressor'
    ]
    desc_sets = ['2d', 'moe', 'pmapper', 'morse', 'mbtr']
    aggs_3d = ['Boltzman weight', 'mean', 'global minimum', 'random', 'no_agg']
    aggs_2d = ['ecfp4_count']

    if TEST_MODE:
        SAVE_DIR = f'{SAVE_DIR}_test'
        seeds = [42]
        desc_sets = ['moe']
        aggs_3d = ['mean', 'no_agg']
        algorithms = ['InstanceNetRegressor', 'BagNetRegressor']

    if len(sys.argv) == 2:
        arg = int(sys.argv[1])
        if arg == 1:
            desc_sets = ['2d']
        elif arg == 2:
            desc_sets = ['moe']
        elif arg == 3:
            desc_sets = ['pmapper']
        elif arg == 4:
            desc_sets = ['morse']
        elif arg == 5:
            desc_sets = ['mbtr']
    '''
    Directory hierarchy:
    seed -> desc_set -> algorithm -> agg -> 5fold -> predictions for each fold
    '''

    for seed in seeds:
        for desc_set in desc_sets:

            # Set up logging
            SAVE_DIR_LOG = os.path.join(SAVE_DIR, f'seed_{seed}', desc_set)
            os.makedirs(SAVE_DIR_LOG, exist_ok=True)
            LOG_FILE_PATH = os.path.join(SAVE_DIR_LOG, 'result.log')
            LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            logging.basicConfig(
                filename=LOG_FILE_PATH,
                level=logging.INFO,
                format=LOG_FORMAT,
            )

            for algorithm in algorithms:
                if desc_set == 'moe' or desc_set == 'pmapper' or desc_set == 'morse' or desc_set == 'mbtr':
                    for agg in aggs_3d:
                        save_dir_each = f'{SAVE_DIR}/seed_{seed}/{desc_set}/{algorithm}/{agg}'
                        os.makedirs(save_dir_each, exist_ok=True)
                        run_cv(save_dir_each,
                               desc_set,
                               agg,
                               algorithm,
                               TEST_MODE,
                               cv_seed=seed)
                elif desc_set == '2d':
                    for agg in aggs_2d:
                        save_dir_each = f'{SAVE_DIR}/seed_{seed}/{desc_set}/{algorithm}/{agg}'
                        os.makedirs(save_dir_each, exist_ok=True)
                        run_cv(save_dir_each,
                               desc_set,
                               agg,
                               algorithm,
                               TEST_MODE,
                               cv_seed=seed)
