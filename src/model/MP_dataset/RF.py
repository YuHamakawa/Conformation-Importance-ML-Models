import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from my_module.evaluation import calcMetrics
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm


def select_dataset(desc_set, agg, test_mode):
    """
    Function to select which dataset to use for analysis.
    desc_set: '2d', 'moe', 'pmapper', 'morse', 'mbtr'
    """
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

    if test_mode:
        df = df.sample(n=10, random_state=0)
    return df


def split_x_y(desc_tr, desc_te, desc_set):
    """
    Function to split the given train and test data into features and target variables.
    """
    if desc_set == 'moe':  # 117 descs
        desc_list = desc_tr.loc[:, 'ASA':'vsurf_Wp8'].columns.to_list()
        x_tr = desc_tr.loc[:, desc_list]
        x_te = desc_te.loc[:, desc_list]
        y_tr = desc_tr.loc[:, 'mpC']
        y_te = desc_te.loc[:, 'mpC']
    elif desc_set == 'pmapper':  # 575 descs
        desc_list = desc_tr.loc[:, 'pmapper_0':'pmapper_574'].columns.to_list()
        x_tr = desc_tr.loc[:, desc_list]
        x_te = desc_te.loc[:, desc_list]
        y_tr = desc_tr.loc[:, 'mpC']
        y_te = desc_te.loc[:, 'mpC']
    elif desc_set == 'morse':  # 160 descs
        desc_list = desc_tr.loc[:, 'Mor01':'Mor32p'].columns.to_list()
        x_tr = desc_tr.loc[:, desc_list]
        x_te = desc_te.loc[:, desc_list]
        y_tr = desc_tr.loc[:, 'mpC']
        y_te = desc_te.loc[:, 'mpC']
    elif desc_set == 'mbtr':  # 950 descs
        desc_list = desc_tr.loc[:, 'MBTR_0':'MBTR_949'].columns.to_list()
        x_tr = desc_tr.loc[:, desc_list]
        x_te = desc_te.loc[:, desc_list]
        y_tr = desc_tr.loc[:, 'mpC']
        y_te = desc_te.loc[:, 'mpC']
    elif desc_set == '2d':  # ecfp4 count 2048 dim
        x_tr = desc_tr.loc[:, '0':'2047']
        x_te = desc_te.loc[:, '0':'2047']
        y_tr = desc_tr.loc[:, 'mpC']
        y_te = desc_te.loc[:, 'mpC']
    else:
        raise ValueError(f'desc_set is invalid. {desc_set}')

    return x_tr, x_te, y_tr, y_te


def scale_data(x_train, x_test):
    '''
    for scaling data
    '''
    # Create a StandardScaler object
    scaler = StandardScaler()
    # Fit the scaler to the training data and transform
    x_train = scaler.fit_transform(x_train)
    # Use the same scaler to transform the test data
    x_test = scaler.transform(x_test)
    return x_train, x_test


def build_model(save_dir, which_model, x_tr, x_te, y_tr, test_mode):
    ### model selection
    if which_model == 'RF':
        params = {
            'n_estimators': 500,
            'max_depth': 50,
            'max_features': None,
        }
        if test_mode:
            params = {
                'n_estimators': 5,
                'max_depth': 5,
            }

        # Create and evaluate the random forest model with the suggested hyperparameters
        model = RandomForestRegressor(n_jobs=-1, random_state=42, **params)

    elif which_model == 'ElasticNet':
        params = {
            'alpha': 1.0,
            'l1_ratio': 0.5,
        }
        model = ElasticNet(**params)

    elif which_model == 'PLS':
        n_components = 5
        model = PLSRegression(n_components=n_components)

    elif which_model == 'SVM':
        params = {'C': 1, 'epsilon': 0.1, 'kernel': 'rbf'}
        model = SVR(**params)

    ### model fitting
    model.fit(x_tr, y_tr)
    tr_pred = model.predict(x_tr).ravel()
    te_pred = model.predict(x_te).ravel()

    ### model interpretation
    if which_model == 'ElasticNet' or which_model == 'PLS':
        coefficients = model.coef_.ravel()
        coef_df = pd.DataFrame({'Coefficient': coefficients})
        coef_df = coef_df.reset_index().rename(columns={'index': 'Features'})
        coef_df.to_csv(os.path.join(save_dir, 'coefficients.csv'), index=False)

    return tr_pred, te_pred


def save_pred(save_dir, tr_csid, te_csid, y_tr, y_te, tr_pred, te_pred, agg):
    '''
    Function to save the model's prediction results for each fold as a dataframe.
    '''
    # create df and save train_cid, y_pred_train, y_train
    pred_tr = pd.DataFrame({
        'csid': tr_csid,
        'predicted': tr_pred,
        'actual': y_tr
    })
    pred_tr.to_csv(os.path.join(save_dir, 'train_set_predictions.csv'),
                   index=False)
    # create df and save test_cid, y_pred_test, y_test
    pred_te = pd.DataFrame({
        'csid': te_csid,
        'predicted': te_pred,
        'actual': y_te
    })
    pred_te.to_csv(os.path.join(save_dir, 'test_set_predictions.csv'),
                   index=False)

    if agg == 'no_agg':
        # If agg is 'no_agg', use the mean of the test predictions
        pred_tr_mean = pred_tr.groupby(by='csid').mean().reset_index()
        pred_tr_mean.to_csv(os.path.join(save_dir,
                                         'train_set_predictions_mean.csv'),
                            index=False)
        pred_te_mean = pred_te.groupby(by='csid').mean().reset_index()
        pred_te_mean.to_csv(os.path.join(save_dir,
                                         'test_set_predictions_mean.csv'),
                            index=False)
        r2_train, mse_train, mae_train = calcMetrics(y_tr, tr_pred)
        r2_test, mse_test, mae_test = calcMetrics(pred_te_mean['actual'],
                                                  pred_te_mean['predicted'])
    else:
        # calculate metrics
        r2_train, mse_train, mae_train = calcMetrics(y_tr, tr_pred)
        r2_test, mse_test, mae_test = calcMetrics(y_te, te_pred)

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


def run_cv(save_dir, desc_set, agg, test_mode, cv_seed=None):
    '''
    Function to perform cross-validation.
    1. Load the data.
    2. Load the pre-prepared index and split into Train and Test sets.
    3. Split into features and target variables.
    4. Build the model for each fold and save the results.
    5. Combine and save the results for all folds.

    args:
        cv_seed: Specify which seed's CV to perform.
    '''
    WHICH_MODEL = 'RF'
    NUM_FOLDS = 5
    if test_mode:
        NUM_FOLDS = 2

    df = select_dataset(desc_set, agg, test_mode)

    results_df = pd.DataFrame(columns=[
        'R2 Train', 'R2 Test', 'MSE Train', 'MSE Test', 'MAE Train', 'MAE Test'
    ])
    for fold in range(NUM_FOLDS):
        save_dir_each_fold = f'{save_dir}/fold_{fold}'
        os.makedirs(save_dir_each_fold, exist_ok=True)

        INDEX_DIR = 'xxx'
        tr_csid = np.load(f'{INDEX_DIR}/seed_{cv_seed}/train_{fold}.npy')
        te_csid = np.load(f'{INDEX_DIR}/seed_{cv_seed}/test_{fold}.npy')
        desc_tr = df[df['csid'].isin(tr_csid)]
        desc_te = df[df['csid'].isin(te_csid)]
        x_tr, x_te, y_tr, y_te = split_x_y(desc_tr, desc_te, desc_set)
        x_tr, x_te = scale_data(x_tr, x_te)
        tr_pred, te_pred = build_model(save_dir_each_fold, WHICH_MODEL, x_tr,
                                       x_te, y_tr, test_mode)
        df_append = save_pred(save_dir_each_fold, desc_tr['csid'],
                              desc_te['csid'], y_tr, y_te, tr_pred, te_pred,
                              agg)
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
    desc_sets = ['2d', 'moe', 'pmapper', 'morse', 'mbtr']
    aggs_3d = ['Boltzman weight', 'mean', 'global minimum', 'random', 'no_agg']
    aggs_2d = ['ecfp4_count']
    # models = ['RF', 'ElasticNet', 'PLS', 'SVM']

    if TEST_MODE:
        SAVE_DIR = f'{SAVE_DIR}_test'
        seeds = [42]

    if len(sys.argv) == 2:
        match int(sys.argv[1]):
            case 1:
                desc_sets = ['2d']
            case 2:
                desc_sets = ['moe']
            case 3:
                desc_sets = ['pmapper']
            case 4:
                desc_sets = ['morse']
            case 5:
                desc_sets = ['mbtr']
    '''
    Directory structure:
    seed -> desc_set -> agg -> 5fold -> predictions for each fold
    '''

    for desc_set in desc_sets:
        for seed in seeds:
            if desc_set == 'moe' or desc_set == 'pmapper' or desc_set == 'morse' or desc_set == 'mbtr':
                for agg in aggs_3d:
                    save_dir_each = f'{SAVE_DIR}/seed_{seed}/{desc_set}/{agg}'
                    os.makedirs(save_dir_each, exist_ok=True)
                    run_cv(save_dir_each,
                           desc_set,
                           agg,
                           TEST_MODE,
                           cv_seed=seed)
            elif desc_set == '2d':
                for agg in aggs_2d:
                    save_dir_each = f'{SAVE_DIR}/seed_{seed}/{desc_set}/{agg}'
                    os.makedirs(save_dir_each, exist_ok=True)
                    run_cv(save_dir_each,
                           desc_set,
                           agg,
                           TEST_MODE,
                           cv_seed=seed)
