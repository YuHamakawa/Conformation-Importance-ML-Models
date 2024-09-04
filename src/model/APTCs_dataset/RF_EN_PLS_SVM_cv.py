import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm


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
    Function to select which dataset to use for analysis.
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


def split_x_y(desc_tr, desc_te, desc_set, agg, dataset):
    '''
    Function to split the given train and test data into features and target variables.
    '''
    if desc_set == 'moe':
        # All descs are 117 descs
        desc_list = desc_tr.loc[:, 'ASA':'vsurf_Wp8'].columns.to_list()
        x_tr = desc_tr.loc[:, desc_list]
        x_te = desc_te.loc[:, desc_list]
        y_tr = desc_tr.loc[:, 'ACTIVITY']
        y_te = desc_te.loc[:, 'ACTIVITY']
    elif desc_set == 'pmapper':
        if dataset == 'aptc1':
            desc_list = desc_tr.loc[:, '0':'1201'].columns.to_list()
        elif dataset == 'aptc2':
            desc_list = desc_tr.loc[:, '0':'1555'].columns.to_list()
        x_tr = desc_tr.loc[:, desc_list]
        x_te = desc_te.loc[:, desc_list]
        y_tr = desc_tr.loc[:, 'ACTIVITY']
        y_te = desc_te.loc[:, 'ACTIVITY']
    elif desc_set == '2d':
        if agg == 'pharm_2d':
            x_tr = desc_tr.loc[:, '0':'2047']
            x_te = desc_te.loc[:, '0':'2047']
        else:
            x_tr = desc_tr.loc[:, '0':'1023']
            x_te = desc_te.loc[:, '0':'1023']
        y_tr = desc_tr.loc[:, 'ACTIVITY']
        y_te = desc_te.loc[:, 'ACTIVITY']
    elif desc_set == 'unirep':
        x_tr = desc_tr.loc[:, '0':'511']
        x_te = desc_te.loc[:, '0':'511']
        y_tr = desc_tr.loc[:, 'target']
        y_te = desc_te.loc[:, 'target']
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
    '''
    Function to select and build the model.
    Available models: ['RF', 'ElasticNet', 'PLS', 'SVM']
    '''

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


def save_pred(save_dir, tr_cid, te_cid, y_tr, y_te, tr_pred, te_pred, agg):
    '''
    Function to save the model's predictions for each fold as a dataframe.
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

    if agg == 'no_agg':
        # If agg is 'no_agg', the test predictions should be averaged
        pred_te_mean = pred_te.groupby(by='cid').mean().reset_index()
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


def run_cv(save_dir,
           desc_set,
           agg,
           dataset,
           which_model,
           test_mode,
           cv_seed=None):
    '''
    Function to perform cross-validation.
    1. Read the data.
    2. Load the pre-defined indices for train-test split.
    3. Split the data into features and target variables.
    4. Build the model for each fold and save the results.
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
        x_tr, x_te, y_tr, y_te = split_x_y(desc_tr, desc_te, desc_set, agg,
                                           dataset)
        x_tr, x_te = scale_data(x_tr, x_te)
        tr_pred, te_pred = build_model(save_dir_each_fold, which_model, x_tr,
                                       x_te, y_tr, test_mode)
        df_append = save_pred(save_dir_each_fold, desc_tr['cid'],
                              desc_te['cid'], y_tr, y_te, tr_pred, te_pred,
                              agg)
        results_df = pd.concat([results_df, df_append],
                               axis=0,
                               ignore_index=True)

    results_df.to_csv(f'{save_dir}/results.csv', index=False)

    if dataset == 'aptc1':
        mean_std_df = pd.DataFrame({
            f'Mean_{desc_set}_{dataset}_{cv_seed}_{which_model}_{agg}':
            results_df.mean(),
            f'Std_{desc_set}_{dataset}_{cv_seed}_{which_model}_{agg}':
            results_df.std()
        }).T
    elif dataset == 'aptc2':
        mean_std_df = pd.DataFrame({
            f'Mean_{desc_set}_{dataset}_{which_model}_{agg}':
            results_df.mean(),
            f'Std_{desc_set}_{dataset}_{which_model}_{agg}':
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
    SAVE_DIR = 'xxx'
    test_mode = False
    datasets = ['aptc1', 'aptc2']
    desc_sets = ['2d', 'moe', 'pmapper', 'unirep']

    aggs_moe_pmapper = [
        'Boltzman weight', 'mean', 'global minimum', 'random', 'no_agg',
        'random_12', 'random_22', 'random_32', 'random_52'
    ]
    aggs_2d = ['ecfp_bit', 'ecfp_count', 'fcfp_bit', 'fcfp_count', 'pharm_2d']
    aggs_unirep = ['global minimum', 'no_agg']

    APTC1_seeds = ['seed_12', 'seed_22', 'seed_32', 'seed_42', 'seed_52']

    models = ['RF', 'ElasticNet', 'PLS', 'SVM']

    unite_all_csv(SAVE_DIR, filename='mean_std.csv')
    # MIL result
    unite_all_csv('3D-MIL-QSSR/output/240806_aptc_MIL',
                  filename='mean_std.csv')
    # # Uni-Mol result
    # unite_all_csv('Uni-Mol/unimol_tools/240807_aptc_cv',
    #               filename='mean_std.csv')

    desc_sets = ['2d']
    aggs_2d = ['pharm_2d']

    if test_mode:
        desc_sets = ['moe']
        datasets = ['aptc2']
        aggs_moe_pmapper = ['mean', 'no_agg']
        #models = ['RF']

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
    Directory structure
    APTC-1: 
    desc_set -> aptc1 -> seed -> model -> agg -> 5fold -> predictions for each fold
    
    APTC-2:
    desc_set -> aptc2 -> model -> agg -> 40fold(LOO) -> predictions for each fold
    '''

    for desc_set in desc_sets:
        for dataset in datasets:
            if dataset == 'aptc1':
                for seed in APTC1_seeds:
                    for model in models:
                        if desc_set == 'moe' or desc_set == 'pmapper':
                            for agg in aggs_moe_pmapper:
                                save_dir_each = f'{SAVE_DIR}/{desc_set}/{dataset}/{seed}/{model}/{agg}'
                                os.makedirs(save_dir_each, exist_ok=True)
                                run_cv(save_dir_each,
                                       desc_set,
                                       agg,
                                       dataset,
                                       model,
                                       test_mode,
                                       cv_seed=seed)
                        elif desc_set == '2d':
                            for agg in aggs_2d:
                                save_dir_each = f'{SAVE_DIR}/{desc_set}/{dataset}/{seed}/{model}/{agg}'
                                os.makedirs(save_dir_each, exist_ok=True)
                                run_cv(save_dir_each,
                                       desc_set,
                                       agg,
                                       dataset,
                                       model,
                                       test_mode,
                                       cv_seed=seed)
                        elif desc_set == 'unirep':
                            for agg in aggs_unirep:
                                save_dir_each = f'{SAVE_DIR}/{desc_set}/{dataset}/{seed}/{model}/{agg}'
                                os.makedirs(save_dir_each, exist_ok=True)
                                run_cv(save_dir_each,
                                       desc_set,
                                       agg,
                                       dataset,
                                       model,
                                       test_mode,
                                       cv_seed=seed)

            elif dataset == 'aptc2':
                for model in models:
                    if desc_set == 'moe' or desc_set == 'pmapper':
                        for agg in aggs_moe_pmapper:
                            save_dir_each = f'{SAVE_DIR}/{desc_set}/{dataset}/{model}/{agg}'
                            os.makedirs(save_dir_each, exist_ok=True)
                            run_cv(save_dir_each, desc_set, agg, dataset,
                                   model, test_mode)

                    elif desc_set == '2d':
                        for agg in aggs_2d:
                            save_dir_each = f'{SAVE_DIR}/{desc_set}/{dataset}/{model}/{agg}'
                            os.makedirs(save_dir_each, exist_ok=True)
                            run_cv(save_dir_each, desc_set, agg, dataset,
                                   model, test_mode)
                    elif desc_set == 'unirep':
                        for agg in aggs_unirep:
                            save_dir_each = f'{SAVE_DIR}/{desc_set}/{dataset}/{model}/{agg}'
                            os.makedirs(save_dir_each, exist_ok=True)
                            run_cv(save_dir_each, desc_set, agg, dataset,
                                   model, test_mode)
