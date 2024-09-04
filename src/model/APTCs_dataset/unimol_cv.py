import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from unimol_tools import MolPredict, MolTrain, UniMolRepr


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


def model_run(
        dataset,
        save_dir,
        agg,
        cv_seed,
        test_mode,
        epochs,
        learning_rate,
        batch_size,
        folds,  # temporary to separate case6 task
):
    '''
    dataset : 'aptc1', 'aptc2'
    agg : 'global minimum', 'no_agg'
    cv_seed: Specify which Seed's CV to use for APTC1
    '''

    if dataset == 'aptc1':
        cv_split = 5  # outer CV
        kfold = 5  # inner CV
        if agg == 'global minimum':
            data_path = '2d3d/dataset/APTC/step7/unimol/aptc1/1conf.csv'
        elif agg == 'no_agg':
            data_path = '2d3d/dataset/APTC/step7/unimol/aptc1/descs_no_agg.csv'
        data_df = pd.read_csv(data_path)

    elif dataset == 'aptc2':
        cv_split = 40  # outer CV
        kfold = 5  # inner CV
        if agg == 'global minimum':
            data_path = '2d3d/dataset/APTC/step7/unimol/aptc2/1conf.csv'
        elif agg == 'no_agg':
            data_path = '2d3d/dataset/APTC/step7/unimol/aptc2/descs_no_agg.csv'
        data_df = pd.read_csv(data_path)

    if test_mode:
        cv_split = 2  # outer CV
        kfold = 2  # inner CV

    index_dir = '2d3d/dataset/APTC/step7/index'
    # Create a dataframe to record the results of all folds
    result_df_all = pd.DataFrame(columns=[
        'R2 Train', 'R2 Test', 'MSE Train', 'MSE Test', 'MAE Train', 'MAE Test'
    ])

    # for fold in range(cv_split):
    for fold in folds:  # temporary to separate case6 task
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

        train_df = data_df[data_df['cid'].isin(tr_cid)]
        test_df = data_df[data_df['cid'].isin(te_cid)]
        train_df['atoms'] = train_df['atoms'].apply(ast.literal_eval)
        train_df['coordinates'] = train_df['coordinates'].apply(
            ast.literal_eval)
        test_df['atoms'] = test_df['atoms'].apply(ast.literal_eval)
        test_df['coordinates'] = test_df['coordinates'].apply(ast.literal_eval)

        train_data = train_df.loc[:,
                                  ['cid', 'atoms', 'coordinates', 'ACTIVITY'
                                   ]].rename(columns={'ACTIVITY': 'target'})
        test_data = test_df.loc[:, ['cid', 'atoms', 'coordinates', 'ACTIVITY'
                                    ]].rename(columns={'ACTIVITY': 'target'})

        if dataset == 'aptc2' and agg == 'global minimum':
            # Duplicate the test data when performing LOOCV because there is only one test data point and it will cause an error.
            test_data = pd.concat([test_data, test_data], ignore_index=True)

        # Convert the merged DataFrame to a dictionary
        train_dict = train_data.to_dict(orient='list')
        test_dict = test_data.to_dict(orient='list')

        # Make DataFrame
        clf = MolTrain(
            task='regression',
            data_type='molecule',
            epochs=epochs,  #100,
            learning_rate=learning_rate,  #1e-4,
            kfold=kfold,  # inner CV
            early_stopping=5,
            batch_size=batch_size,
            metrics='mse',
            #freeze_layers=['encoder', 'gbf'],
            remove_hs=True,
            save_path=save_dir_each_fold)
        clf.fit(data=train_dict)

        clf = MolPredict(load_model=save_dir_each_fold)
        pred_train = clf.predict(data=train_dict)
        pred_test = clf.predict(data=test_dict)

        train_save_df = pd.concat([
            train_data[['cid', 'target']].rename(columns={
                'target': 'actual'
            }).reset_index(drop=True),
            pd.DataFrame(pred_train, columns=['predicted'
                                              ]).reset_index(drop=True)
        ],
                                  axis=1)

        test_save_df = pd.concat([
            test_data[['cid', 'target']].rename(columns={
                'target': 'actual'
            }).reset_index(drop=True),
            pd.DataFrame(pred_test, columns=['predicted'
                                             ]).reset_index(drop=True)
        ],
                                 axis=1)

        if dataset == 'aptc2' and agg == 'global minimum':
            # LOOCVの時で複製したTestデータを削除する．
            test_save_df = test_save_df.drop_duplicates(subset=['cid'],
                                                        keep='first')

        train_save_df.to_csv(f'{save_dir_each_fold}/train_set_predictions.csv',
                             index=False)
        test_save_df.to_csv(f'{save_dir_each_fold}/test_set_predictions.csv',
                            index=False)

        # Calculate metrics for data
        r2_train, mse_train, mae_train = calcMetrics(
            train_save_df['actual'], train_save_df['predicted'])

        if agg == 'no_agg':
            pred_te_mean = test_save_df.groupby('cid').mean().reset_index()
            pred_te_mean.to_csv(os.path.join(save_dir_each_fold,
                                             'test_set_predictions_mean.csv'),
                                index=False)
            r2_test, mse_test, mae_test = calcMetrics(
                pred_te_mean['actual'], pred_te_mean['predicted'])
        else:
            r2_test, mse_test, mae_test = calcMetrics(
                test_save_df['actual'], test_save_df['predicted'])

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
        result_df.to_csv(os.path.join(save_dir_each_fold, 'metrics.csv'),
                         index=False)

        result_df_all = pd.concat([result_df_all, result_df],
                                  axis=0,
                                  ignore_index=True)

    # save all fold results
    result_df_all.to_csv(f'{save_dir}/results.csv', index=False)
    if dataset == 'aptc1':
        # calc mean and std
        mean_std_df = pd.DataFrame({
            f'Mean_{dataset}_{cv_seed}_{agg}':
            result_df_all.mean(),
            f'Std_{dataset}_{cv_seed}_{agg}':
            result_df_all.std()
        }).T
    elif dataset == 'aptc2':
        mean_std_df = pd.DataFrame({
            f'Mean_{dataset}_{agg}':
            result_df_all.mean(),
            f'Std_{dataset}_{agg}':
            result_df_all.std()
        }).T
    mean_std_df.to_csv(f'{save_dir}/mean_std.csv', index=True)


def unite_all_csv(root_dir, filename='mean_std.csv'):
    '''
    Function to combine all calculated results
    '''
    root_path = Path(root_dir)
    csv_files = root_path.rglob(filename)
    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.rename(columns={'Unnamed: 0': 'Name'}, inplace=True)
    combined_df.to_csv(f'{root_path}/all_mean_std.csv', index=False)
    return combined_df


def unite_fold_csv(root_dir, filename='metrics.csv'):
    '''
    Function to combine the results of calculated folds
    This function is specific to the calculation of APTC2, which is divided into separate tasks
    '''
    root_path = Path(root_dir)
    csv_files = root_path.rglob(filename)
    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    mean_std_df = pd.DataFrame({
        f'Mean_aptc2_no_agg': combined_df.mean(),
        f'Std_aptc2_no_agg': combined_df.std()
    }).T
    combined_df.to_csv(f'{root_path}/results.csv', index=False)
    mean_std_df.to_csv(f'{root_path}/mean_std.csv', index=True)
    return combined_df


if __name__ == '__main__':
    SAVE_DIR = 'Uni-Mol/unimol_tools/240807_aptc_cv'
    TEST_MODE = False
    aggs_unimol = ['global minimum', 'no_agg']
    APTC1_seeds = ['seed_12', 'seed_22', 'seed_32', 'seed_42', 'seed_52']
    datasets = ['aptc1', 'aptc2']

    epochs = 100
    learning_rate = 1e-4
    batch_size = 8

    # unite_fold_csv('Uni-Mol/unimol_tools/240807_aptc_cv/aptc2/no_agg')

    if TEST_MODE:
        SAVE_DIR = f'{SAVE_DIR}_test'
        epochs = 2
        aggs_unimol = ['global minimum']
        APTC1_seeds = ['seed_12']
        # datasets = ['aptc2']

    if len(sys.argv) == 2:
        match int(sys.argv[1]):
            case 1:
                aggs_unimol = ['no_agg']
                APTC1_seeds = ['seed_12']
                datasets = ['aptc1']
            case 2:
                aggs_unimol = ['no_agg']
                APTC1_seeds = ['seed_22']
                datasets = ['aptc1']
            case 3:
                aggs_unimol = ['no_agg']
                APTC1_seeds = ['seed_32']
                datasets = ['aptc1']
            case 4:
                aggs_unimol = ['no_agg']
                APTC1_seeds = ['seed_42']
                datasets = ['aptc1']
            case 5:  # cost much time
                aggs_unimol = ['no_agg']
                APTC1_seeds = ['seed_52']
                datasets = ['aptc1']
            case 6:  # cost so much time, should sepalate folds
                aggs_unimol = ['no_agg']
                datasets = ['aptc2']
            case 7:
                aggs_unimol = ['global minimum']
                APTC1_seeds = ['seed_12', 'seed_22']
                datasets = ['aptc1']
            case 8:
                aggs_unimol = ['global minimum']
                APTC1_seeds = ['seed_32', 'seed_42']
                datasets = ['aptc1']
            case 9:
                aggs_unimol = ['global minimum']
                APTC1_seeds = ['seed_52']
                datasets = ['aptc1', 'aptc2']
            # separate case6 task by separating folds
            case 61:
                aggs_unimol = ['no_agg']
                datasets = ['aptc2']
                # folds = [0, 1, 2, 3, 4]
                folds = [4]
            case 62:
                aggs_unimol = ['no_agg']
                datasets = ['aptc2']
                folds = [5, 6, 7, 8, 9]
            case 63:
                aggs_unimol = ['no_agg']
                datasets = ['aptc2']
                folds = [10, 11, 12, 13, 14]
            case 64:
                aggs_unimol = ['no_agg']
                datasets = ['aptc2']
                folds = [15, 16, 17, 18, 19]
            case 65:
                aggs_unimol = ['no_agg']
                datasets = ['aptc2']
                folds = [20, 21, 22, 23, 24]
            case 66:
                aggs_unimol = ['no_agg']
                datasets = ['aptc2']
                folds = [25, 26, 27, 28, 29]
            case 67:
                aggs_unimol = ['no_agg']
                datasets = ['aptc2']
                folds = [30, 31, 32, 33, 34]
            case 68:
                aggs_unimol = ['no_agg']
                datasets = ['aptc2']
                folds = [35, 36, 37, 38, 39]

    for dataset in datasets:
        if dataset == 'aptc1':
            for seed in APTC1_seeds:
                for agg in aggs_unimol:
                    save_dir = f'{SAVE_DIR}/{dataset}/{seed}/{agg}'
                    os.makedirs(save_dir, exist_ok=True)
                    model_run(
                        dataset,
                        save_dir,
                        agg=agg,
                        cv_seed=seed,
                        test_mode=TEST_MODE,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                    )
        elif dataset == 'aptc2':
            for agg in aggs_unimol:
                save_dir = f'{SAVE_DIR}/{dataset}/{agg}'
                os.makedirs(save_dir, exist_ok=True)
                model_run(
                    dataset,
                    save_dir,
                    agg=agg,
                    cv_seed=None,
                    test_mode=TEST_MODE,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    folds=folds,  # temporary to separate case6 task
                )
