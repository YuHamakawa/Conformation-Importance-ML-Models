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
    save_dir,
    agg,
    cv_seed,
    test_mode,
    epochs,
    learning_rate,
    batch_size,
):
    '''
    agg : 'global minimum', 'no_agg'
    cv_seed: Specify which seed's CV to use
    '''

    cv_split = 5  # outer CV
    kfold = 5  # inner CV
    if agg == 'global minimum':
        data_path = 'xxx'
    elif agg == 'no_agg':
        data_path = 'xxx'
    data_df = pd.read_csv(data_path)

    if test_mode:
        cv_split = 2  # outer CV
        kfold = 2  # inner CV

    index_dir = 'xxx'
    result_df_all = pd.DataFrame(columns=[
        'R2 Train', 'R2 Test', 'MSE Train', 'MSE Test', 'MAE Train', 'MAE Test'
    ])

    for fold in range(cv_split):
        save_dir_each_fold = f'{save_dir}/fold_{fold}'
        os.makedirs(save_dir_each_fold, exist_ok=True)

        tr_csid = np.load(f'{index_dir}/seed_{cv_seed}/train_{fold}.npy')
        te_csid = np.load(f'{index_dir}/seed_{cv_seed}/test_{fold}.npy')

        train_df = data_df[data_df['csid'].isin(tr_csid)]
        test_df = data_df[data_df['csid'].isin(te_csid)]
        train_df['atoms'] = train_df['atoms'].apply(ast.literal_eval)
        train_df['coordinates'] = train_df['coordinates'].apply(
            ast.literal_eval)
        test_df['atoms'] = test_df['atoms'].apply(ast.literal_eval)
        test_df['coordinates'] = test_df['coordinates'].apply(ast.literal_eval)

        train_data = train_df.loc[:, ['csid', 'atoms', 'coordinates', 'mpC'
                                      ]].rename(columns={'mpC': 'target'})
        test_data = test_df.loc[:, ['csid', 'atoms', 'coordinates', 'mpC'
                                    ]].rename(columns={'mpC': 'target'})

        # Convert the merged DataFrame to a dictionary
        train_dict = train_data.to_dict(orient='list')
        test_dict = test_data.to_dict(orient='list')

        # DataFrameの作成
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
            train_data[['csid', 'target']].rename(columns={
                'target': 'actual'
            }).reset_index(drop=True),
            pd.DataFrame(pred_train, columns=['predicted'
                                              ]).reset_index(drop=True)
        ],
                                  axis=1)

        test_save_df = pd.concat([
            test_data[['csid', 'target']].rename(columns={
                'target': 'actual'
            }).reset_index(drop=True),
            pd.DataFrame(pred_test, columns=['predicted'
                                             ]).reset_index(drop=True)
        ],
                                 axis=1)

        train_save_df.to_csv(f'{save_dir_each_fold}/train_set_predictions.csv',
                             index=False)
        test_save_df.to_csv(f'{save_dir_each_fold}/test_set_predictions.csv',
                            index=False)

        # Calculate metrics for data
        r2_train, mse_train, mae_train = calcMetrics(
            train_save_df['actual'], train_save_df['predicted'])

        if agg == 'no_agg':
            pred_te_mean = test_save_df.groupby('csid').mean().reset_index()
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
    mean_std_df = pd.DataFrame({
        f'Mean_{cv_seed}_{agg}': result_df_all.mean(),
        f'Std_{cv_seed}_{agg}': result_df_all.std()
    }).T
    mean_std_df.to_csv(f'{save_dir}/mean_std.csv', index=True)


if __name__ == '__main__':
    SAVE_DIR = 'xxx'
    TEST_MODE = False
    aggs_unimol = ['global minimum', 'no_agg']
    seeds = [12, 22, 32, 42, 52]

    epochs = 100
    learning_rate = 1e-4
    batch_size = 8

    if TEST_MODE:
        SAVE_DIR = f'{SAVE_DIR}_test'
        epochs = 1
        seeds = [42]

    if len(sys.argv) == 2:
        match int(sys.argv[1]):
            case 1:
                seeds = [12]
            case 2:
                seeds = [22]
            case 3:
                seeds = [32]
            case 4:
                seeds = [42]
            case 5:
                seeds = [52]

    for seed in seeds:
        for agg in aggs_unimol:
            save_dir = f'{SAVE_DIR}/seed_{seed}/{agg}'
            os.makedirs(save_dir, exist_ok=True)
            model_run(
                save_dir,
                agg=agg,
                cv_seed=seed,
                test_mode=TEST_MODE,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
            )
