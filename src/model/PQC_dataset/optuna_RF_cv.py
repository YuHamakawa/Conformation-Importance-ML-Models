import logging
import os
import sys
import threading

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from optuna.pruners import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import export_text, plot_tree
from tqdm import tqdm

logger = logging.getLogger(__name__)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def preprocess_2d(y_name):
    df_ecfp = pd.read_csv('2d3d/dataset/PubChemQC-PM6/2D/step2/ecfp_97696.tsv',
                          sep='\t')
    # sort by cid
    df_ecfp.sort_values('cid', inplace=True)
    # delete nan
    if df_ecfp[y_name].isna().sum() > 0:
        missing = df_ecfp[df_ecfp[y_name].isna()]
        print(f'some elements have missing value\n{missing}')
        df_ecfp.dropna(subset=[y_name], inplace=True)

    X = df_ecfp.loc[:, '0':'2047']
    y = df_ecfp.loc[:, y_name]
    return X, y


def preprocess_no_agg(y_name):
    descs_no_agg = pd.read_csv(
        '2d3d/dataset/PubChemQC-PM6/step15/descs_no_agg.csv')
    # sort by cid
    descs_no_agg.sort_values('cid', inplace=True)
    # delete nan
    if descs_no_agg[y_name].isna().sum() > 0:
        missing = descs_no_agg[descs_no_agg[y_name].isna()]
        print(f'some elements have missing value\n{missing}')
        descs_no_agg.dropna(subset=[y_name], inplace=True)

    # All descs are 116 descs (= 117 descs - 'dipole')
    desc_list = descs_no_agg.loc[:, 'ASA':'vsurf_Wp8'].columns.to_list()
    if 'dipole' in desc_list:
        desc_list.remove('dipole')

    # add bags column that group conformers
    descs_no_agg['bags'] = descs_no_agg['cid'].apply(lambda x: x.split('_')[0])
    desc_list += ['bags']

    x = descs_no_agg.loc[:, desc_list]
    y = descs_no_agg.loc[:, y_name]

    return x, y


def preprocess_data_for_optunaRF(agg, y_name, save_index=False):
    '''preprocess data for model, also can save index for no_agg
    Args:
        agg: method of aggregate conformers
        y_name: target label
    Returns:
        X: training data
        y: target label
    '''

    if agg == '2d':
        return preprocess_2d(y_name)

    if agg == 'no_agg':
        return preprocess_no_agg(y_name)

    basefd = '2d3d/dataset/PubChemQC-PM6'
    # read dataset
    desc_weight_mean = pd.read_csv(f'{basefd}/step14/HFweight.tsv', sep='\t')
    desc_eq_mean = pd.read_csv(f'{basefd}/step14/eq_weight.tsv', sep='\t')
    desc_min = pd.read_csv(f'{basefd}/step14/1conf.tsv', sep='\t')
    desc_random = pd.read_csv(f'{basefd}/step14/random.tsv', sep='\t')
    desc_rmsd_max = pd.read_csv(f'{basefd}/step14/rmsd_max.tsv', sep='\t')
    desc_rmsd_min = pd.read_csv(f'{basefd}/step14/rmsd_min.tsv', sep='\t')
    desc_correct = pd.read_csv(f'{basefd}/step10/correct.tsv', sep='\t')
    desc_correct = desc_correct[desc_correct['cid'].isin(
        desc_weight_mean['cid'])]

    # sort by cid
    desc_weight_mean.sort_values('cid', inplace=True)
    desc_eq_mean.sort_values('cid', inplace=True)
    desc_min.sort_values('cid', inplace=True)
    desc_random.sort_values('cid', inplace=True)
    desc_rmsd_max.sort_values('cid', inplace=True)
    desc_rmsd_min.sort_values('cid', inplace=True)
    desc_correct.sort_values('cid', inplace=True)

    # check whether cid match among all aggregation data
    if ((desc_eq_mean['cid'].isin(desc_weight_mean['cid']) == False).sum() != 0
            or (desc_eq_mean['cid'].isin(desc_min['cid']) == False).sum() != 0
            or
        (desc_eq_mean['cid'].isin(desc_random['cid']) == False).sum() != 0 or
        (desc_eq_mean['cid'].isin(desc_rmsd_max['cid']) == False).sum() != 0 or
        (desc_eq_mean['cid'].isin(desc_rmsd_min['cid']) == False).sum() != 0):
        raise ValueError('cid among aggregation do not match!!')

    # delete nan
    if desc_weight_mean[y_name].isna().sum() > 0:
        missing = desc_weight_mean[desc_weight_mean[y_name].isna()]
        print(f'some elements have missing value\n{missing}')
        desc_weight_mean.dropna(subset=[y_name], inplace=True)
        desc_eq_mean.dropna(subset=[y_name], inplace=True)
        desc_min.dropna(subset=[y_name], inplace=True)
        desc_random.dropna(subset=[y_name], inplace=True)
        desc_rmsd_max.dropna(subset=[y_name], inplace=True)
        desc_rmsd_min.dropna(subset=[y_name], inplace=True)
        desc_correct.dropna(subset=[y_name], inplace=True)

    # All descs are 116 descs (= 117 descs - 'dipole')
    desc_list = desc_weight_mean.loc[:, 'ASA':'vsurf_Wp8'].columns.to_list()
    if 'dipole' in desc_list:
        desc_list.remove('dipole')

    x_weight = desc_weight_mean.loc[:, desc_list]
    y = desc_weight_mean.loc[:, y_name]
    x_mean = desc_eq_mean.loc[:, desc_list]
    x_max = desc_min.loc[:, desc_list]
    x_random = desc_random.loc[:, desc_list]
    x_rmsd_max = desc_rmsd_max.loc[:, desc_list]
    x_rmsd_min = desc_rmsd_min.loc[:, desc_list]
    x_correct = desc_correct.loc[:, desc_list]

    match agg:
    # original data shape is (97696, 127) *contain missing 1 row (enthalpy)
        case 'Boltzman weight':
            X = x_weight
        case 'mean':
            X = x_mean
        case 'global minimum':
            X = x_max
        case 'random':
            X = x_random
        case 'rmsd_max':
            X = x_rmsd_max
        case 'rmsd_min':
            X = x_rmsd_min
        # only correct have different shape (99927, 125) *contain missing 1 row (enthalpy)
        case 'correct':
            X, y = x_correct, desc_correct.loc[:, y_name]

    if save_index:
        # for save index, index is needed for no-agg
        save_dir = f'2d3d/dataset/PubChemQC-PM6/step15/index/{y_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        for i, (train_index, test_index) in enumerate(outer_cv.split(X)):
            # cid is sorted, so it is same if use any desc_***
            train_cid = desc_min.iloc[train_index, 0].to_numpy()
            test_cid = desc_min.iloc[test_index, 0].to_numpy()
            np.save(f'{save_dir}/train_cid_fold_{i}', train_cid)
            np.save(f'{save_dir}/test_cid_fold_{i}', test_cid)

    return X, y


def objective(trial,
              x,
              y,
              use_cv=True,
              cv=KFold(n_splits=5, shuffle=True, random_state=42),
              scoring='neg_mean_squared_error'):
    '''Objective function for Optuna to minimize the mean squared error of a random forest model.
    Args:
        trial: Optuna's trial object
        X_train: training data
        y_train: training label
        X_test: test data
        y_test: test label
    Returns:
        mse: mean squared error
    '''

    # Define the search space for hyperparameters
    params = {
        'n_estimators':
        trial.suggest_int('n_estimators', 10, 500),
        'max_depth':
        trial.suggest_int('max_depth', 2, 128, log=True),
        'max_features':
        trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        # 'criterion':
        # trial.suggest_categorical(
        #     'criterion',
        #     ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']),
        # 'min_samples_split':
        # trial.suggest_int('min_samples_split', 2, 10),
        # 'min_samples_leaf':
        # trial.suggest_int('min_samples_leaf', 1, 10),
        # 'bootstrap':
        # trial.suggest_categorical('bootstrap', [True, False]),
    }

    # for test
    # params = {
    #     'n_estimators':
    #     trial.suggest_int('n_estimators', 2, 5),
    #     'max_depth':
    #     trial.suggest_int('max_depth', 2, 4, log=True),
    #     'max_features':
    #     trial.suggest_categorical('max_features', ['sqrt', 'log2']),
    # }

    # Create and evaluate the random forest model with the suggested hyperparameters
    model = RandomForestRegressor(n_jobs=-1, random_state=42, **params)

    if use_cv:
        score = cross_val_score(model, x, y, cv=cv, scoring=scoring, n_jobs=-1)
        return score.mean()

    else:
        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        model.fit(x_train, y_train)
        y_pred_test = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred_test)
        # adjust mse to 'maximize'
        return -mse


def optimize_hyperparameters(
        x_train,
        y_train,
        n_trials,
        direction,
        pruner=MedianPruner(),
        use_cv=True,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
):
    '''Optimize hyperparameters of a random forest model using Optuna.
    Args:
        X_train: training data
        y_train: training label
        X_test: test data
        y_test: test label
        n_trials: number of trials of Optuna
    Returns:
        best_params: best hyperparameters
    '''

    study = optuna.create_study(
        direction=direction,
        pruner=pruner,
    )
    study.optimize(lambda trial: objective(trial,
                                           x_train,
                                           y_train,
                                           use_cv=use_cv,
                                           cv=cv,
                                           scoring='neg_mean_squared_error'),
                   n_trials=n_trials)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    return trial.params


def search_best_params(x_train,
                       x_test,
                       y_train,
                       y_test,
                       agg,
                       y_name,
                       n_trials,
                       save_dir,
                       use_cv=True,
                       cv=KFold(n_splits=5, shuffle=True, random_state=42)):
    '''
    for calc. optimized by MOPAC
    Main function to train a random forest model and optimize hyperparameters.

    Args:
        agg (str): Method of aggregate conformers.
        y_name (str): Target label (e.g., gap, dipole).
        date (str): Today's date for saving figures.
        n_trials (int): Number of trials of Optuna.
    '''

    # always maximize if use sklearn's metrics
    best_params = optimize_hyperparameters(x_train,
                                           y_train,
                                           n_trials,
                                           direction='maximize',
                                           pruner=MedianPruner(),
                                           use_cv=use_cv,
                                           cv=cv)
    # best_params = {}
    # best_params['n_estimators'] = 2
    # best_params['max_depth'] = 2
    # best_params['max_features'] = 'sqrt'

    # Use the best hyperparameters to train the final model
    final_model = RandomForestRegressor(n_jobs=-1, **best_params)
    final_model.fit(x_train, y_train)
    y_pred_train = final_model.predict(x_train)
    y_pred_test = final_model.predict(x_test)

    plot_scatter(y_train, y_pred_train, y_test, y_pred_test, y_name, agg,
                 best_params, save_dir)
    plot_finel_model_tree(final_model, x_train.columns, save_dir)

    return y_pred_train, y_pred_test


def plot_scatter(y_train, y_pred_train, y_test, y_pred_test, y_name, agg,
                 best_params, save_dir):

    # Calculate and print metrics
    r2_train, mse_train, mae_train = calcMetrics(y_train, y_pred_train)
    r2_test, mse_test, mae_test = calcMetrics(y_test, y_pred_test)

    logger.info(f'y_name = {y_name}, agg = {agg}')
    logger.info(
        '(Train) R2: {:.3}, MSE: {:.3}, MAE: {:.3}, best_params: {}'.format(
            r2_train, mse_train, mae_train, best_params))
    logger.info(
        '(Test ) R2: {:.3}, MSE: {:.3}, MAE: {:.3}, best_params: {}'.format(
            r2_test, mse_test, mae_test, best_params))

    # Visualize prediction results
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(y_train, y_pred_train, label='Train', alpha=0.3)
    plt.scatter(y_test, y_pred_test, label='Test', alpha=0.3)
    # Prepare for drawing a diagonal line
    yvalues = np.concatenate([y_test, y_pred_test]).flatten()
    ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
    # Display R2, MSE, MAE values on the graph
    plt.text(
        0.01,
        0.98,
        #f'         (Train)  (Test)\n    R2: {r2_train:.3f}, {r2_test:.3f}\n MSE: {mse_train:.3f}, {mse_test:.3f}\n MAE: {mae_train:.3f}, {mae_test:.3f}',
        f'(Train) R2: {r2_train:.3f}, MSE: {mse_train:.3f}, MAE: {mae_train:.3f}\n(Test) R2: {r2_test:.3f}, MSE: {mse_test:.3f}, MAE: {mae_test:.3f}',
        ha='left',
        va='top',
        transform=ax.transAxes)
    # Draw a diagonal line
    plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01],
             [ymin - yrange * 0.01, ymax + yrange * 0.01],
             color='k')
    plt.xlabel('Observations')
    plt.ylabel('Predictions')
    plt.title(f'True vs. Pred ({agg})')
    plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, f'RF_yy-plot_{y_name}_{agg}.pdf'))
    plt.close()


def plot_finel_model_tree(final_model,
                          feature_names,
                          save_dir,
                          number_of_tree=5,
                          max_depth=3):
    # Prepare plotting, make save dir
    tree_dir = os.path.join(save_dir, 'tree')
    if not os.path.exists(tree_dir):
        os.makedirs(tree_dir)
    # Visualize random forest
    # Get one decision tree from the trained random forest model
    # one_tree = final_model.estimators_[0]
    for i, tree in enumerate(final_model.estimators_[:number_of_tree]):
        # Display the structure of the decision tree as text
        #tree_text = export_text(tree, feature_names=X_train.columns.tolist())
        #print(tree_text)

        # Visualize and save the structure of the decision tree
        plt.figure(figsize=(20, 10))
        plot_tree(tree,
                  max_depth=max_depth,
                  feature_names=feature_names,
                  filled=True,
                  rounded=True,
                  class_names=True)
        plt.savefig(os.path.join(tree_dir, f'tree_{i+1}.pdf'))
        plt.close()


def nested_cross_validation_multi(
    agg,
    y_name,
    save_dir,
    use_cv_inner=True,
    use_cv_outer=True,
    n_split_inner=5,
    n_split_outer=3,
    n_trials=30,
):
    # Data preprocessing
    x, y = preprocess_data_for_optunaRF(agg, y_name)

    # inner cv : search hyperparameters using optuna
    inner_cv = KFold(n_splits=n_split_inner, shuffle=True, random_state=42)

    if use_cv_outer:
        # initialize data frame
        results_df = pd.DataFrame(columns=[
            'Fold', 'R2 Train', 'R2 Test', 'MSE Train', 'MSE Test',
            'MAE Train', 'MAE Test'
        ])

        # outer cv : evaluate model
        outer_cv = KFold(n_splits=n_split_outer, shuffle=True, random_state=42)

        # Outer loop : parallel processing
        with Parallel(n_jobs=-1, verbose=1) as parallel:
            results = parallel(
                delayed(evaluate_fold)
                (x.iloc[train_index], x.iloc[test_index], y.iloc[train_index],
                 y.iloc[test_index], agg, y_name, n_trials, n_split_outer,
                 use_cv_inner, inner_cv, save_dir, i)
                for i, (train_index,
                        test_index) in enumerate(outer_cv.split(x)))

        for result in results:
            results_df = pd.concat([results_df, result],
                                   axis=0,
                                   ignore_index=True)

        results_df.to_csv(os.path.join(save_dir, 'results.csv'), index=False)

        # calculate mean and std of metrics
        mean_std_df = pd.DataFrame({
            f'Mean_{y_name}_{agg}': results_df.mean(),
            f'Std_{y_name}_{agg}': results_df.std()
        }).T

        return mean_std_df

    else:  # in case, not use_cv_outer
        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        # inner loop, if use_cv_inner is True
        search_best_params(x_train=x_train,
                           x_test=x_test,
                           y_train=y_train,
                           y_test=y_test,
                           agg=agg,
                           y_name=y_name,
                           n_trials=n_trials,
                           save_dir=save_dir,
                           use_cv=use_cv_inner,
                           cv=inner_cv)

        return


def evaluate_fold(x_train, x_test, y_train, y_test, agg, y_name, n_trials,
                  n_split_outer, use_cv_inner, inner_cv, save_dir, fold):
    # make save dir for each fold
    save_dir_each_fold = os.path.join(save_dir, f'fold_{fold}')
    if not os.path.exists(save_dir_each_fold):
        os.makedirs(save_dir_each_fold)

    logger.info(f'fold = {fold+1} / {n_split_outer}')

    # inner loop, if use_cv_inner is True
    y_pred_train, y_pred_test = search_best_params(x_train=x_train,
                                                   x_test=x_test,
                                                   y_train=y_train,
                                                   y_test=y_test,
                                                   agg=agg,
                                                   y_name=y_name,
                                                   n_trials=n_trials,
                                                   save_dir=save_dir_each_fold,
                                                   use_cv=use_cv_inner,
                                                   cv=inner_cv)

    r2_train, mse_train, mae_train = calcMetrics(y_train, y_pred_train)
    r2_test, mse_test, mae_test = calcMetrics(y_test, y_pred_test)

    # add results to a dataframe
    df_append = pd.DataFrame(
        {
            'Fold': fold,
            'R2 Train': r2_train,
            'R2 Test': r2_test,
            'MSE Train': mse_train,
            'MSE Test': mse_test,
            'MAE Train': mae_train,
            'MAE Test': mae_test
        },
        index=[0])

    return df_append


def nested_cross_validation_single(
    agg,
    y_name,
    save_dir,
    use_cv_inner=True,
    use_cv_outer=True,
    n_split_inner=5,
    n_split_outer=3,
    n_trials=30,
    test_mode=False,
):
    # Data preprocessing
    x, y = preprocess_data_for_optunaRF(agg, y_name)

    # inner cv : search hyperparameters using optuna
    inner_cv = KFold(n_splits=n_split_inner, shuffle=True, random_state=42)

    if use_cv_outer:
        # initialize data frame
        results_df = pd.DataFrame(columns=[
            'Fold', 'R2 Train', 'R2 Test', 'MSE Train', 'MSE Test',
            'MAE Train', 'MAE Test'
        ])

        # outer cv : evaluate model
        outer_cv = KFold(n_splits=n_split_outer, shuffle=True, random_state=42)

        # Outer loop
        for i, (train_index, test_index) in enumerate(outer_cv.split(x)):
            ### read index with npy, it can generate by preprocess.py
            ### this is for 5-fold cv only
            data_dir = f'2d3d/dataset/PubChemQC-PM6/step15/index/{y_name}'
            train_cid = np.load(f'{data_dir}/train_cid_fold_{i}.npy')
            test_cid = np.load(f'{data_dir}/test_cid_fold_{i}.npy')
            # Split data
            if agg == 'no_agg':
                # extract data with cid
                data = pd.concat([x, y], axis=1)
                train_data = data[data['bags'].astype(int).isin(train_cid)]
                test_data = data[data['bags'].astype(int).isin(test_cid)]
                # split x and y, also record cid
                x_train = train_data.loc[:, 'ASA':'vsurf_Wp8']
                y_train = train_data.loc[:, y_name]
                train_cid = train_data.loc[:, 'bags']
                x_test = test_data.loc[:, 'ASA':'vsurf_Wp8']
                y_test = test_data.loc[:, y_name]
                test_cid = test_data.loc[:, 'bags']

            else:
                x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # make save dir for each fold
            save_dir_each_fold = os.path.join(save_dir, f'fold_{i}')
            if not os.path.exists(save_dir_each_fold):
                os.makedirs(save_dir_each_fold)

            logger.info(f'fold = {i+1} / {n_split_outer}')

            # inner loop, if use_cv_inner is True
            y_pred_train, y_pred_test = search_best_params(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                agg=agg,
                y_name=y_name,
                n_trials=n_trials,
                save_dir=save_dir_each_fold,
                use_cv=use_cv_inner,
                cv=inner_cv)

            # create df and save train_cid, y_pred_train, y_train
            pred_tr = pd.DataFrame({
                'cid': train_cid,
                'predicted': y_pred_train,
                'actual': y_train
            })
            pred_tr.to_csv(os.path.join(save_dir_each_fold,
                                        'train_set_predictions.csv'),
                           index=False)
            # create df and save test_cid, y_pred_test, y_test
            pred_te = pd.DataFrame({
                'cid': test_cid,
                'predicted': y_pred_test,
                'actual': y_test
            })
            pred_te.to_csv(os.path.join(save_dir_each_fold,
                                        'test_set_predictions.csv'),
                           index=False)

            # calculate metrics
            r2_train, mse_train, mae_train = calcMetrics(y_train, y_pred_train)
            r2_test, mse_test, mae_test = calcMetrics(y_test, y_pred_test)

            # add results to a dataframe
            df_append = pd.DataFrame(
                {
                    'Fold': i,
                    'R2 Train': r2_train,
                    'R2 Test': r2_test,
                    'MSE Train': mse_train,
                    'MSE Test': mse_test,
                    'MAE Train': mae_train,
                    'MAE Test': mae_test
                },
                index=[0])
            results_df = pd.concat([results_df, df_append],
                                   axis=0,
                                   ignore_index=True)

            if test_mode:
                # only 1-fold for test
                break

        results_df.to_csv(os.path.join(save_dir, 'results.csv'), index=False)

        # calculate mean and std of metrics
        mean_std_df = pd.DataFrame({
            f'Mean_{y_name}_{agg}': results_df.mean(),
            f'Std_{y_name}_{agg}': results_df.std()
        }).T

        return mean_std_df

    else:  # in case, not use_cv_outer
        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        # inner loop, if use_cv_inner is True
        search_best_params(x_train=x_train,
                           x_test=x_test,
                           y_train=y_train,
                           y_test=y_test,
                           agg=agg,
                           y_name=y_name,
                           n_trials=n_trials,
                           save_dir=save_dir,
                           use_cv=use_cv_inner,
                           cv=inner_cv)

        return


if __name__ == '__main__':
    TEST_MODE = False

    TODAY = 'xxx'
    N_TRIALS = 30
    N_SPLIT_OUTER = 5

    aggs = [
        '2d', 'correct', 'mean', 'Boltzman weight', 'global minimum', 'random',
        'rmsd_max', 'rmsd_min', 'no_agg'
    ]

    y_names = ['dipoleMoment', 'homo', 'gap', 'lumo', 'energy', 'enthalpy']

    if len(sys.argv) == 2:
        match int(sys.argv[1]):
            case 1:
                y_names = ['dipoleMoment', 'homo']
            case 2:
                y_names = ['gap', 'lumo']
            case 3:
                y_names = ['energy', 'enthalpy']

    if TEST_MODE:
        TODAY = 'test'
        N_TRIALS = 1
        #N_SPLIT_OUTER = 2
        if 0:
            aggs = ['no_agg']
            y_names = ['dipoleMoment']
        if 1:
            aggs = ['2d', 'Boltzman weight']
            y_names = ['dipoleMoment', 'homo']

    SAVE_DIR_1 = os.path.join('2d3d/output', TODAY)
    if not os.path.exists(SAVE_DIR_1):
        os.makedirs(SAVE_DIR_1)

    # Set up logging
    LOG_FILE_PATH = os.path.join(SAVE_DIR_1, 'results.log')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        level=logging.INFO,
        format=LOG_FORMAT,
    )

    # Initialize a list to store results for multiple datasets
    mean_std_df_list = []

    for y_name in tqdm(y_names, desc='y_name'):
        for agg in tqdm(aggs, desc='agg'):
            SAVE_DIR_2 = os.path.join(SAVE_DIR_1, y_name, agg)
            if not os.path.exists(SAVE_DIR_2):
                os.makedirs(SAVE_DIR_2)

            mean_std_df = nested_cross_validation_single(
                agg=agg,
                y_name=y_name,
                save_dir=SAVE_DIR_2,
                use_cv_inner=False,
                use_cv_outer=True,
                n_split_inner=2,
                n_split_outer=N_SPLIT_OUTER,
                n_trials=N_TRIALS,
                test_mode=TEST_MODE,
            )

            # save result for each y_name and agg
            mean_std_df.to_csv(os.path.join(SAVE_DIR_2, 'mean_std.csv'))

            mean_std_df_list.append(mean_std_df)
            # save result for every loop
            mean_std_df_all = pd.concat(mean_std_df_list, axis=0)

            mean_std_df_all.to_csv(os.path.join(SAVE_DIR_1, 'mean_std.csv'),
                                   index=True)

            print(f'{y_name} {agg} Finish!')

    print('All loops Finish!')
