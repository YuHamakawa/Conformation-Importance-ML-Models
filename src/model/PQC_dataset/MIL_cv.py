import logging
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from estimators.attention_nets import AttentionNetRegressor
from estimators.mi_nets import BagNetRegressor, InstanceNetRegressor
from estimators.wrappers import (BagWrapperMLPRegressor,
                                 InstanceWrapperMLPRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from tqdm import tqdm
from utils import (calc_catalyst_descr, calc_pm6_descr, calc_reaction_descr,
                   concat_react_cat_descr, gen_catalyst_confs, read_input_data,
                   scale_data)

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


def calc_pmapper_descs():
    data_path = '2d3d/dataset/PubChemQC-PM6/step12/PM6_confs_3Ddescs.sdf'
    save_dir = '2d3d/dataset/PubChemQC-PM6/pmapper'
    smarts_file = os.path.join(save_dir, 'smarts_features.txt')

    bags_dict_pm6 = calc_pm6_descr(conf_file=data_path,
                                   smarts_features=smarts_file,
                                   ncpu=39,
                                   num_descr=[3],
                                   path=save_dir)

    cid_list = []
    bags_list = []
    for cid, bags in bags_dict_pm6.items():
        cid_list.append(cid)
        bags_list.append(bags)

    # Convert bags_list to a DataFrame
    bags_df = pd.DataFrame({'cid': cid_list, 'bags': bags_list})
    # Save the DataFrame to a TSV file
    # bags_df.to_csv(os.path.join(save_dir, 'pmapper_descs_100.tsv'),
    #                sep='\t',
    #                index=False)
    bags_df.to_pickle(os.path.join(save_dir, 'pmapper_descs.pkl'))


def preprocess_pmapper_descs(data_dir: str,
                             y_name: str,
                             test_mode: bool = False
                             ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    data_path_bags = os.path.join(data_dir, 'pmapper', 'pmapper_descs.pkl')
    data_path_pm6 = os.path.join(data_dir, 'step2',
                                 'CHON500noSalt_rotatable_top100k.csv')
    bags_df = pd.read_pickle(data_path_bags)
    if test_mode:
        bags_df = bags_df.head(100)  # for small-data testing
    pm6 = pd.read_csv(data_path_pm6)
    pm6 = pm6.loc[:, [
        'pubchem.cid', 'pubchem.PM6.properties.total dipole moment',
        'pubchem.PM6.properties.energy.alpha.homo',
        'pubchem.PM6.properties.energy.alpha.gap',
        'pubchem.PM6.properties.energy.alpha.lumo',
        'pubchem.PM6.properties.energy.total',
        'pubchem.PM6.properties.enthalpy',
        'pubchem.PM6.openbabel.Canonical SMILES'
    ]]

    pm6.rename(columns={
        'pubchem.PM6.properties.total dipole moment': 'dipoleMoment',
        'pubchem.PM6.properties.energy.alpha.homo': 'homo',
        'pubchem.PM6.properties.energy.alpha.gap': 'gap',
        'pubchem.PM6.properties.energy.alpha.lumo': 'lumo',
        'pubchem.PM6.properties.energy.total': 'energy',
        'pubchem.PM6.properties.enthalpy': 'enthalpy',
        'pubchem.PM6.openbabel.Canonical SMILES': 'CanonicalSMILES'
    },
               inplace=True)

    descs = pd.merge(
        bags_df,
        pm6.drop_duplicates(subset='pubchem.cid').loc[:, 'pubchem.cid':],
        left_on='cid',
        right_on='pubchem.cid').drop(columns='pubchem.cid')
    # sorting
    descs.sort_values('cid', inplace=True)
    if descs[y_name].isna().sum() > 0:
        missing = descs[descs[y_name].isna()]
        logger.info(f'some elements have missing value\n{missing}')
        descs.dropna(subset=[y_name], inplace=True)

    x = descs['bags'].to_numpy()
    y = descs[y_name].to_numpy()
    df_cid = descs['cid']
    return x, y, df_cid


def preprocess_moe_descs(
        data_dir: str,
        y_name: str,
        test_mode: bool = False
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    '''Preprocesses MOE descriptors and returns the preprocessed data.

    Args:
        data_dir: The directory containing the data.
        y_name: The name of the target variable.
        test_mode: Whether to run in test mode.

    Returns:
        x_train_scaled: The scaled training data.
        x_test_scaled: The scaled testing data.
        y_train: The training labels.
        y_test: The testing labels.
    '''
    name_2d = 'cid'
    basefd = data_dir
    idx_descs = pd.read_csv(f'{basefd}/step12/PM6_confs_3Ddescs.csv')
    if test_mode:
        idx_descs = idx_descs.head(100)  # for small-data testing
    idx_descs.rename(columns={'mol2d_idx': 'cid'}, inplace=True)
    descs_columns = [name_2d] + idx_descs.columns[18:-1].to_list()
    descs = idx_descs.loc[:, descs_columns]
    drop_list = [
        'dipoleX',
        'dipoleY',
        'dipoleZ',
        'E_rele',
        'E_rnb',
        'E_rsol',
        'E_rvdw',
        'pmiX',
        'pmiY',
        'pmiZ',
        'dipole',
    ]
    descs.drop(columns=drop_list, inplace=True)
    descs[name_2d] = descs[name_2d].apply(lambda x: x.split('_')[0])

    def convert_to_array(group):
        '''Converts a group of descriptors to a numpy array'''
        return group[descs.columns[1:].to_list()].values

    # apply the function to the groups(cid)
    descs = descs.groupby(name_2d).apply(convert_to_array).reset_index()
    descs['cid'] = descs['cid'].astype(int)

    pm6 = pd.read_csv(f'{basefd}/step2/CHON500noSalt_rotatable_top100k.csv')
    pm6 = pm6.loc[:, [
        'pubchem.cid', 'pubchem.PM6.properties.total dipole moment',
        'pubchem.PM6.properties.energy.alpha.homo',
        'pubchem.PM6.properties.energy.alpha.gap',
        'pubchem.PM6.properties.energy.alpha.lumo',
        'pubchem.PM6.properties.energy.total',
        'pubchem.PM6.properties.enthalpy',
        'pubchem.PM6.openbabel.Canonical SMILES'
    ]]

    pm6.rename(columns={
        'pubchem.PM6.properties.total dipole moment': 'dipoleMoment',
        'pubchem.PM6.properties.energy.alpha.homo': 'homo',
        'pubchem.PM6.properties.energy.alpha.gap': 'gap',
        'pubchem.PM6.properties.energy.alpha.lumo': 'lumo',
        'pubchem.PM6.properties.energy.total': 'energy',
        'pubchem.PM6.properties.enthalpy': 'enthalpy',
        'pubchem.PM6.openbabel.Canonical SMILES': 'CanonicalSMILES'
    },
               inplace=True)

    descs = pd.merge(
        descs,
        pm6.drop_duplicates(subset='pubchem.cid').loc[:, 'pubchem.cid':],
        left_on='cid',
        right_on='pubchem.cid').drop(columns='pubchem.cid')
    # sorting
    descs.sort_values('cid', inplace=True)
    if descs[y_name].isna().sum() > 0:
        missing = descs[descs[y_name].isna()]
        logger.info(f'some elements have missing value\n{missing}')
        descs.dropna(subset=[y_name], inplace=True)

    x = descs[0].to_numpy()
    y = descs[y_name].to_numpy()
    df_cid = descs['cid']
    return x, y, df_cid


def preprocess_single_conf(
        data_dir: str,
        y_name: str,
        test_mode: bool = False
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    '''
    Use mean-aggregated descriptors as input to the neural network instead of MIL to evaluate the performance of the neural network. 
    This allows us to determine whether "RF has poor accuracy and NN has good accuracy" or "MIL approach is superior".
    can use Correct data as input to the MIL model.
    '''
    basefd = data_dir
    # read dataset
    desc_eq_mean = pd.read_csv(f'{basefd}/step14/eq_weight.tsv', sep='\t')
    if test_mode:
        desc_eq_mean = desc_eq_mean.head(10)
    # sort by cid
    desc_eq_mean.sort_values('cid', inplace=True)

    # delete nan
    if desc_eq_mean[y_name].isna().sum() > 0:
        missing = desc_eq_mean[desc_eq_mean[y_name].isna()]
        print(f'some elements have missing value\n{missing}')
        desc_eq_mean.dropna(subset=[y_name], inplace=True)

    # if you use all descs, use below code
    # All descs are 116 descs (= 117 descs - 'dipole')
    desc_list = desc_eq_mean.loc[:, 'ASA':'vsurf_Wp8'].columns.to_list()
    if 'dipole' in desc_list:
        desc_list.remove('dipole')

    x = desc_eq_mean.loc[:, desc_list].to_numpy()
    x = np.array([np.reshape(item, (1, -1)) for item in x])
    y = desc_eq_mean.loc[:, y_name].to_numpy()
    df_cid = desc_eq_mean['cid']
    return x, y, df_cid


def preprocess_correct(
        data_dir: str,
        y_name: str,
        test_mode: bool = False
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    '''
    Use Correct data as input to the MIL model.
    '''
    basefd = data_dir
    # read dataset
    desc_eq_mean = pd.read_csv(f'{basefd}/step14/eq_weight.tsv', sep='\t')
    desc_correct = pd.read_csv(f'{basefd}/step10/correct.tsv', sep='\t')
    # Since the number of rows in decs_correct is different from other Dataframes,
    # delete the rows with CIDs that do not exist in desc_eq_mean.
    desc_correct = desc_correct[desc_correct['cid'].isin(desc_eq_mean['cid'])]

    if test_mode:
        desc_correct = desc_correct.head(10)
    # sort by cid
    desc_correct.sort_values('cid', inplace=True)

    # delete nan
    if desc_correct[y_name].isna().sum() > 0:
        missing = desc_correct[desc_correct[y_name].isna()]
        print(f'some elements have missing value\n{missing}')
        desc_correct.dropna(subset=[y_name], inplace=True)

    # if you use all descs, use below code
    # All descs are 116 descs (= 117 descs - 'dipole')
    desc_list = desc_correct.loc[:, 'ASA':'vsurf_Wp8'].columns.to_list()
    if 'dipole' in desc_list:
        desc_list.remove('dipole')

    x = desc_correct.loc[:, desc_list].to_numpy()
    x = np.array([np.reshape(item, (1, -1)) for item in x])
    y = desc_correct.loc[:, y_name].to_numpy()
    df_cid = desc_correct['cid']
    return x, y, df_cid


def build_model(x_train_scaled,
                x_test_scaled,
                y_train,
                y_test,
                df_cid_train,
                df_cid_test,
                save_dir,
                algorithm,
                pooling='mean',
                n_epoch=500,
                learning_rate=0.001,
                weight_decay=0.0001,
                batch_size=9999999):
    # train model
    logger.info('Model training...')

    algorithms = {
        'BagWrapperMLPRegressor': BagWrapperMLPRegressor,
        'InstanceWrapperMLPRegressor': InstanceWrapperMLPRegressor,
        'BagNetRegressor': BagNetRegressor,
        'InstanceNetRegressor': InstanceNetRegressor,
        'AttentionNetRegressor': AttentionNetRegressor
    }
    #
    n_dim = [x_test_scaled[0].shape[-1]] + [256, 128, 64]

    init_cuda = torch.cuda.is_available()

    if algorithm == 'AttentionNetRegressor':
        det_ndim = n_dim
        net = algorithms[algorithm](ndim=n_dim,
                                    det_ndim=det_ndim,
                                    init_cuda=init_cuda)
        net.fit(
            x_train_scaled,
            y_train,
            n_epoch=n_epoch,
            lr=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            #use for gumbel softmax that is culculated in attention_nets.py
            dropout=1,
            #verbose=True
        )

    else:
        net = algorithms[algorithm](ndim=n_dim,
                                    pool=pooling,
                                    init_cuda=init_cuda)
        net.fit(
            x_train_scaled,
            y_train,
            n_epoch=n_epoch,
            lr=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            #verbose=True
        )

    # save predictions
    logger.info('Save predictions...')

    pred_tr = pd.DataFrame({
        'cid': df_cid_train,
        'predicted': net.predict(x_train_scaled).flatten(),
        'actual': y_train
    })
    pred_te = pd.DataFrame({
        'cid': df_cid_test,
        'predicted': net.predict(x_test_scaled).flatten(),
        'actual': y_test
    })

    pred_tr.to_csv(f'{save_dir}/train_set_predictions.csv', index=False)
    pred_te.to_csv(f'{save_dir}/test_set_predictions.csv', index=False)

    return net.predict(x_train_scaled), net.predict(x_test_scaled)


def outer_cross_validation(data_dir, save_dir, y_name, algorithm, descs_name,
                           n_split_outer, pooling, n_epoch, learning_rate,
                           weight_decay, batch_size, test_mode):

    if test_mode:
        n_split_outer = 2
        n_epoch = 1
    # preprocess data
    logger.info(f'Preprocess data ...')
    if descs_name == 'pmapper':
        x, y, df_cid = preprocess_pmapper_descs(data_dir,
                                                y_name,
                                                test_mode=test_mode)
    elif descs_name == 'moe':
        x, y, df_cid = preprocess_moe_descs(data_dir,
                                            y_name,
                                            test_mode=test_mode)
    elif descs_name == 'agg_mean':
        '''
        If descs_name = 'agg_mean', the aggregated descriptors are input to MIL to evaluate 
        the accuracy of the neural network itself, rather than MIL.
        '''
        x, y, df_cid = preprocess_single_conf(data_dir,
                                              y_name,
                                              test_mode=test_mode)
    elif descs_name == 'correct':
        '''
        If descs_name = 'correct', input Correct data to MIL.
        Verify if MIL can correctly predict using the correct coordinates.
        '''
        x, y, df_cid = preprocess_correct(data_dir,
                                          y_name,
                                          test_mode=test_mode)
    else:
        raise ValueError(f'Invalid descs_name: {descs_name}')

    # Initialize results DataFrame
    results_df = pd.DataFrame(columns=[
        'Fold', 'R2 Train', 'R2 Test', 'MSE Train', 'MSE Test', 'MAE Train',
        'MAE Test'
    ])

    # outer cv : evaluate model
    outer_cv = KFold(n_splits=n_split_outer, shuffle=True, random_state=42)

    # Outer loop
    for i, (train_index, test_index) in enumerate(outer_cv.split(x)):

        # Split data and scaling
        x_train, x_test = x[train_index], x[test_index]
        x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
        y_train, y_test = y[train_index], y[test_index]
        df_cid_train, df_cid_test = df_cid.iloc[train_index], df_cid.iloc[
            test_index]

        # make save dir for each fold
        save_dir_each_fold = os.path.join(save_dir, f'fold_{i}')
        if not os.path.exists(save_dir_each_fold):
            os.makedirs(save_dir_each_fold)

        logger.info(f'fold = {i+1} / {n_split_outer}')

        y_pred_train, y_pred_test = build_model(
            x_train_scaled,
            x_test_scaled,
            y_train,
            y_test,
            df_cid_train,
            df_cid_test,
            save_dir_each_fold,
            algorithm,
            pooling=pooling,
            n_epoch=n_epoch,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
        )

        r2_train, mse_train, mae_train = calcMetrics(y_train, y_pred_train)
        r2_test, mse_test, mae_test = calcMetrics(y_test, y_pred_test)

        logger.info('(Train) R2: {:.3}, MSE: {:.3}, MAE: {:.3}'.format(
            r2_train, mse_train, mae_train))
        logger.info('(Test ) R2: {:.3}, MSE: {:.3}, MAE: {:.3}'.format(
            r2_test, mse_test, mae_test))

        # add results to DataFrame
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

    results_df.to_csv(os.path.join(save_dir, 'results.csv'), index=False)

    # calculate mean and std
    mean_std_df = pd.DataFrame({
        f'Mean_{y_name}_{algorithm}': results_df.mean(),
        f'Std_{y_name}_{algorithm}': results_df.std()
    }).T

    return mean_std_df


def main_single_node():
    '''Main function for running on a Single Node
    '''
    TODAY = '240403'
    DATA_DIR = '2d3d/dataset/PubChemQC-PM6'
    SAVE_DIR_1 = f'3D-MIL-QSSR/output/{TODAY}'
    if not os.path.exists(SAVE_DIR_1):
        os.makedirs(SAVE_DIR_1)

    # Set up logging
    LOG_FILE_PATH = os.path.join(SAVE_DIR_1, 'result.log')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        level=logging.INFO,
        format=LOG_FORMAT,
    )

    y_name_list = ['dipoleMoment', 'homo', 'gap', 'lumo', 'energy', 'enthalpy']

    algorithms = [
        'BagWrapperMLPRegressor', 'InstanceWrapperMLPRegressor',
        'BagNetRegressor', 'InstanceNetRegressor', 'AttentionNetRegressor'
    ]

    pooling = 'mean'

    result_df_list = []

    for y_name in y_name_list:
        for algorithm in algorithms:
            logger.info(f'algorithm: {algorithm}, y_name: {y_name}')
            SAVE_DIR_2 = os.path.join(SAVE_DIR_1, algorithm, y_name)
            if not os.path.exists(SAVE_DIR_2):
                os.makedirs(SAVE_DIR_2)

            logger.info(
                f'y_name = {y_name}, algorithm = {algorithm}, pooling = {pooling}'
            )

            result_df = build_model(
                data_dir=DATA_DIR,
                save_dir=SAVE_DIR_2,
                y_name=y_name,
                algorithm=algorithm,
                pooling=pooling,
                n_epoch=2000,  #500,
                learning_rate=0.001,
                weight_decay=0.0001,
                batch_size=9999999,
            )

            result_df_list.append(result_df)

            # save result for every loop
            result_df_all = pd.concat(result_df_list, axis=0)

            result_df_all.to_csv(os.path.join(SAVE_DIR_1, 'result.csv'),
                                 index=True)

            logger.info(f'{y_name} {algorithm} Finish!')


def main_muti_node(descs_name, algorithm):
    TODAY = '240508'
    TEST_MODE = False
    DATA_DIR = '2d3d/dataset/PubChemQC-PM6'
    SAVE_DIR_1 = os.path.join('3D-MIL-QSSR/output', TODAY, descs_name,
                              algorithm)
    if not os.path.exists(SAVE_DIR_1):
        os.makedirs(SAVE_DIR_1)

    # Set up logging
    LOG_FILE_PATH = os.path.join(SAVE_DIR_1, 'result.log')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        level=logging.INFO,
        format=LOG_FORMAT,
    )

    y_name_list = ['dipoleMoment', 'homo', 'gap', 'lumo', 'energy', 'enthalpy']

    pooling = 'mean'

    mean_std_df_list = []

    for y_name in y_name_list:
        logger.info(f'algorithm: {algorithm}, y_name: {y_name}')
        SAVE_DIR_2 = os.path.join(SAVE_DIR_1, y_name)
        if not os.path.exists(SAVE_DIR_2):
            os.makedirs(SAVE_DIR_2)

        logger.info(
            f'y_name = {y_name}, algorithm = {algorithm}, pooling = {pooling}')

        mean_std_df = outer_cross_validation(
            data_dir=DATA_DIR,
            save_dir=SAVE_DIR_2,
            y_name=y_name,
            algorithm=algorithm,
            descs_name=descs_name,
            n_split_outer=5,
            pooling=pooling,
            n_epoch=1000,  #500,
            learning_rate=0.001,
            weight_decay=0.0001,
            batch_size=128,  #9999999
            test_mode=TEST_MODE,
        )

        mean_std_df_list.append(mean_std_df)

        # save result for every loop
        mean_std_df_all = pd.concat(mean_std_df_list, axis=0)

        mean_std_df_all.to_csv(os.path.join(SAVE_DIR_1, 'result.csv'),
                               index=True)

        logger.info(f'{y_name} {algorithm} Finish!')


if __name__ == '__main__':
    RUN_MULTI = True
    if RUN_MULTI:
        # Change the Algorithm based on the value of sys.argv[1]
        match int(sys.argv[1]):
            case 1:
                ALGORITHM = 'BagWrapperMLPRegressor'
            case 2:
                ALGORITHM = 'InstanceWrapperMLPRegressor'
            case 3:
                ALGORITHM = 'BagNetRegressor'
            case 4:
                ALGORITHM = 'InstanceNetRegressor'
            case 5:
                ALGORITHM = 'AttentionNetRegressor'

        # Change the Algorithm based on the value of sys.argv[1]
        match int(sys.argv[2]):
            case 1:
                DESCS_NAME = 'pmapper'
            case 2:
                DESCS_NAME = 'moe'
            case 3:
                '''If there is only one conformer for each compound,
                MIL can be applied as is.
                '''
                DESCS_NAME = 'agg_mean'
            case 4:
                '''Correct data for MIL
                '''
                DESCS_NAME = 'correct'

        main_muti_node(DESCS_NAME, ALGORITHM)
    else:
        main_single_node()
        #calc_pmapper_descs()
        #preprocess_pmapper_descs()
