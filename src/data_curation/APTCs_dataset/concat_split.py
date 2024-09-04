'''
Since it was found that the accuracy of APTC data varies significantly depending on the splitting method, 
we will concatenate the data first and then split it again for training. 
The compounds to be split are aligned for all aggregations, so we will save them as npy files. 
For APTC1, we will perform 5-fold cross-validation 5 times, so we will save the splitting methods for each fold as npy files. 
For APTC2, we will perform LOOCV (Leave-One-Out Cross-Validation) once, so we will save the splitting method as an npy file. 
Please note that only non-aggregated data will be split compound-wise.
'''

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def concat_moe_pmapper_descs(data_dir, save_dir):
    '''
    Concatenate the aggregated descriptors of MOE and pmapper.
    Add 'cid' column to identify the catalyst in the original data.
    '''
    # Boltzmann Wight
    tr_weig = pd.read_csv(f'{data_dir}_train/HFweight.tsv',
                          sep='\t',
                          index_col=0)
    te_weig = pd.read_csv(f'{data_dir}_test/HFweight.tsv',
                          sep='\t',
                          index_col=0)
    tr_weig['cid'] = 'tr_' + tr_weig.index.astype(str)
    te_weig['cid'] = 'te_' + te_weig.index.astype(str)
    df_weig = pd.concat([tr_weig, te_weig], ignore_index=True)
    df_weig.to_csv(f'{save_dir}/HFweight.tsv', sep='\t', index=False)
    # Mean
    tr_mean = pd.read_csv(f'{data_dir}_train/eq_weight.tsv',
                          sep='\t',
                          index_col=0)
    te_mean = pd.read_csv(f'{data_dir}_test/eq_weight.tsv',
                          sep='\t',
                          index_col=0)
    tr_mean['cid'] = 'tr_' + tr_mean.index.astype(str)
    te_mean['cid'] = 'te_' + te_mean.index.astype(str)
    df_mean = pd.concat([tr_mean, te_mean], ignore_index=True)
    df_mean.to_csv(f'{save_dir}/eq_weight.tsv', sep='\t', index=False)

    # Global Minimum
    tr_mini = pd.read_csv(f'{data_dir}_train/1conf.tsv', sep='\t', index_col=0)
    te_mini = pd.read_csv(f'{data_dir}_test/1conf.tsv', sep='\t', index_col=0)
    tr_mini['cid'] = 'tr_' + tr_mini.index.astype(str)
    te_mini['cid'] = 'te_' + te_mini.index.astype(str)
    df_mini = pd.concat([tr_mini, te_mini], ignore_index=True)
    df_mini.to_csv(f'{save_dir}/1conf.tsv', sep='\t', index=False)

    # No Aggregation
    tr_noag = pd.read_csv(f'{data_dir}_train/descs_no_agg.csv')
    te_noag = pd.read_csv(f'{data_dir}_test/descs_no_agg.csv')
    if data_dir.split('/')[-2] == 'step3':  # MOE
        tr_noag['cid'] = 'tr_' + tr_noag['cid'].astype(str)
        te_noag['cid'] = 'te_' + te_noag['cid'].astype(str)
    if data_dir.split('/')[-2] == 'step6':  # pmapper
        tr_noag['cid'] = 'tr_' + tr_noag['comp_id'].astype(str)
        te_noag['cid'] = 'te_' + te_noag['comp_id'].astype(str)
        tr_noag = tr_noag.drop(columns=['comp_id'])
        te_noag = te_noag.drop(columns=['comp_id'])
    df_noag = pd.concat([tr_noag, te_noag], ignore_index=True)
    df_noag.to_csv(f'{save_dir}/descs_no_agg.csv', index=False)

    # Random 0
    tr_ran0 = pd.read_csv(f'{data_dir}_train/random.tsv',
                          sep='\t',
                          index_col=0)
    te_ran0 = pd.read_csv(f'{data_dir}_test/random.tsv', sep='\t', index_col=0)
    tr_ran0['cid'] = 'tr_' + tr_ran0.index.astype(str)
    te_ran0['cid'] = 'te_' + te_ran0.index.astype(str)
    df_ran0 = pd.concat([tr_ran0, te_ran0], ignore_index=True)
    df_ran0.to_csv(f'{save_dir}/random.tsv', sep='\t', index=False)

    # Random 1
    tr_ran1 = pd.read_csv(f'{data_dir}_train/random_12.tsv',
                          sep='\t',
                          index_col=0)
    te_ran1 = pd.read_csv(f'{data_dir}_test/random_12.tsv',
                          sep='\t',
                          index_col=0)
    tr_ran1['cid'] = 'tr_' + tr_ran1.index.astype(str)
    te_ran1['cid'] = 'te_' + te_ran1.index.astype(str)
    df_ran1 = pd.concat([tr_ran1, te_ran1], ignore_index=True)
    df_ran1.to_csv(f'{save_dir}/random_12.tsv', sep='\t', index=False)

    # Random 2
    tr_ran2 = pd.read_csv(f'{data_dir}_train/random_22.tsv',
                          sep='\t',
                          index_col=0)
    te_ran2 = pd.read_csv(f'{data_dir}_test/random_22.tsv',
                          sep='\t',
                          index_col=0)
    tr_ran2['cid'] = 'tr_' + tr_ran2.index.astype(str)
    te_ran2['cid'] = 'te_' + te_ran2.index.astype(str)
    df_ran2 = pd.concat([tr_ran2, te_ran2], ignore_index=True)
    df_ran2.to_csv(f'{save_dir}/random_22.tsv', sep='\t', index=False)

    # Random 3
    tr_ran3 = pd.read_csv(f'{data_dir}_train/random_32.tsv',
                          sep='\t',
                          index_col=0)
    te_ran3 = pd.read_csv(f'{data_dir}_test/random_32.tsv',
                          sep='\t',
                          index_col=0)
    tr_ran3['cid'] = 'tr_' + tr_ran3.index.astype(str)
    te_ran3['cid'] = 'te_' + te_ran3.index.astype(str)
    df_ran3 = pd.concat([tr_ran3, te_ran3], ignore_index=True)
    df_ran3.to_csv(f'{save_dir}/random_32.tsv', sep='\t', index=False)

    # Random 4
    tr_ran4 = pd.read_csv(f'{data_dir}_train/random_52.tsv',
                          sep='\t',
                          index_col=0)
    te_ran4 = pd.read_csv(f'{data_dir}_test/random_52.tsv',
                          sep='\t',
                          index_col=0)
    tr_ran4['cid'] = 'tr_' + tr_ran4.index.astype(str)
    te_ran4['cid'] = 'te_' + te_ran4.index.astype(str)
    df_ran4 = pd.concat([tr_ran4, te_ran4], ignore_index=True)
    df_ran4.to_csv(f'{save_dir}/random_52.tsv', sep='\t', index=False)


def concat_unimol_input(data_dir, save_dir):
    '''
    Concatenate the input data of Unimol.
    Add 'cid' column to identify the catalyst in the original data.
    '''
    # Global Minimum
    tr_mini = pd.read_csv(f'{data_dir}/train/conformers/catalyst_1conf.csv')
    te_mini = pd.read_csv(f'{data_dir}/test/conformers/catalyst_1conf.csv')
    tr_mini['cid'] = 'tr_' + tr_mini.index.astype(str)
    te_mini['cid'] = 'te_' + te_mini.index.astype(str)
    df_mini = pd.concat([tr_mini, te_mini], ignore_index=True)
    df_mini.to_csv(f'{save_dir}/1conf.csv', index=False)

    # No Aggregation
    tr_noag = pd.read_csv(f'{data_dir}/train/conformers/catalyst_allconf.csv')
    te_noag = pd.read_csv(f'{data_dir}/test/conformers/catalyst_allconf.csv')
    tr_noag['cid'] = 'tr_' + tr_noag['cid'].astype(str)
    te_noag['cid'] = 'te_' + te_noag['cid'].astype(str)
    df_noag = pd.concat([tr_noag, te_noag], ignore_index=True)
    df_noag.to_csv(f'{save_dir}/descs_no_agg.csv', index=False)


def concat_unimol_rep(data_dir, save_dir):
    '''
    Concatenate and load the Unimol representations.
    Add 'cid' column to identify the catalyst in the original data.
    '''
    # Global Minimum
    tr_mini = pd.read_csv(f'{data_dir}/train/repr_1conf.csv')
    te_mini = pd.read_csv(f'{data_dir}/test/repr_1conf.csv')
    tr_mini['cid'] = 'tr_' + tr_mini['cid'].astype(str)
    te_mini['cid'] = 'te_' + te_mini['cid'].astype(str)
    df_mini = pd.concat([tr_mini, te_mini], ignore_index=True)
    df_mini.to_csv(f'{save_dir}/1conf.csv', index=False)

    # No Aggregation
    tr_noag = pd.read_csv(f'{data_dir}/train/repr_allconf.csv')
    te_noag = pd.read_csv(f'{data_dir}/test/repr_allconf.csv')
    tr_noag['cid'] = 'tr_' + tr_noag['cid'].astype(str)
    te_noag['cid'] = 'te_' + te_noag['cid'].astype(str)
    df_noag = pd.concat([tr_noag, te_noag], ignore_index=True)
    df_noag.to_csv(f'{save_dir}/descs_no_agg.csv', index=False)


def concat_2d_descs(data_path, save_dir):
    '''
    Concatenate and load the four 2D descriptors.
    Add 'cid' column to identify the catalyst in the original data.
    '''
    tr_2d = pd.read_pickle(f'{data_path}_train.pkl')
    te_2d = pd.read_pickle(f'{data_path}_test.pkl')

    # change 'ECFP' to '0'~'1023'
    desc_tr_ecfp = pd.DataFrame(tr_2d['ECFP'].tolist(),
                                columns=[str(i) for i in range(1024)])
    tr_2d = pd.concat([tr_2d.drop(columns=['ECFP']), desc_tr_ecfp], axis=1)

    desc_te_ecfp = pd.DataFrame(te_2d['ECFP'].tolist(),
                                columns=[str(i) for i in range(1024)])
    te_2d = pd.concat([te_2d.drop(columns=['ECFP']), desc_te_ecfp], axis=1)
    tr_2d['cid'] = 'tr_' + tr_2d.index.astype(str)
    te_2d['cid'] = 'te_' + te_2d.index.astype(str)
    df_2d = pd.concat([tr_2d, te_2d], ignore_index=True)
    df_2d.to_csv(f'{save_dir}/descs_2d.tsv', sep='\t', index=False)


def concat_pharm_2d(data_path, save_dir):
    '''
    Concatenate and load the pharm2D descriptors.
    Add 'cid' column to identify the catalyst in the original data.
    '''
    tr_2d = pd.read_pickle(f'{data_path}_train.pkl')
    te_2d = pd.read_pickle(f'{data_path}_test.pkl')

    # change 'ECFP' to '0'~'1023'
    desc_tr_ecfp = pd.DataFrame(tr_2d['ECFP'].tolist(),
                                columns=[str(i) for i in range(2048)])
    tr_2d = pd.concat([tr_2d.drop(columns=['ECFP']), desc_tr_ecfp], axis=1)

    desc_te_ecfp = pd.DataFrame(te_2d['ECFP'].tolist(),
                                columns=[str(i) for i in range(2048)])
    te_2d = pd.concat([te_2d.drop(columns=['ECFP']), desc_te_ecfp], axis=1)
    tr_2d['cid'] = 'tr_' + tr_2d.index.astype(str)
    te_2d['cid'] = 'te_' + te_2d.index.astype(str)
    df_2d = pd.concat([tr_2d, te_2d], ignore_index=True)
    df_2d.to_csv(f'{save_dir}/descs_2d.tsv', sep='\t', index=False)


def make_index_for_aptc1(save_dir):
    '''
    For APTC1, we will perform 5-fold cross-validation 5 times, 
    so we will save the splitting methods for each fold as npy files.
    '''
    df = pd.read_csv('2d3d/dataset/APTC/step7/moe/aptc1/eq_weight.tsv',
                     sep='\t')
    seeds = [12, 22, 32, 42, 52]
    for seed in seeds:
        save_dir_seed = f'{save_dir}/index/aptc1/seed_{seed}'
        os.makedirs(save_dir_seed, exist_ok=True)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
            arr_train = df.loc[train_idx, 'cid'].to_list()
            arr_test = df.loc[test_idx, 'cid'].to_list()
            np.save(f'{save_dir_seed}/train_{fold}.npy', arr_train)
            np.save(f'{save_dir_seed}/test_{fold}.npy', arr_test)


def make_index_for_aptc2(save_dir):
    '''24/08/02
    For APTC2, we will perform LOOCV (Leave-One-Out Cross-Validation) once, 
    so we will save the splitting method as an npy file.
    '''
    df = pd.read_csv('2d3d/dataset/APTC/step7/moe/aptc2/eq_weight.tsv',
                     sep='\t')
    save_dir = f'{save_dir}/index/aptc2'
    os.makedirs(save_dir, exist_ok=True)
    kf = KFold(n_splits=df.shape[0], shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        arr_train = df.loc[train_idx, 'cid'].to_list()
        arr_test = df.loc[test_idx, 'cid'].to_list()
        np.save(f'{save_dir}/train_{fold}.npy', arr_train)
        np.save(f'{save_dir}/test_{fold}.npy', arr_test)


def test_split():
    '''
    Check if the created indexes are correct.
    Verify if the data can be properly split using all the indexes.
    '''
    # read all made data
    base_dir = '2d3d/dataset/APTC/step7'
    ecfp_bit = pd.read_csv(f'{base_dir}/ecfp_bit/aptc1/descs_2d.tsv', sep='\t')
    ecfp_count = pd.read_csv(f'{base_dir}/ecfp_count/aptc1/descs_2d.tsv',
                             sep='\t')
    fcfp_bit = pd.read_csv(f'{base_dir}/fcfp_bit/aptc1/descs_2d.tsv', sep='\t')
    fcfp_count = pd.read_csv(f'{base_dir}/fcfp_count/aptc1/descs_2d.tsv',
                             sep='\t')
    pharm_2d = pd.read_csv(f'{base_dir}/pharm_2d/aptc1/descs_2d.tsv', sep='\t')
    moe_weig = pd.read_csv(f'{base_dir}/moe/aptc1/HFweight.tsv', sep='\t')
    moe_mean = pd.read_csv(f'{base_dir}/moe/aptc1/eq_weight.tsv', sep='\t')
    moe_mini = pd.read_csv(f'{base_dir}/moe/aptc1/1conf.tsv', sep='\t')
    moe_noag = pd.read_csv(f'{base_dir}/moe/aptc1/descs_no_agg.csv')
    moe_ran0 = pd.read_csv(f'{base_dir}/moe/aptc1/random.tsv', sep='\t')
    moe_ran1 = pd.read_csv(f'{base_dir}/moe/aptc1/random_12.tsv', sep='\t')
    moe_ran2 = pd.read_csv(f'{base_dir}/moe/aptc1/random_22.tsv', sep='\t')
    moe_ran3 = pd.read_csv(f'{base_dir}/moe/aptc1/random_32.tsv', sep='\t')
    moe_ran4 = pd.read_csv(f'{base_dir}/moe/aptc1/random_52.tsv', sep='\t')
    pm_weig = pd.read_csv(f'{base_dir}/pmapper/aptc1/HFweight.tsv', sep='\t')
    pm_mean = pd.read_csv(f'{base_dir}/pmapper/aptc1/eq_weight.tsv', sep='\t')
    pm_mini = pd.read_csv(f'{base_dir}/pmapper/aptc1/1conf.tsv', sep='\t')
    pm_noag = pd.read_csv(f'{base_dir}/pmapper/aptc1/descs_no_agg.csv')
    pm_ran0 = pd.read_csv(f'{base_dir}/pmapper/aptc1/random.tsv', sep='\t')
    pm_ran1 = pd.read_csv(f'{base_dir}/pmapper/aptc1/random_12.tsv', sep='\t')
    pm_ran2 = pd.read_csv(f'{base_dir}/pmapper/aptc1/random_22.tsv', sep='\t')
    pm_ran3 = pd.read_csv(f'{base_dir}/pmapper/aptc1/random_32.tsv', sep='\t')
    pm_ran4 = pd.read_csv(f'{base_dir}/pmapper/aptc1/random_52.tsv', sep='\t')
    unimol_mini = pd.read_csv(f'{base_dir}/unimol/aptc1/1conf.csv')
    unimol_noag = pd.read_csv(f'{base_dir}/unimol/aptc1/descs_no_agg.csv')
    unirep_mini = pd.read_csv(f'{base_dir}/unirep/aptc1/1conf.csv')
    unirep_noag = pd.read_csv(f'{base_dir}/unirep/aptc1/descs_no_agg.csv')

    data_list = [
        ecfp_bit, ecfp_count, fcfp_bit, fcfp_count, pharm_2d, moe_weig,
        moe_mean, moe_mini, moe_noag, moe_ran0, moe_ran1, moe_ran2, moe_ran3,
        moe_ran4, pm_weig, pm_mean, pm_mini, pm_noag, pm_ran0, pm_ran1,
        pm_ran2, pm_ran3, pm_ran4, unimol_mini, unimol_noag, unirep_mini,
        unirep_noag
    ]

    # read all index
    index_dir = '2d3d/dataset/APTC/step7/index/aptc1'
    seeds = ['seed_12', 'seed_22', 'seed_32', 'seed_42', 'seed_52']
    folds = [0, 1, 2, 3, 4]
    for seed in seeds:
        for fold in folds:
            tr_cid = np.load(f'{index_dir}/{seed}/train_{fold}.npy')
            te_cid = np.load(f'{index_dir}/{seed}/test_{fold}.npy')
            for data in data_list:
                tr_data = data[data['cid'].isin(tr_cid)]
                te_data = data[data['cid'].isin(te_cid)]
                print(
                    f'{seed} {fold} {data.shape[0]} {tr_data.shape[0]} {te_data.shape[0]}'
                )


def concat_and_save(save_sir):
    '''
    Concatenate and save all input data related to APTC.
    '''
    # concat MOE APTC1
    DATA_DIR_MOE_1 = '2d3d/dataset/APTC/step3/aptc1'
    SAVE_DIR_MOE_1 = f'{save_sir}/moe/aptc1'
    os.makedirs(SAVE_DIR_MOE_1, exist_ok=True)
    concat_moe_pmapper_descs(DATA_DIR_MOE_1, SAVE_DIR_MOE_1)
    # concat MOE APTC2
    DATA_DIR_MOE_2 = '2d3d/dataset/APTC/step3/aptc2'
    SAVE_DIR_MOE_2 = f'{save_sir}/moe/aptc2'
    os.makedirs(SAVE_DIR_MOE_2, exist_ok=True)
    concat_moe_pmapper_descs(DATA_DIR_MOE_2, SAVE_DIR_MOE_2)
    # concat pmapper APTC1
    DATA_DIR_PM_1 = '2d3d/dataset/APTC/step6/aptc1'
    SAVE_DIR_PM_1 = f'{save_sir}/pmapper/aptc1'
    os.makedirs(SAVE_DIR_PM_1, exist_ok=True)
    concat_moe_pmapper_descs(DATA_DIR_PM_1, SAVE_DIR_PM_1)
    # concat pmapper APTC2
    DATA_DIR_PM_2 = '2d3d/dataset/APTC/step6/aptc2'
    SAVE_DIR_PM_2 = f'{save_sir}/pmapper/aptc2'
    os.makedirs(SAVE_DIR_PM_2, exist_ok=True)
    concat_moe_pmapper_descs(DATA_DIR_PM_2, SAVE_DIR_PM_2)
    # concat unimol input APTC1
    DATA_DIR_UNIMOL_1 = '2d3d/dataset/APTC/step1/aptc-1'
    SAVE_DIR_UNIMOL_1 = f'{save_sir}/unimol/aptc1'
    os.makedirs(SAVE_DIR_UNIMOL_1, exist_ok=True)
    concat_unimol_input(DATA_DIR_UNIMOL_1, SAVE_DIR_UNIMOL_1)
    # concat unimol input APTC2
    DATA_DIR_UNIMOL_2 = '2d3d/dataset/APTC/step1/aptc-2'
    SAVE_DIR_UNIMOL_2 = f'{save_sir}/unimol/aptc2'
    os.makedirs(SAVE_DIR_UNIMOL_2, exist_ok=True)
    concat_unimol_input(DATA_DIR_UNIMOL_2, SAVE_DIR_UNIMOL_2)
    # concat Uni-Mol Representaion APTC1
    DATA_DIR_UNIREP_1 = '2d3d/dataset/APTC/step4/aptc_1'
    SAVE_DIR_UNIREP_1 = f'{save_sir}/unirep/aptc1'
    os.makedirs(SAVE_DIR_UNIREP_1, exist_ok=True)
    concat_unimol_rep(DATA_DIR_UNIREP_1, SAVE_DIR_UNIREP_1)
    # concat Uni-Mol Representaion APTC2
    DATA_DIR_UNIREP_2 = '2d3d/dataset/APTC/step4/aptc_2'
    SAVE_DIR_UNIREP_2 = f'{save_sir}/unirep/aptc2'
    os.makedirs(SAVE_DIR_UNIREP_2, exist_ok=True)
    concat_unimol_rep(DATA_DIR_UNIREP_2, SAVE_DIR_UNIREP_2)
    # concat 2D descriptors
    datasets = ['aptc1', 'aptc2']
    descs = ['ecfp_bit', 'fcfp_bit', 'ecfp_count', 'fcfp_count']
    for dataset in datasets:
        for desc in descs:
            DATA_PATH_2D = f'2d3d/dataset/APTC/step1/{desc}/{dataset}_{desc}'
            SAVE_DIR_2D = f'{save_sir}/{desc}/{dataset}'
            os.makedirs(SAVE_DIR_2D, exist_ok=True)
            concat_2d_descs(DATA_PATH_2D, SAVE_DIR_2D)
    # concat pharm 2d
    for dataset in datasets:
        DATA_PATH_2D = f'2d3d/dataset/APTC/step1/pharm_2d/{dataset}_pharm_2d'
        SAVE_DIR_2D = f'{save_sir}/pharm_2d/{dataset}'
        os.makedirs(SAVE_DIR_2D, exist_ok=True)
        concat_pharm_2d(DATA_PATH_2D, SAVE_DIR_2D)


if __name__ == '__main__':
    SAVE_DIR = '2d3d/dataset/APTC/step7'
    os.makedirs(SAVE_DIR, exist_ok=True)
    concat_and_save(SAVE_DIR)
    # make_index_for_aptc1(SAVE_DIR)
    # make_index_for_aptc2(SAVE_DIR)
    # test_split()
