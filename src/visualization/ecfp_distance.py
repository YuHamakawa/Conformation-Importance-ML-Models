'''
Calculate the distance between the ECFP of compounds in the training set and the ECFP of compounds in the test set by performing a 5-fold cross-validation and training an RF model. This will help identify similar compounds that are present in both the training and test sets.
Specifically, calculate the similarity between each compound in the training set and the ECFP of the training set, and plot the histogram of the closest similarity values.
'''

import os
import sys
import tempfile

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xlsxwriter
from rdkit import Chem, DataStructs
from rdkit.Chem.Draw import MolToImage
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm


def AddHeadFormat(workbook):
    """
    Add a style to the workbook that is used in the header of the table
    """
    format = workbook.add_format(
        dict(bold=True, align='center', valign='vcenter', size=12))
    format.set_bg_color('white')
    format.set_border_color('black')
    format.set_border()
    return format


def AddDataFormat(workbook):
    """
    Add a style to the workbook data place 
    """
    format = workbook.add_format(
        dict(bold=False, align='center', valign='vcenter', size=15))
    format.set_text_wrap()
    format.set_bg_color('white')
    format.set_border_color('black')
    format.set_border()

    return format


def writeImageToFile(mol, ofile, width=300, height=300):
    """
    write molecule image to a file with formatting options
    """
    img = MolToImage(mol, (width, height))
    img.save(ofile, bitmap_format='png')


def WriteDataFrameSmilesToXls(pd_table,
                              smiles_colnames,
                              out_filename,
                              smirks_colnames=None,
                              max_mols=10000,
                              retain_smiles_col=False,
                              use_romol=False):
    """
    Write panads DataFrame containing smiles as molcular image using rdkit.

    input:
    ------
    pd_table:
    smiles_colnames: must set the smiles column names where smiles are converted to images
    max_mol: For avoid generating too much data 
    out_filename: output file name 
    smirks_colname: reaction smiles (smirks) column name which is decomposed to left and right parts to visualization
    retain_smiles_col: (retaining SMIELS columns or mot )
    output:
    ------
    None: 

    """
    if isinstance(smiles_colnames, str):
        smiles_colnames = [smiles_colnames]

    if smiles_colnames is None:
        smiles_colnames = ['']

    if use_romol:
        pd_table[smiles_colnames] = pd_table[smiles_colnames].applymap(
            lambda x: Chem.MolToSmiles(x) if x is not None else '')

    if retain_smiles_col:
        pd_smiles = pd_table[smiles_colnames].copy()
        pd_smiles.columns = ['{}_SMI'.format(s) for s in smiles_colnames]
        pd_table = pd.concat([pd_table, pd_smiles], axis=1)

    if smirks_colnames is not None:
        if isinstance(smirks_colnames, str):
            smirks_colnames = [smirks_colnames]
        for smirks_col in smirks_colnames:
            lname, midname, rname = f'left_{smirks_col}', f'middle_{smirks_col}', f'right_{smirks_col}'
            pd_table[lname] = pd_table[smirks_col].str.split('>').str[0]
            pd_table[midname] = pd_table[smirks_col].str.split('>').str[1]
            pd_table[rname] = pd_table[smirks_col].str.split('>').str[2]

            # check the middle part (if no condition for all the smirks, remove it)
            if (pd_table[midname] == '').all():
                del pd_table[midname]
                smirks_names = [lname, rname]
            else:
                smirks_names = [lname, midname, rname]
            smiles_colnames.extend(smirks_names)

    # if the column contain objects then it convers to string
    array_columns = [
        col for col in pd_table.columns
        if isinstance(pd_table[col].iloc[0], np.ndarray)
    ]
    pd_table[array_columns] = pd_table[array_columns].applymap(
        lambda x: str(x))

    # set up depiction option
    width, height = 250, 250

    if not isinstance(pd_table, pd.DataFrame):
        raise ValueError("pd_table must be pandas DataFrame")

    if len(pd_table) > max_mols:
        raise ValueError('maximum number of rows is set to %d but input %d' %
                         (max_mols, len(pd_table)))

    workbook = xlsxwriter.Workbook(out_filename)
    worksheet = workbook.add_worksheet()

    # Set header to a workbook
    headformat = AddHeadFormat(workbook)
    dataformat = AddDataFormat(workbook)

    # Estimate the width of columns
    maxwidths = dict()

    if not pd_table.index.name:
        pd_table.index.name = 'index'

    for column in pd_table:
        if column in smiles_colnames:  # for structure column
            maxwidths[column] = width * 0.15  # I do not know why this works
        else:
            if pd_table[column].dtype == list:  # list to str
                pd_table[column] = pd_table[column].apply(str)
            l_txt = pd_table[column].apply(lambda x: len(str(x)))

            l_len = np.max(l_txt)
            l_len = max(l_len, len(str(column)))
            maxwidths[column] = l_len * 1.2  # misterious scaling

    # Generate header (including index part)
    row, col = 0, 0
    worksheet.set_row(row, None, headformat)
    worksheet.set_column(col, col, len(str(pd_table.index.name)))
    worksheet.write(row, col, pd_table.index.name)

    for colname in pd_table:
        col += 1
        worksheet.set_column(col, col, maxwidths[colname])
        worksheet.write(row, col, colname)

    # temporary folder for storing figs
    with tempfile.TemporaryDirectory() as tmp_dir:

        # Generate the data
        for idx, val in pd_table.iterrows():
            row += 1
            worksheet.set_row(row, height * 0.75, dataformat)

            col = 0
            worksheet.write(row, col, idx)

            # contents
            for cname, item in val.items():
                col += 1
                if cname in smiles_colnames:
                    fname = os.path.join(tmp_dir, '%d_%d.png' % (row, col))
                    if isinstance(item, str):
                        mol = Chem.MolFromSmiles(item)
                    else:
                        mol = item
                    if mol is not None:
                        writeImageToFile(mol, fname, int(width * 0.9),
                                         int(height * 0.9))
                        worksheet.insert_image(
                            row, col, fname,
                            dict(object_position=1, x_offset=1, y_offset=1))
                else:
                    try:
                        worksheet.write(row, col, item)
                    except:
                        continue
        workbook.close()


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


def convert_to_bitvect(ecfp):
    '''Convert ECFP to bit vector in order to calculate Tanimoto Similarity.
    Since it is counting, there may be bits with a value of 1 or more.
    Convert bits with a value of 1 or more to 1.
    '''
    bitvect = DataStructs.ExplicitBitVect(len(ecfp))
    for i, bit in enumerate(ecfp):
        # countなので1以上のBitもある．
        if bit >= 1:
            bitvect.SetBit(i)
    return bitvect


def calc_tanimoto(data_path, index_dir, save_dir, cv=5, test_mode=False):
    '''Create a TSV file that contains CID, Tanimoto Similarity, and the difference in property values.
    '''
    # load ecfp
    df_ecfp = pd.read_csv(data_path, sep='\t')
    # drop unnecessary columns (cid and ECFP are remained)
    # drop_list = ['smiles', 'dipoleMoment', 'homo', 'gap', 'lumo', 'energy',
    #    'enthalpy', 'CanonicalSMILES']
    # drop_list = ['dipoleMoment', 'homo', 'gap', 'lumo', 'energy',
    #    'enthalpy', 'CanonicalSMILES']
    drop_list = ['CanonicalSMILES']
    df_ecfp.drop(columns=drop_list, inplace=True)

    # Load the CIDs (Compound IDs) for each fold
    for i in range(cv):
        print(f'===========fold {i}===========')
        save_dir_fold = os.path.join(save_dir, f'fold_{i}')
        if not os.path.exists(save_dir_fold):
            os.makedirs(save_dir_fold)

        train_cid = np.load(f'{index_dir}/train_cid_fold_{i}.npy')
        test_cid = np.load(f'{index_dir}/test_cid_fold_{i}.npy')

        # extract train and test for each fold
        df_train = df_ecfp[df_ecfp['cid'].isin(train_cid)].copy()
        df_test = df_ecfp[df_ecfp['cid'].isin(test_cid)].copy()
        # for small test
        if test_mode:
            df_train = df_train.head(100)
            df_test = df_test.head(100)
        # convert to bit vector
        print('convert to bit vector...')
        df_train_bitvects = [
            convert_to_bitvect(ecfp)
            for ecfp in df_train.loc[:, '0':'2047'].values
        ]
        df_test_bitvects = [
            convert_to_bitvect(ecfp)
            for ecfp in df_test.loc[:, '0':'2047'].values
        ]

        # Calculate the maximum Tanimoto coefficient between each test compound and the compounds in the training set
        max_tanimoto_pairs = []
        for idx, test_bitvect in enumerate(tqdm(df_test_bitvects)):
            tanimotos = DataStructs.BulkTanimotoSimilarity(
                test_bitvect, df_train_bitvects)
            max_tanimoto = max(tanimotos)
            max_tanimoto_index = tanimotos.index(max_tanimoto)
            test_cid = df_test.iloc[idx]['cid']
            # calculate difference of properties between test and train
            dipole_diff = np.abs(
                df_test.iloc[idx]['dipoleMoment'] -
                df_train.iloc[max_tanimoto_index]['dipoleMoment'])
            homo_diff = np.abs(df_test.iloc[idx]['homo'] -
                               df_train.iloc[max_tanimoto_index]['homo'])
            gap_diff = np.abs(df_test.iloc[idx]['gap'] -
                              df_train.iloc[max_tanimoto_index]['gap'])
            lumo_diff = np.abs(df_test.iloc[idx]['lumo'] -
                               df_train.iloc[max_tanimoto_index]['lumo'])
            energy_diff = np.abs(df_test.iloc[idx]['energy'] -
                                 df_train.iloc[max_tanimoto_index]['energy'])
            enthalpy_diff = np.abs(
                df_test.iloc[idx]['enthalpy'] -
                df_train.iloc[max_tanimoto_index]['enthalpy'])
            max_tanimoto_pairs.append(
                (test_cid, df_test.iloc[idx]['smiles'],
                 df_train.iloc[max_tanimoto_index]['smiles'], max_tanimoto,
                 dipole_diff, homo_diff, gap_diff, lumo_diff, energy_diff,
                 enthalpy_diff))
        # After calculating max_tanimoto_smiles_pairs
        df_pairs = pd.DataFrame(max_tanimoto_pairs,
                                columns=[
                                    'test_cid', 'test_smiles', 'train_smiles',
                                    'tanimoto', 'dipole_diff', 'homo_diff',
                                    'gap_diff', 'lumo_diff', 'energy_diff',
                                    'enthalpy_diff'
                                ])
        # Save the dataframe as a TSV file
        df_pairs.to_csv(f'{save_dir_fold}/max_tanimoto_pairs.tsv',
                        sep='\t',
                        index=False)


def plot_tanimoto_hist(max_tanimotos, save_dir):
    '''Create a plot. The x-axis represents Tanimoto Similarity, and the y-axis represents Count.
    '''
    fig = plt.figure()
    plt.hist(max_tanimotos, bins=100)
    plt.xlabel('Tanimoto Similarity', fontsize=18)
    plt.ylabel('Number of Test Compounds', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    # plt.title('Maximum similarity of test to training compounds', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tanimoto_similarity.png')


def plot_tanimoto_vs_property_diff(df_pairs, save_dir):
    '''Create a plot. The x-axis represents Tanimoto Similarity, and the y-axis represents the difference in property values.
    By examining the difference in property values in the region where Tanimoto Similarity is close to 1, we can infer the ease of predicting each property.
    '''
    properties = [
        'dipole_diff', 'homo_diff', 'gap_diff', 'lumo_diff', 'energy_diff',
        'enthalpy_diff'
    ]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    for i, prop in enumerate(properties):
        ax = axs[i // 3, i % 3]
        sns.scatterplot(x='tanimoto', y=prop, data=df_pairs, ax=ax)
        ax.set_title(f'{prop} - Tanimoto')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/tanimoto_vs_properties_diff.png')


def plot_all(data_dir, cv=5):
    '''Execute the two plot functions.
    '''
    for i in tqdm(range(cv)):
        data_path = f'{data_dir}/fold_{i}/max_tanimoto_pairs.tsv'
        save_dir = f'{data_dir}/fold_{i}'
        df_pairs = pd.read_csv(data_path, sep='\t')
        plot_tanimoto_hist(df_pairs.loc[:, 'tanimoto'], save_dir)
        # plot_tanimoto_vs_property_diff(df_pairs, save_dir)


def tsv2xlsx(data_dir, cv=5, tanimoto_thres=0.99):
    '''Load the TSV file created by calc_tanimoto and plot the pairs of compounds with a similarity above the threshold in an Excel file.
    '''
    for i in tqdm(range(cv)):
        data_path = f'{data_dir}/fold_{i}/max_tanimoto_pairs.tsv'
        save_path = f'{data_dir}/fold_{i}/max_tanimoto_pairs_{tanimoto_thres}.xlsx'
        mols = pd.read_csv(data_path, sep='\t')
        # extract above tanimoto_thresholdings
        mols = mols[mols.loc[:, 'tanimoto'] >= tanimoto_thres]
        smicols = ['test_smiles', 'train_smiles']
        WriteDataFrameSmilesToXls(mols.sample(n=100, random_state=42),
                                  smicols,
                                  save_path,
                                  retain_smiles_col=True)


def tsv2xlsx4fig(data_dir, cv=5):
    '''For use in Figure of paper
    Read the TSV file created by calc_tanimoto and plot the pairs of compounds with a similarity above the threshold in an Excel file.
    '''
    for i in tqdm(range(cv)):
        data_path = f'{data_dir}/fold_{i}/max_tanimoto_pairs.tsv'
        save_path = f'{data_dir}/fold_{i}/max_tanimoto_pairs4fig.xlsx'
        mols = pd.read_csv(data_path, sep='\t')
        # extract above tanimoto_thresholdings
        mol_df = pd.concat([
            mols[mols.loc[:, 'tanimoto'] == 0.9].sample(n=3, random_state=42),
            mols[mols.loc[:, 'tanimoto'] == 0.8].sample(n=3, random_state=42),
            mols[mols.loc[:, 'tanimoto'] == 0.7].sample(n=3, random_state=42),
            mols[mols.loc[:, 'tanimoto'] == 0.6].sample(n=3, random_state=42),
            mols[mols.loc[:, 'tanimoto'] == 0.5].sample(n=3, random_state=42),
            mols[mols.loc[:, 'tanimoto'] == 0.4].sample(n=3, random_state=42),
            mols[mols.loc[:, 'tanimoto'] == 0.3].sample(n=2, random_state=42),
        ])
        smicols = ['test_smiles', 'train_smiles']
        WriteDataFrameSmilesToXls(mol_df,
                                  smicols,
                                  save_path,
                                  retain_smiles_col=True)


def plot_tanimoto_error_RF(data_dir_tanimoto, cv=5):
    '''Plot the Error on the y-axis against Tanimoto on the x-axis.
    '''

    DATA_PATH_RF = '2d3d/output/240425_RF_use_all_features'

    y_names = ['dipoleMoment', 'homo', 'gap', 'lumo', 'energy', 'enthalpy']

    aggs = [
        '2d', 'no_agg', 'correct', 'mean', 'Boltzman weight', 'global minimum',
        'random', 'rmsd_max', 'rmsd_min'
    ]

    #aggs = ['2d']
    cv = 1

    for agg in aggs:
        for i in range(cv):
            plt.rcParams.update({'font.size': 18})
            ### plot1: scatter plot tanimoto vs error ###
            # prepare fig, axs
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            for j, y_name in enumerate(y_names):
                data_path_pred = os.path.join(DATA_PATH_RF, y_name, agg,
                                              f'fold_{i}',
                                              'test_set_predictions.csv')
                data_path_tanimoto = f'{data_dir_tanimoto}/fold_{i}/max_tanimoto_pairs.tsv'

                cid_tanimoto = pd.read_csv(data_path_tanimoto,
                                           sep='\t',
                                           usecols=['test_cid', 'tanimoto'])
                # sort by test_cid and reset index
                cid_tanimoto.sort_values('test_cid', inplace=True)
                cid_tanimoto.reset_index(drop=True, inplace=True)

                # Load the test set predictions
                pred_test = pd.read_csv(data_path_pred)

                merged_df = pd.merge(
                    pred_test,
                    cid_tanimoto,
                    left_on='cid',
                    right_on='test_cid').drop(columns='test_cid')
                # Calculate the absolute difference between 'predicted' and 'actual' and add it as a new column 'error'
                merged_df['error'] = abs(merged_df['predicted'] -
                                         merged_df['actual'])

                # for position of subplot
                ax = axs[j // 3, j % 3]
                sns.scatterplot(x='tanimoto', y='error', data=merged_df, ax=ax)
                ax.set_xlabel('tanimoto')
                ax.set_ylabel('absolute error')
                ax.set_title(y_name)
            fig.suptitle(f'Model=RF, Aggregation={agg}, fold_{i}')
            plt.tight_layout(pad=0.5)
            #save fig
            save_dir = f'{data_dir_tanimoto}/fold_{i}/plot_error/RF'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f'{save_dir}/error_{agg}.png')
            plt.close(fig)


def plot_tanimoto_error_MIL(data_dir_tanimoto, cv=5):
    '''Plot Tanimoto on the x-axis and Error on the y-axis.
    '''

    DATA_PATH_MIL = '3D-MIL-QSSR/output/240508'
    descs_sets = ['agg_mean']  #['moe', 'pmapper', 'agg_mean']
    algorithms = [
        'BagWrapperMLPRegressor', 'InstanceWrapperMLPRegressor',
        'BagNetRegressor', 'InstanceNetRegressor', 'AttentionNetRegressor'
    ]

    y_names = ['dipoleMoment', 'homo', 'gap', 'lumo', 'energy', 'enthalpy']

    cv = 1

    for descs_set in descs_sets:
        for algorithm in algorithms:
            for i in range(cv):
                plt.rcParams.update({'font.size': 18})
                ### plot1: scatter plot tanimoto vs error ###
                # prepare fig, axs
                fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                for j, y_name in enumerate(y_names):
                    data_path_pred = os.path.join(DATA_PATH_MIL, descs_set,
                                                  algorithm, y_name,
                                                  f'fold_{i}',
                                                  'test_set_predictions.csv')

                    data_path_tanimoto = f'{data_dir_tanimoto}/fold_{i}/max_tanimoto_pairs.tsv'

                    cid_tanimoto = pd.read_csv(
                        data_path_tanimoto,
                        sep='\t',
                        usecols=['test_cid', 'tanimoto'])
                    # sort by test_cid and reset index
                    cid_tanimoto.sort_values('test_cid', inplace=True)
                    cid_tanimoto.reset_index(drop=True, inplace=True)

                    # Load the test set predictions
                    pred_test = pd.read_csv(data_path_pred)

                    merged_df = pd.merge(
                        pred_test,
                        cid_tanimoto,
                        left_on='cid',
                        right_on='test_cid').drop(columns='test_cid')
                    # Calculate the absolute difference between 'predicted' and 'actual' and add it as a new column 'error'
                    merged_df['error'] = abs(merged_df['predicted'] -
                                             merged_df['actual'])

                    # for position of subplot
                    ax = axs[j // 3, j % 3]
                    sns.scatterplot(x='tanimoto',
                                    y='error',
                                    data=merged_df,
                                    ax=ax)
                    ax.set_xlabel('tanimoto')
                    ax.set_ylabel('absolute error')
                    ax.set_title(y_name)
                fig.suptitle(
                    f'Model=MIL, Descs_Set={descs_set}, Algorithm={algorithm}, fold_{i}'
                )
                plt.tight_layout(pad=0.5)
                #save fig
                save_dir = f'{data_dir_tanimoto}/fold_{i}/plot_error/MIL/{descs_set}'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(f'{save_dir}/error_{algorithm}.png')
                plt.close(fig)


def plot_mae(data_dir_tanimoto, cv=5, which_plot='moe'):
    '''Plot MAE on the y-axis against Tanimoto Thresholding on the x-axis.
    '''

    y_names = ['dipoleMoment', 'homo', 'gap', 'lumo', 'energy', 'enthalpy']
    # y_names = ['enthalpy']

    # for RF
    DATA_PATH_RF = '2d3d/output/240425_RF_use_all_features'
    aggs = [
        '2d', 'no_agg', 'correct', 'mean', 'Boltzman weight', 'global minimum',
        'random', 'rmsd_max', 'rmsd_min'
    ]
    #aggs = ['2d', 'correct']

    # for MIL
    DATA_PATH_MIL = '3D-MIL-QSSR/output/240508'
    descs_sets = ['moe', 'agg_mean', 'pmapper']
    algorithms = [
        'BagWrapperMLPRegressor', 'InstanceWrapperMLPRegressor',
        'BagNetRegressor', 'InstanceNetRegressor', 'AttentionNetRegressor'
    ]

    cv = 1

    for n_fold in range(cv):
        #plt.rcParams.update({'font.size': 18})
        ### plot2: bar plot tanimoto_thresholdings vs MAE ###
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        for n_y_name, y_name in enumerate(y_names):
            df_list = []
            # for prepare plot result of RF in this roop
            for agg in aggs:
                data_path_pred = os.path.join(DATA_PATH_RF, y_name, agg,
                                              f'fold_{n_fold}',
                                              'test_set_predictions.csv')
                data_path_tanimoto = f'{data_dir_tanimoto}/fold_{n_fold}/max_tanimoto_pairs.tsv'

                cid_tanimoto = pd.read_csv(data_path_tanimoto,
                                           sep='\t',
                                           usecols=['test_cid', 'tanimoto'])
                # sort by test_cid and reset index
                cid_tanimoto.sort_values('test_cid', inplace=True)
                cid_tanimoto.reset_index(drop=True, inplace=True)

                # Load the test set predictions
                pred_test = pd.read_csv(data_path_pred)

                merged_df = pd.merge(
                    pred_test,
                    cid_tanimoto,
                    left_on='cid',
                    right_on='test_cid').drop(columns='test_cid')

                mae_list = []
                for k in range(100):
                    tanimoto_thres = (k + 1) / 100
                    # extract under tanimoto_thresholdings
                    cut_df = merged_df[merged_df.loc[:, 'tanimoto'] <=
                                       tanimoto_thres]

                    # Calculate the metrics
                    if cut_df.empty:
                        mae = 0
                    else:
                        _, _, mae = calcMetrics(cut_df['actual'],
                                                cut_df['predicted'])
                    mae_list.append(mae)

                # Create a DataFrame for this agg and append it to the list
                df = pd.DataFrame({
                    agg: mae_list  # Set column name to agg
                })
                df_list.append(df)

            # for prepare plot result of MIL in this roop
            for descs_set in descs_sets:
                for algorithm in algorithms:
                    data_path_pred = os.path.join(DATA_PATH_MIL, descs_set,
                                                  algorithm, y_name,
                                                  f'fold_{n_fold}',
                                                  'test_set_predictions.csv')
                    data_path_tanimoto = f'{data_dir_tanimoto}/fold_{n_fold}/max_tanimoto_pairs.tsv'

                    cid_tanimoto = pd.read_csv(
                        data_path_tanimoto,
                        sep='\t',
                        usecols=['test_cid', 'tanimoto'])
                    # sort by test_cid and reset index
                    cid_tanimoto.sort_values('test_cid', inplace=True)
                    cid_tanimoto.reset_index(drop=True, inplace=True)

                    # Load the test set predictions
                    pred_test = pd.read_csv(data_path_pred)

                    merged_df = pd.merge(
                        pred_test,
                        cid_tanimoto,
                        left_on='cid',
                        right_on='test_cid').drop(columns='test_cid')

                    mae_list = []
                    for k in range(100):
                        tanimoto_thres = (k + 1) / 100
                        # extract under tanimoto_thresholdings
                        cut_df = merged_df[merged_df.loc[:, 'tanimoto'] <=
                                           tanimoto_thres]

                        # Calculate the metrics
                        if cut_df.empty:
                            mae = 0
                        else:
                            _, _, mae = calcMetrics(cut_df['actual'],
                                                    cut_df['predicted'])
                        mae_list.append(mae)

                    # Create a DataFrame for this agg and append it to the list
                    df = pd.DataFrame({
                        f'{descs_set}_{algorithm}':
                        mae_list  # Set column name to agg
                    })
                    df_list.append(df)

            # Concatenate all the DataFrames in the list
            mae_df = pd.concat(df_list, axis=1)
            # for position of subplot
            ax = axs[n_y_name // 3, n_y_name % 3]
            # Create color maps
            cmap_red = cm.get_cmap('autumn')
            cmap_blue = cm.get_cmap('winter')

            if which_plot == 'moe':
                ### for plot RF and MIL-MOE ###
                # plot RF result
                for i, agg in enumerate(aggs):
                    color = cmap_red(i / len(aggs))
                    ax.plot(np.linspace(0, 1, 100),
                            mae_df[agg],
                            label=f'RF_{agg}',
                            color=color)
                # plot MIL result
                for descs_set in ['moe']:
                    for i, algorithm in enumerate(algorithms):
                        color = cmap_blue(i / len(algorithms))
                        ax.plot(np.linspace(0, 1, 100),
                                mae_df[f'{descs_set}_{algorithm}'],
                                label=f'MIL_{descs_set}_{algorithm}',
                                color=color)
                ####################################
            elif which_plot == 'mean':
                ### for plot RF-mean and MIL-MOEmean ###
                # plot RF result
                for i, agg in enumerate(['mean']):
                    ax.plot(np.linspace(0, 1, 100),
                            mae_df[agg],
                            label=f'RF_{agg}')
                # plot MIL result
                for descs_set in ['agg_mean']:
                    for i, algorithm in enumerate(algorithms):
                        ax.plot(np.linspace(0, 1, 100),
                                mae_df[f'{descs_set}_{algorithm}'],
                                label=f'MIL_{descs_set}_{algorithm}')
                ####################################

            elif which_plot == 'pmapper':
                # plot MIL result
                for descs_set in ['pmapper']:
                    for i, algorithm in enumerate(algorithms):
                        ax.plot(np.linspace(0, 1, 100),
                                mae_df[f'{descs_set}_{algorithm}'],
                                label=f'MIL_{descs_set}_{algorithm}')

            ax.set_xlabel('tanimoto threshold')
            ax.set_ylabel('MAE')
            ax.set_title(y_name)
            ax.legend(fontsize=6)
            #ax.legend(fontsize=4, ncol=2)

        fig.suptitle(f'fold_{n_fold}')
        plt.tight_layout()  #pad=0.5
        #save fig
        plt.savefig(
            f'{data_dir_tanimoto}/fold_{n_fold}/MAE_tanimoto_thres_{which_plot}.pdf'
        )
        plt.close(fig)


def plot_mae_with_error_bar(data_dir_tanimoto, cv=5, which_plot='moe'):
    '''
    Plot MAE on the y-axis against Tanimoto Thresholding on the x-axis.
    Add error bars.
    Only the results of Uni-Mol are reflected in this function.
    '''

    y_names = ['dipoleMoment', 'homo', 'gap', 'lumo', 'energy', 'enthalpy']

    # for RF
    DATA_PATH_RF = '2d3d/output/240425_RF_use_all_features'
    aggs_RF = [
        '2d', 'no_agg', 'correct', 'mean', 'Boltzman weight', 'global minimum',
        'random', 'rmsd_max', 'rmsd_min'
    ]
    aggs_RF = ['2d', 'mean']

    # for MIL
    DATA_PATH_MIL = '3D-MIL-QSSR/output/240508'
    descs_sets = ['moe', 'agg_mean', 'pmapper']
    algorithms = [
        'BagWrapperMLPRegressor', 'InstanceWrapperMLPRegressor',
        'BagNetRegressor', 'InstanceNetRegressor', 'AttentionNetRegressor'
    ]
    algorithms = ['InstanceWrapperMLPRegressor']

    DATA_PATH_Uni = 'Uni-Mol/unimol_tools/240515_exp'
    aggs_Uni = ['correct', 'global minimum', 'rmsd_max']
    aggs_Uni = ['correct']

    #plt.rcParams.update({'font.size': 18})
    ### plot2: bar plot tanimoto_thresholdings vs MAE ###
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    for n_y_name, y_name in tqdm(enumerate(y_names),
                                 desc='y_name',
                                 leave=False):
        df_list = []
        # for prepare plot result of RF in this roop
        for agg in tqdm(aggs_RF, desc='RF_agg', leave=False):
            mae_list_all_folds = []
            for n_fold in tqdm(range(cv), desc='RF_fold', leave=False):
                data_path_pred = os.path.join(DATA_PATH_RF, y_name, agg,
                                              f'fold_{n_fold}',
                                              'test_set_predictions.csv')
                data_path_tanimoto = f'{data_dir_tanimoto}/fold_{n_fold}/max_tanimoto_pairs.tsv'

                cid_tanimoto = pd.read_csv(data_path_tanimoto,
                                           sep='\t',
                                           usecols=['test_cid', 'tanimoto'])
                # sort by test_cid and reset index
                cid_tanimoto.sort_values('test_cid', inplace=True)
                cid_tanimoto.reset_index(drop=True, inplace=True)

                # Load the test set predictions
                pred_test = pd.read_csv(data_path_pred)

                merged_df = pd.merge(
                    pred_test,
                    cid_tanimoto,
                    left_on='cid',
                    right_on='test_cid').drop(columns='test_cid')

                mae_list = []
                for k in range(100):
                    tanimoto_thres = (k + 1) / 100
                    # extract under tanimoto_thresholdings
                    cut_df = merged_df[merged_df.loc[:, 'tanimoto'] <=
                                       tanimoto_thres]
                    # Calculate the metrics
                    if cut_df.empty:
                        mae = 0
                    else:
                        mae = mean_absolute_error(cut_df['actual'],
                                                  cut_df['predicted'])
                    mae_list.append(mae)
                mae_list_all_folds.append(mae_list)

            # Create a DataFrame for this agg and append its mean and std to the list
            df = pd.DataFrame({
                f'{agg}_mean':
                np.array(mae_list_all_folds).mean(axis=0),
                f'{agg}_std':
                np.array(mae_list_all_folds).std(axis=0)
            })
            df_list.append(df)

        # for prepare plot result of MIL in this loop
        for descs_set in tqdm(descs_sets, desc='MIL_descs_set', leave=False):
            for algorithm in tqdm(algorithms,
                                  desc='MIL_algorithm',
                                  leave=False):
                mae_list_all_folds = []
                for n_fold in range(cv):
                    data_path_pred = os.path.join(DATA_PATH_MIL, descs_set,
                                                  algorithm, y_name,
                                                  f'fold_{n_fold}',
                                                  'test_set_predictions.csv')
                    data_path_tanimoto = f'{data_dir_tanimoto}/fold_{n_fold}/max_tanimoto_pairs.tsv'

                    cid_tanimoto = pd.read_csv(
                        data_path_tanimoto,
                        sep='\t',
                        usecols=['test_cid', 'tanimoto'])
                    # sort by test_cid and reset index
                    cid_tanimoto.sort_values('test_cid', inplace=True)
                    cid_tanimoto.reset_index(drop=True, inplace=True)

                    # Load the test set predictions
                    pred_test = pd.read_csv(data_path_pred)

                    merged_df = pd.merge(
                        pred_test,
                        cid_tanimoto,
                        left_on='cid',
                        right_on='test_cid').drop(columns='test_cid')

                    mae_list = []
                    for k in range(100):
                        tanimoto_thres = (k + 1) / 100
                        # extract under tanimoto_thresholdings
                        cut_df = merged_df[merged_df.loc[:, 'tanimoto'] <=
                                           tanimoto_thres]

                        # Calculate the metrics
                        if cut_df.empty:
                            mae = 0
                        else:
                            mae = mean_absolute_error(cut_df['actual'],
                                                      cut_df['predicted'])
                        mae_list.append(mae)
                    mae_list_all_folds.append(mae_list)

                # Create a DataFrame for this agg and append it to the list
                df = pd.DataFrame({
                    f'{descs_set}_{algorithm}_mean':
                    np.array(mae_list_all_folds).mean(axis=0),
                    f'{descs_set}_{algorithm}_std':
                    np.array(mae_list_all_folds).std(axis=0)
                })
                df_list.append(df)

        # for prepare plot result of Uni-Mol in this loop
        for agg in tqdm(aggs_Uni, desc='Uni_agg', leave=False):
            mae_list_all_folds = []
            for n_fold in tqdm(range(cv), desc='Uni_fold', leave=False):
                data_path_pred = os.path.join(DATA_PATH_Uni, y_name, agg,
                                              f'fold_{n_fold}',
                                              'test_set_predictions.csv')
                data_path_tanimoto = f'{data_dir_tanimoto}/fold_{n_fold}/max_tanimoto_pairs.tsv'

                cid_tanimoto = pd.read_csv(data_path_tanimoto,
                                           sep='\t',
                                           usecols=['test_cid', 'tanimoto'])
                # sort by test_cid and reset index
                cid_tanimoto.sort_values('test_cid', inplace=True)
                cid_tanimoto.reset_index(drop=True, inplace=True)

                # Load the test set predictions
                pred_test = pd.read_csv(data_path_pred)

                merged_df = pd.merge(
                    pred_test,
                    cid_tanimoto,
                    left_on='cid',
                    right_on='test_cid').drop(columns='test_cid')

                mae_list = []
                for k in range(100):
                    tanimoto_thres = (k + 1) / 100
                    # extract under tanimoto_thresholdings
                    cut_df = merged_df[merged_df.loc[:, 'tanimoto'] <=
                                       tanimoto_thres]

                    # Calculate the metrics
                    if cut_df.empty:
                        mae = 0
                    else:
                        mae = mean_absolute_error(cut_df['actual'],
                                                  cut_df['predicted'])
                    mae_list.append(mae)
                mae_list_all_folds.append(mae_list)

            # Create a DataFrame for this agg and append its mean and std to the list
            df = pd.DataFrame({
                f'uni_{agg}_mean':
                np.array(mae_list_all_folds).mean(axis=0),
                f'uni_{agg}_std':
                np.array(mae_list_all_folds).std(axis=0)
            })
            df_list.append(df)

        # Concatenate all the DataFrames in the list
        mae_df = pd.concat(df_list, axis=1)
        # for position of subplot
        ax = axs[n_y_name % 3, n_y_name // 3]
        # Create color maps
        cmap_red = cm.get_cmap('autumn')
        cmap_blue = cm.get_cmap('winter')

        if which_plot == 'moe':
            ### for plot RF and MIL-MOE ###
            # plot RF result
            for i, agg in enumerate(aggs_RF):
                color = cmap_red(i / len(aggs_RF))
                ax.errorbar(np.linspace(0, 1, 100),
                            mae_df[f'{agg}_mean'],
                            yerr=mae_df[f'{agg}_std'],
                            label=f'RF_{agg}',
                            color=color)
            # plot MIL result
            for descs_set in ['moe']:
                for i, algorithm in enumerate(algorithms):
                    color = cmap_blue(i / len(algorithms))
                    ax.errorbar(np.linspace(0, 1, 100),
                                mae_df[f'{descs_set}_{algorithm}_mean'],
                                yerr=mae_df[f'{descs_set}_{algorithm}_std'],
                                label=f'MIL_{descs_set}_{algorithm}',
                                color=color)
            ####################################

        elif which_plot == 'mean':
            ### for plot RF-mean and MIL-MOEmean ###
            # plot RF result
            for i, agg in enumerate(['mean']):
                ax.plot(np.linspace(0, 1, 100), mae_df[agg], label=f'RF_{agg}')
            # plot MIL result
            for descs_set in ['agg_mean']:
                for i, algorithm in enumerate(algorithms):
                    ax.errorbar(np.linspace(0, 1, 100),
                                mae_df[f'{descs_set}_{algorithm}_mean'],
                                yerr=mae_df[f'{descs_set}_{algorithm}_std'],
                                label=f'MIL_{descs_set}_{algorithm}')
            ####################################

        elif which_plot == 'pmapper':
            # plot MIL result
            for descs_set in ['pmapper']:
                for i, algorithm in enumerate(algorithms):
                    ax.errorbar(np.linspace(0, 1, 100),
                                mae_df[f'{descs_set}_{algorithm}_mean'],
                                yerr=mae_df[f'{descs_set}_{algorithm}_std'],
                                label=f'MIL_{descs_set}_{algorithm}')

        elif which_plot == 'best':
            # ax.errorbar(
            #     np.linspace(0, 1, 100),
            #     mae_df['mean_mean'],
            #     yerr=mae_df[f'mean_std'],
            #     label=f'MOE_RF_Mean',
            # )
            # ax.errorbar(
            #     np.linspace(0, 1, 100),
            #     mae_df['2d_mean'],
            #     yerr=mae_df[f'2d_std'],
            #     label=f'ECFP4_RF',
            # )
            # ax.errorbar(
            #     np.linspace(0, 1, 100),
            #     mae_df['moe_InstanceWrapperMLPRegressor_mean'],
            #     yerr=mae_df[f'moe_InstanceWrapperMLPRegressor_std'],
            #     label=f'MOE_Instance-Wrapper',
            # )
            # ax.errorbar(
            #     np.linspace(0, 1, 100),
            #     mae_df['agg_mean_InstanceWrapperMLPRegressor_mean'],
            #     yerr=mae_df[f'agg_mean_InstanceWrapperMLPRegressor_std'],
            #     label=f'MOE-Mean_Instance-Wrapper',
            # )
            # # ax.errorbar(
            # #     np.linspace(0, 1, 100),
            # #     mae_df['pmapper_InstanceWrapperMLPRegressor_mean'],
            # #     yer=mae_df[f'pmapper_InstanceWrapperMLPRegressor_std'],
            # #     label=f'pmapper_Instance-Wrapper',
            # # )
            # ax.errorbar(
            #     np.linspace(0, 1, 100),
            #     mae_df['uni_correct_mean'],
            #     yerr=mae_df[f'uni_correct_std'],
            #     label=f'Uni-Mol_Correct',
            # )

            x = np.linspace(0, 1, 100)
            # Remove cases where there are no compounds at all. Remove only if there are no compounds. The range is different because only Enthalpy has missing values.
            if y_name == 'enthalpy':
                mae_df = mae_df.iloc[19:, :]
                x = x[19:]
            else:
                mae_df = mae_df.iloc[16:, :]
                x = x[16:]

            # MOE_RF_Mean
            ax.plot(x,
                    mae_df['mean_mean'],
                    label='RF using Mean aggregated MOE descriptors')
            ax.fill_between(x,
                            mae_df['mean_mean'] - mae_df['mean_std'],
                            mae_df['mean_mean'] + mae_df['mean_std'],
                            alpha=0.3)

            # ECFP4_RF
            ax.plot(x, mae_df['2d_mean'], label='RF using ECFP4 count')
            ax.fill_between(x,
                            mae_df['2d_mean'] - mae_df['2d_std'],
                            mae_df['2d_mean'] + mae_df['2d_std'],
                            alpha=0.3)

            # MOE_Instance-Wrapper
            ax.plot(
                x,
                mae_df['moe_InstanceWrapperMLPRegressor_mean'],
                label='MIL Non-aggregation algorithm using MOE descriptors')
            ax.fill_between(x,
                            mae_df['moe_InstanceWrapperMLPRegressor_mean'] -
                            mae_df['moe_InstanceWrapperMLPRegressor_std'],
                            mae_df['moe_InstanceWrapperMLPRegressor_mean'] +
                            mae_df['moe_InstanceWrapperMLPRegressor_std'],
                            alpha=0.3)

            # MOE-Mean_Instance-Wrapper
            ax.plot(
                x,
                mae_df['agg_mean_InstanceWrapperMLPRegressor_mean'],
                label=
                'MIL Non-aggregation algorithm using Mean aggregated MOE descriptors'
            )
            ax.fill_between(
                x,
                mae_df['agg_mean_InstanceWrapperMLPRegressor_mean'] -
                mae_df['agg_mean_InstanceWrapperMLPRegressor_std'],
                mae_df['agg_mean_InstanceWrapperMLPRegressor_mean'] +
                mae_df['agg_mean_InstanceWrapperMLPRegressor_std'],
                alpha=0.3)

            # Uni-Mol_Correct
            ax.plot(x,
                    mae_df['uni_correct_mean'],
                    label='Uni-Mol using Ground-truth conformation')
            ax.fill_between(
                x,
                mae_df['uni_correct_mean'] - mae_df['uni_correct_std'],
                mae_df['uni_correct_mean'] + mae_df['uni_correct_std'],
                alpha=0.3)

        if y_name == 'dipoleMoment':
            title = 'Dipole moment'
        elif y_name == 'homo':
            title = 'HOMO'
        elif y_name == 'gap':
            title = 'HOMO-LUMO gap'
        elif y_name == 'lumo':
            title = 'LUMO'
        elif y_name == 'energy':
            title = 'Energy'
        elif y_name == 'enthalpy':
            title = 'Enthalpy'
        ax.set_title(title, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        #ax.legend(fontsize=4, ncol=2)

    #fig.suptitle(f'fold_{n_fold}')
    axs[0, 0].set_xlabel('Tanimoto Similarity Threshold', fontsize=18)
    axs[0, 0].set_ylabel('Mean Absolute Error', fontsize=18)
    # Separate the legend into a separate figure because it is too large.
    handles, labels = axs[0, 0].get_legend_handles_labels()
    plt.tight_layout()  #pad=0.5
    #save fig
    plt.savefig(f'{data_dir_tanimoto}/MAE_tanimoto_thres_{which_plot}.pdf')
    plt.close(fig)

    # Create a figure for the legend only
    fig_legend = plt.figure(figsize=(3, 1))
    ax_legend = fig_legend.add_subplot(111)
    # Hide the axes
    ax_legend.axis('off')
    # Add the legend
    ax_legend.legend(handles, labels, loc='center', fontsize=16)
    plt.tight_layout()
    # Save the figure
    plt.savefig(
        f'{data_dir_tanimoto}/MAE_tanimoto_thres_{which_plot}_legend.pdf',
        bbox_inches='tight')
    plt.close(fig)


def calc_metrics_without_similar(data_dir_tanimoto,
                                 data_dir_pred,
                                 file_name_prefix,
                                 cv=5,
                                 tanimoto_thres=0.99):
    '''
    Load max_tanimoto_pairs.tsv for each fold and extract test_cid with Tanimoto Similarity below a certain threshold.
    Load test_set_predictions.csv for each property, aggregation method, and fold, and extract the rows corresponding to the extracted test_cid.
    Calculate the metrics using calcMetrics for each fold and save them in df_result.
    Finally, calculate the mean and standard deviation of the metrics and save them in df_mean_std.
    '''
    df_result = pd.DataFrame(columns=['fold', 'r2', 'mse', 'mae'])
    # Load the test set predictions for each fold
    for i in range(cv):
        data_path_tanimoto = f'{data_dir_tanimoto}/fold_{i}/max_tanimoto_pairs.tsv'
        data_path_pred = f'{data_dir_pred}/fold_{i}/test_set_predictions.csv'
        mols = pd.read_csv(data_path_tanimoto, sep='\t')
        # sort by test_cid and reset index
        mols.sort_values('test_cid', inplace=True)
        mols.reset_index(drop=True, inplace=True)

        # extract under tanimoto_thresholdings
        mols = mols[mols.loc[:, 'tanimoto'] <= tanimoto_thres]
        # Load the test set predictions
        pred_test = pd.read_csv(data_path_pred)

        # Extract the rows corresponding to the test_cid extracted above
        pred_test = pred_test[pred_test['cid'].isin(mols['test_cid'])]

        # Calculate the metrics
        r2, mse, mae = calcMetrics(pred_test['actual'], pred_test['predicted'])
        # Create a new DataFrame with these values
        new_row = pd.DataFrame(
            [[f'{file_name_prefix}_fold_{i}', r2, mse, mae]],
            columns=df_result.columns)
        # Concatenate the new row to the original DataFrame
        df_result = pd.concat([df_result, new_row], ignore_index=True)

    # Calculate the mean and std of r2, mse, mae
    mean_r2 = df_result['r2'].mean()
    std_r2 = df_result['r2'].std()
    mean_mse = df_result['mse'].mean()
    std_mse = df_result['mse'].std()
    mean_mae = df_result['mae'].mean()
    std_mae = df_result['mae'].std()
    # Create new rows with these values
    mean_row = pd.DataFrame(
        [[f'{file_name_prefix}_Mean', mean_r2, mean_mse, mean_mae]],
        columns=df_result.columns)
    std_row = pd.DataFrame(
        [[f'{file_name_prefix}_Std', std_r2, std_mse, std_mae]],
        columns=df_result.columns)
    # Concatenate the new rows
    df_mean_std = pd.concat([mean_row, std_row], ignore_index=True)

    return df_result, df_mean_std


def calc_matrics_wrapper(data_dir_tanimoto, tanimoto_thres=0.99):
    '''Calculate metrics by loading results from RF, MIL, and UniMol, excluding compounds with Tanimoto Similarity below a certain threshold.
    '''
    SAVE_DIR = f'{data_dir_tanimoto}/metrics'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    # properties
    y_names = ['dipoleMoment', 'homo', 'gap', 'lumo', 'energy', 'enthalpy']

    # for RF
    DATA_PATH_RF = '2d3d/output/240425_RF_use_all_features'
    aggs = [
        '2d', 'no_agg', 'correct', 'mean', 'Boltzman weight', 'global minimum',
        'random', 'rmsd_max', 'rmsd_min'
    ]
    df_result_all_RF = pd.DataFrame()
    df_mean_std_all_RF = pd.DataFrame()
    for y_name in y_names:
        for agg in aggs:
            data_dir_pred = os.path.join(DATA_PATH_RF, y_name, agg)
            file_name_prefix = f'{y_name}_{agg}'
            df_result, df_mean_std = calc_metrics_without_similar(
                data_dir_tanimoto,
                data_dir_pred,
                file_name_prefix,
                tanimoto_thres=tanimoto_thres)
            # Append df to df_all
            df_result_all_RF = pd.concat([df_result_all_RF, df_result],
                                         ignore_index=True)
            df_mean_std_all_RF = pd.concat([df_mean_std_all_RF, df_mean_std],
                                           ignore_index=True)
    # save df as csv
    df_result_all_RF.to_csv(f'{SAVE_DIR}/RF_each_fold_{tanimoto_thres}.csv',
                            index=False)
    df_mean_std_all_RF.to_csv(f'{SAVE_DIR}/RF_mean_std_{tanimoto_thres}.csv',
                              index=False)

    # for MIL
    DATA_PATH_MIL = '3D-MIL-QSSR/output/240508'
    descs_sets = ['moe', 'pmapper', 'agg_mean']
    algorithms = [
        'BagWrapperMLPRegressor', 'InstanceWrapperMLPRegressor',
        'BagNetRegressor', 'InstanceNetRegressor', 'AttentionNetRegressor'
    ]
    df_result_all_MIL = pd.DataFrame()
    df_mean_std_all_MIL = pd.DataFrame()
    for descs_set in descs_sets:
        for algorithm in algorithms:
            for y_name in y_names:
                data_dir_pred = os.path.join(DATA_PATH_MIL, descs_set,
                                             algorithm, y_name)
                file_name_prefix = f'{descs_set}_{algorithm}_{y_name}'
                df_result, df_mean_std = calc_metrics_without_similar(
                    data_dir_tanimoto,
                    data_dir_pred,
                    file_name_prefix,
                    tanimoto_thres=tanimoto_thres)
                # Append df to df_all
                df_result_all_MIL = pd.concat([df_result_all_MIL, df_result],
                                              ignore_index=True)
                df_mean_std_all_MIL = pd.concat(
                    [df_mean_std_all_MIL, df_mean_std], ignore_index=True)
    # save df as csv
    df_result_all_MIL.to_csv(f'{SAVE_DIR}/MIL_each_fold_{tanimoto_thres}.csv',
                             index=False)
    df_mean_std_all_MIL.to_csv(f'{SAVE_DIR}/MIL_mean_std_{tanimoto_thres}.csv',
                               index=False)

    # for unimol
    DATA_PATH_UNIMOL = 'Uni-Mol/unimol_tools/240515_exp'
    aggs = ['correct', 'global minimum', 'rmsd_max']
    df_result_all_uni = pd.DataFrame()
    df_mean_std_all_uni = pd.DataFrame()
    for y_name in y_names:
        for agg in aggs:
            data_dir_pred = os.path.join(DATA_PATH_UNIMOL, y_name, agg)
            file_name_prefix = f'{y_name}_{agg}'
            df_result, df_mean_std = calc_metrics_without_similar(
                data_dir_tanimoto,
                data_dir_pred,
                file_name_prefix,
                tanimoto_thres=tanimoto_thres)
            # Append df to df_all
            df_result_all_uni = pd.concat([df_result_all_uni, df_result],
                                          ignore_index=True)
            df_mean_std_all_uni = pd.concat([df_mean_std_all_uni, df_mean_std],
                                            ignore_index=True)
    df_result_all_uni.to_csv(f'{SAVE_DIR}/Uni_each_fold_{tanimoto_thres}.csv',
                             index=False)
    df_mean_std_all_uni.to_csv(f'{SAVE_DIR}/Uni_mean_std_{tanimoto_thres}.csv',
                               index=False)


if __name__ == '__main__':
    TODAY = '240425_tanimoto'
    DATA_PATH = '2d3d/dataset/PubChemQC-PM6/2D/step2/ecfp_97696.tsv'
    INDEX_DIR = '2d3d/dataset/PubChemQC-PM6/step15/index/dipoleMoment'
    SAVE_DIR = f'2d3d/output/{TODAY}'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    #calc_tanimoto(DATA_PATH, INDEX_DIR, SAVE_DIR, test_mode=False)
    # plot_all(SAVE_DIR, 1)
    # tsv2xlsx(SAVE_DIR)
    # tsv2xlsx4fig(SAVE_DIR)
    # calc_matrics_wrapper(SAVE_DIR, tanimoto_thres=1.0)
    # calc_matrics_wrapper(SAVE_DIR, tanimoto_thres=0.9)
    #plot_tanimoto_error_RF(SAVE_DIR)
    #plot_tanimoto_error_MIL(SAVE_DIR)
    # plot_mae(SAVE_DIR, which_plot='pmapper')
    plot_mae_with_error_bar(SAVE_DIR, which_plot='best')
