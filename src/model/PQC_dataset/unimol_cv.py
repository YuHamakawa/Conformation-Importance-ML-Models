import ast
import sys

import numpy as np
import pandas as pd
from unimol_tools import MolPredict, MolTrain

if __name__ == '__main__':
    #extract_coord() # execute step16_extract_cords.py

    today = '240515'
    y_names = ['dipoleMoment', 'homo', 'gap', 'lumo', 'energy', 'enthalpy']
    # Modify y_names based on the value of sys.argv[1]
    match int(sys.argv[1]):
        case 1:
            y_names = ['dipoleMoment']
        case 2:
            y_names = ['homo']
        case 3:
            y_names = ['gap']
        case 4:
            y_names = ['lumo']
        case 5:
            y_names = ['energy']
        case 6:
            y_names = ['enthalpy']
        case 12:
            y_names = ['dipoleMoment', 'homo']
        case 34:
            y_names = ['gap', 'lumo']
        case 56:
            y_names = ['energy', 'enthalpy']

    aggs = ['correct', 'global minimum', 'random', 'rmsd_max', 'rmsd_min']
    match int(sys.argv[2]):
        case 1:
            aggs = ['correct']
            DATA_PATH = '2d3d/dataset/PubChemQC-PM6/step16/correct_coords_props_97696.csv'
            merged_df = pd.read_csv(DATA_PATH)
        case 2:
            aggs = ['global minimum']
            DATA_PATH = '2d3d/dataset/PubChemQC-PM6/step16/1conf_97696.tsv'
            merged_df = pd.read_csv(DATA_PATH, sep='\t')
        case 3:
            aggs = ['rmsd_max']
            DATA_PATH = '2d3d/dataset/PubChemQC-PM6/step16/rmsd_max_97696.tsv'
            merged_df = pd.read_csv(DATA_PATH, sep='\t')

    N_FOLD = 5
    merged_df['atoms'] = merged_df['atoms'].apply(ast.literal_eval)
    merged_df['coordinates'] = merged_df['coordinates'].apply(ast.literal_eval)
    for y_name in y_names:
        for agg in aggs:
            for fold in range(N_FOLD):
                index_dir = f'/home/yu_hamakawa/program/2d3d/dataset/PubChemQC-PM6/step15/index/{y_name}'
                train_cid = np.load(f'{index_dir}/train_cid_fold_{fold}.npy')
                test_cid = np.load(f'{index_dir}/test_cid_fold_{fold}.npy')
                train_data = merged_df[merged_df['cid'].isin(
                    train_cid
                )].loc[:, ['cid', 'atoms', 'coordinates', y_name]].rename(
                    columns={y_name: 'target'})
                test_data = merged_df[merged_df['cid'].isin(
                    test_cid
                )].loc[:, ['cid', 'atoms', 'coordinates', y_name]].rename(
                    columns={y_name: 'target'})

                # Convert the merged DataFrame to a dictionary
                #merged_dict = merged_df.to_dict(orient='list')
                train_dict = train_data.to_dict(orient='list')
                test_dict = test_data.to_dict(orient='list')

                # make DataFrame
                SAVE_PATH = f'Uni-Mol/unimol_tools/{today}_exp/{y_name}/{agg}/fold_{fold}'
                clf = MolTrain(
                    task='regression',
                    data_type='molecule',
                    epochs=100,
                    kfold=5,  # inner CV
                    batch_size=128,
                    metrics='mse',
                    remove_hs=True,
                    save_path=SAVE_PATH)
                clf.fit(data=train_dict)

                clf = MolPredict(load_model=SAVE_PATH)
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
                train_save_df.to_csv(f'{SAVE_PATH}/train_set_predictions.csv',
                                     index=False)
                test_save_df.to_csv(f'{SAVE_PATH}/test_set_predictions.csv',
                                    index=False)
