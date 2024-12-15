import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


def create_heatmap(csv_path,
                   save_dir,
                   p_column,
                   keyword,
                   name,
                   xy_font_size=10,
                   annot_size=6):
    """
    Create a heatmap from a CSV file containing p-values and save it as a PDF.

    Parameters:
    csv_path (str): Path to the CSV file containing the results.
    save_dir (str): Directory where the heatmap PDF and column CSV will be saved.
    p_column (str): The column name in the CSV file that contains the p-values.
    keyword (Union[str, list, dict, None]): Keyword(s) to filter and map model names.
        - If str, filters models containing the keyword.
        - If list, filters models containing any of the keywords in the list.
        - If dict, maps model names based on the dictionary.
        - If None, no filtering or mapping is applied.
    name (str): The name used for saving the heatmap PDF and column CSV.
    xy_font_size (int, optional): Font size for the x and y axis labels. Default is 10.
    annot_size (int, optional): Font size for the annotations in the heatmap. Default is 6.

    Returns:
    None
    """

    results_df = pd.read_csv(csv_path)
    # Get unique model names
    models = sorted(set(results_df["model_1"]).union(results_df["model_2"]))

    if keyword is not None:
        if isinstance(keyword, dict):
            model_mapping = {}
            for model in models:
                for k, v in keyword.items():
                    if k in model:
                        model_mapping[model] = v
                        break  # Map to the first matching keyword
            # Change model names in `results_df`
            results_df["model_1"] = results_df["model_1"].replace(
                model_mapping)
            results_df["model_2"] = results_df["model_2"].replace(
                model_mapping)
            # Filter to include only models specified in keyword
            models = list(keyword.values())
            results_df = results_df[results_df["model_1"].isin(models)
                                    & results_df["model_2"].isin(models)]
        elif isinstance(keyword, list):
            models = [
                model for model in models if any(k in model for k in keyword)
            ]
        else:
            models = [model for model in models if keyword in model]

    # Initialize p-value matrix for heatmap
    # p_value_matrix = pd.DataFrame(np.nan, index=models, columns=models)

    # Create pivot table
    p_value_matrix = results_df.pivot(
        index="model_1",
        columns="model_2",
        values="p_value"  # Use the corresponding column name
    )

    # Reindex rows and columns with updated model names
    p_value_matrix = p_value_matrix.reindex(index=models, columns=models)

    # Handle missing values appropriately
    p_value_matrix = p_value_matrix.where(pd.notnull(p_value_matrix), np.nan)

    # Embed p-values into the matrix
    for _, row in results_df.iterrows():
        model_1, model_2, p_value = row["model_1"], row["model_2"], row[
            p_column]
        if (model_1 in models) and (model_2 in models):
            p_value_matrix.loc[model_1, model_2] = p_value
            p_value_matrix.loc[model_2, model_1] = p_value

    # Generate a mask for the upper triangle
    # mask = np.triu(np.ones_like(p_value_matrix, dtype=bool))
    mask = p_value_matrix.isnull()

    # Plotting
    plt.figure(figsize=(10, 10))
    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)
    cmap = sns.color_palette("Spectral_r", as_cmap=True)
    ax = sns.heatmap(
        p_value_matrix,
        annot=True,  # Display values
        fmt=".3f",  # Display three decimal places
        annot_kws={"size": annot_size},
        cmap=cmap,  # "coolwarm",
        vmax=0.1,
        cbar_kws={
            #'label': 'p-value',
            "shrink": .6
        },
        square=True,
        mask=mask)
    plt.xticks(rotation=45, ha="right", fontsize=xy_font_size)
    plt.yticks(fontsize=xy_font_size)
    # Remove x and y axis ticks
    # plt.xticks([])
    # plt.yticks([])
    # Remove x and y axis titles
    plt.xlabel('')
    plt.ylabel('')
    # Get the colorbar
    cbar = ax.collections[0].colorbar
    # Change the font size of the colorbar label
    cbar.set_label('p-value', fontsize=18)
    # Change the font size of the colorbar tick labels (if needed)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    # Save the file
    plt.savefig(f'{save_dir}/heatmap_{name}.pdf')
    plt.close()

    save_dir = f'{save_dir}/column_csv'
    os.makedirs(save_dir, exist_ok=True)
    df_columns = pd.DataFrame(p_value_matrix.columns.to_list(),
                              columns=['Model Names'])
    df_columns.to_csv(f'{save_dir}/column_{name}.csv', index=False)


def anova(csv_path, keyword):
    """
    Perform ANOVA (Analysis of Variance) on a CSV file based on specified keywords.

    Parameters:
    csv_path (str): The file path to the CSV file containing the data.
    keyword (dict): A dictionary where keys are substrings to search for in the 'model_name' column 
                    and values are the new names to replace the matching substrings.

    Returns:
    None: Prints the F-statistic and p-value of the ANOVA test.

    The function performs the following steps:
    1. Reads the CSV file into a DataFrame.
    2. Filters rows where 'model_name' contains any of the keys in the keyword dictionary.
    3. Renames the 'model_name' column values based on the keyword dictionary.
    4. Extracts and flattens values for different model categories (Boltzmann weight, Mean, Global minimum, Random, Non-aggregation).
    5. Creates a new DataFrame with these extracted values.
    6. Performs ANOVA on the extracted values and prints the F-statistic and p-value.
    """
    df = pd.read_csv(csv_path)
    # Extract rows where 'model_name' contains any of the keys in the keyword dictionary
    df = df[df['model_name'].apply(
        lambda x: any(k in x for k in keyword.keys()))]
    # Rename 'model_name' based on the values in the keyword dictionary
    df['model_name'] = df['model_name'].apply(
        lambda x: next(v for k, v in keyword.items() if k in x))

    boltzmann_weight_values = df[df['model_name'].str.contains(
        'Boltzmann weight')].iloc[:, 1:].values.flatten()
    mean_values = df[df['model_name'].str.contains(
        'Mean')].iloc[:, 1:].values.flatten()
    global_minimum_values = df[df['model_name'].str.contains(
        'Global minimum')].iloc[:, 1:].values.flatten()
    random_values = df[df['model_name'].str.contains(
        'Random')].iloc[:, 1:].values.flatten()
    no_agg_values = df[df['model_name'].str.contains(
        'Non-aggregation')].iloc[:, 1:].values.flatten()
    df = pd.DataFrame({
        'boltzmann_weight': boltzmann_weight_values,
        'mean': mean_values,
        'global_minimum': global_minimum_values,
        'random': random_values,
        'no_agg': no_agg_values
    })

    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(df['boltzmann_weight'].dropna(),
                                     df['mean'].dropna(),
                                     df['global_minimum'].dropna(),
                                     df['random'].dropna(),
                                     df['no_agg'].dropna())
    print(f'f_statistic : {f_stat:.4}, p_value : {p_value:.4}')


if __name__ == '__main__':
    SAVE_DIR = 'xxx'
    os.makedirs(SAVE_DIR, exist_ok=True)

    DATA_DIR = 'xxx'
    ### PQC dataset ###
    y_names = ['dipoleMoment', 'homo', 'gap', 'lumo', 'energy', 'enthalpy']

    EACH_SAVE_DIR = f'{SAVE_DIR}/table_1'
    os.makedirs(EACH_SAVE_DIR, exist_ok=True)
    for y_name in tqdm(y_names, desc='y_name'):
        NAME = f'R2_Test_PQC_{y_name}'
        keyword = {
            'RF_moe_Boltzman weight': 'RF MOE Boltzmann weight',
            'RF_moe_mean': 'RF MOE Mean',
            'RF_moe_global minimum': 'RF MOE Global minimum',
            'RF_moe_rmsd_max': 'RF MOE RMSD max',
            'RF_moe_no_agg': 'RF MOE Non-aggregation',
            'RF_moe_correct': 'RF MOE Ground-truth',
            'RF_morse_Boltzman weight': 'RF 3D-MoRSE Boltzmann weight',
            'RF_morse_mean': 'RF 3D-MoRSE Mean',
            'RF_morse_global minimum': 'RF 3D-MoRSE Global minimum',
            'RF_morse_rmsd_max': 'RF 3D-MoRSE RMSD max',
            'RF_morse_no_agg': 'RF 3D-MoRSE Non-aggregation',
            'RF_morse_correct': 'RF 3D-MoRSE Ground-truth',
            'RF_mbtr_Boltzman weight': 'RF MBTR Boltzmann weight',
            'RF_mbtr_mean': 'RF MBTR Mean',
            'RF_mbtr_global minimum': 'RF MBTR Global minimum',
            'RF_mbtr_rmsd_max': 'RF MBTR RMSD max',
            'RF_mbtr_no_agg': 'RF MBTR Non-aggregation',
            'RF_mbtr_correct': 'RF MBTR Ground-truth',
            'RF_ecfp_ecfp': 'RF ECFP4 count',
        }
        # create_heatmap(f'{DATA_DIR}/u_test_{NAME}.csv', EACH_SAVE_DIR,
        #                "p_value", keyword, NAME)

    DATA_DIR = 'xxx'
    EACH_SAVE_DIR = f'{SAVE_DIR}/table_1/sign'
    os.makedirs(EACH_SAVE_DIR, exist_ok=True)
    for y_name in tqdm(y_names, desc='y_name'):
        NAME = f'R2_Test_PQC_{y_name}'
        create_heatmap(f'{DATA_DIR}/wilcoxon_sign_{NAME}.csv', EACH_SAVE_DIR,
                       "p_value", keyword, NAME)

    EACH_SAVE_DIR = f'{SAVE_DIR}/table_2'
    os.makedirs(EACH_SAVE_DIR, exist_ok=True)
    for y_name in tqdm(y_names, desc='y_name'):
        NAME = f'R2_Test_PQC_{y_name}'
        keyword = {
            'MIL_moe_no_agg_InstanceWrapperMLPRegressor':
            'MIL MOE Non-aggregation',
            'MIL_moe_no_agg_BagWrapperMLPRegressor': 'MIL MOE Bag-Wrapper',
            'MIL_moe_no_agg_InstanceNetRegressor': 'MIL MOE InstanceNet',
            'MIL_moe_no_agg_BagNetRegressor': 'MIL MOE Bag-Net',
            'MIL_moe_no_agg_AttentionNetRegressor': 'MIL MOE Bag-AttentionNet',
            'MIL_moe_mean_InstanceWrapperMLPRegressor': 'MIL MOE MLP (Mean)',
            'MIL_moe_correct_InstanceWrapperMLPRegressor':
            'MIL MOE MLP (Ground-truth)',
            'RF_moe_no_agg': 'RF MOE Non-aggregation',
            'RF_moe_mean': 'RF MOE Mean',
            'RF_moe_correct': 'RF MOE Ground-truth',
        }
        # create_heatmap(f'{DATA_DIR}/u_test_{NAME}.csv',
        #                EACH_SAVE_DIR,
        #                "p_value",
        #                keyword,
        #                NAME,
        #                xy_font_size=12,
        #                annot_size=8)

    DATA_DIR = 'xxx'
    EACH_SAVE_DIR = f'{SAVE_DIR}/table_2/sign'
    os.makedirs(EACH_SAVE_DIR, exist_ok=True)
    for y_name in tqdm(y_names, desc='y_name'):
        NAME = f'R2_Test_PQC_{y_name}'
        create_heatmap(f'{DATA_DIR}/wilcoxon_sign_{NAME}.csv',
                       EACH_SAVE_DIR,
                       "p_value",
                       keyword,
                       NAME,
                       xy_font_size=12,
                       annot_size=8)

    EACH_SAVE_DIR = f'{SAVE_DIR}/table_3'
    os.makedirs(EACH_SAVE_DIR, exist_ok=True)
    for y_name in tqdm(y_names, desc='y_name'):
        NAME = f'R2_Test_PQC_{y_name}'
        keyword = {
            'gem_ground_truth': 'GEM Ground-truth',
            'gem_global_minimum': 'GEM Global minimum',
            'gem_rmsd_max': 'GEM RMSD max',
            'gem_no_agg': 'GEM Non-aggregation',
            'unimol_ground_truth': 'Uni-Mol Ground-truth',
            'unimol_global_minimum': 'Uni-Mol Global minimum',
            'unimol_rmsd_max': 'Uni-Mol RMSD max',
            'molclr': 'MolCLR',
            'RF_ecfp_ecfp': 'RF ECFP4 count',
        }
        # create_heatmap(f'{DATA_DIR}/u_test_{NAME}.csv',
        #                EACH_SAVE_DIR,
        #                "p_value",
        #                keyword,
        #                NAME,
        #                xy_font_size=14,
        #                annot_size=10)

    DATA_DIR = 'xxx'
    EACH_SAVE_DIR = f'{SAVE_DIR}/table_3/sign'
    os.makedirs(EACH_SAVE_DIR, exist_ok=True)
    for y_name in tqdm(y_names, desc='y_name'):
        NAME = f'R2_Test_PQC_{y_name}'
        create_heatmap(f'{DATA_DIR}/wilcoxon_sign_{NAME}.csv',
                       EACH_SAVE_DIR,
                       "p_value",
                       keyword,
                       NAME,
                       xy_font_size=14,
                       annot_size=10)

    EACH_SAVE_DIR = f'{SAVE_DIR}/table_4'
    os.makedirs(EACH_SAVE_DIR, exist_ok=True)
    # DATA_PATH = '2d3d/output/241121_testing/u_test_MP_MAE_Test.csv'
    DATA_PATH = 'xxx'
    keyword_4_1 = {
        'RF_moe_Boltzman weight': 'RF MOE Boltzmann weight',
        'RF_moe_mean': 'RF MOE Mean',
        'RF_moe_global minimum': 'RF MOE Global minimum',
        'RF_moe_random': 'RF MOE Random',
        'RF_moe_no_agg': 'RF MOE Non-aggregation',
    }
    create_heatmap(DATA_PATH,
                   EACH_SAVE_DIR,
                   "p_value",
                   keyword_4_1,
                   'MP_MAE_Test_sign_moe',
                   xy_font_size=14,
                   annot_size=12)
    keyword_4_2 = {
        'RF_pmapper_Boltzman weight': 'RF Pmapper Boltzmann weight',
        'RF_pmapper_mean': 'RF Pmapper Mean',
        'RF_pmapper_global minimum': 'RF Pmapper Global minimum',
        'RF_pmapper_random': 'RF Pmapper Random',
        'RF_pmapper_no_agg': 'RF Pmapper Non-aggregation',
    }
    create_heatmap(DATA_PATH,
                   EACH_SAVE_DIR,
                   "p_value",
                   keyword_4_2,
                   'MP_MAE_Test_sign_pmapper',
                   xy_font_size=14,
                   annot_size=12)
    keyword_4_3 = {
        'RF_morse_Boltzman weight': 'RF 3D-MoRSE Boltzmann weight',
        'RF_morse_mean': 'RF 3D-MoRSE Mean',
        'RF_morse_global minimum': 'RF 3D-MoRSE Global minimum',
        'RF_morse_random': 'RF 3D-MoRSE Random',
        'RF_morse_no_agg': 'RF 3D-MoRSE Non-aggregation',
    }
    create_heatmap(DATA_PATH,
                   EACH_SAVE_DIR,
                   "p_value",
                   keyword_4_3,
                   'MP_MAE_Test_sign_morse',
                   xy_font_size=14,
                   annot_size=12)
    keyword_4_4 = {
        'RF_mbtr_Boltzman weight': 'RF MBTR Boltzmann weight',
        'RF_mbtr_mean': 'RF MBTR Mean',
        'RF_mbtr_global minimum': 'RF MBTR Global minimum',
        'RF_mbtr_random': 'RF MBTR Random',
        'RF_mbtr_no_agg': 'RF MBTR Non-aggregation',
    }
    create_heatmap(DATA_PATH,
                   EACH_SAVE_DIR,
                   "p_value",
                   keyword_4_4,
                   'MP_MAE_Test_sign_mbtr',
                   xy_font_size=14,
                   annot_size=12)

    EACH_SAVE_DIR = f'{SAVE_DIR}/table_5'
    os.makedirs(EACH_SAVE_DIR, exist_ok=True)
    keyword = {
        'MIL_moe_mean_InstanceWrapperMLPRegressor': 'MLP MOE Mean',
        'MIL_moe_no_agg_InstanceWrapperMLPRegressor':
        'MLP MOE Non-aggregation',
        'MIL_pmapper_mean_InstanceWrapperMLPRegressor': 'MLP Pmapper Mean',
        'MIL_pmapper_no_agg_InstanceWrapperMLPRegressor':
        'MLP Pmapper Non-aggregation',
        'MIL_morse_mean_InstanceWrapperMLPRegressor': 'MLP 3D-MoRSE Mean',
        'MIL_morse_no_agg_InstanceWrapperMLPRegressor':
        'MLP 3D-MoRSE Non-aggregation',
        'MIL_2d_ecfp4_count_InstanceWrapperMLPRegressor': 'MLP ECFP4 count',
        'RF_moe_Boltzman weight': 'RF MOE Boltzmann weight',
        'RF_moe_global minimum': 'RF MOE Global minimum',
        'RF_2d_ecfp4_count': 'RF ECFP4 count',
        'gem_global_minimum': 'GEM Global minimum',
        'gem_no_agg': 'GEM Non-aggregation',
        'unimol_global minimum': 'Uni-Mol Global minimum',
        'unimol_no_agg': 'Uni-Mol Non-aggregation',
        'molclr': 'MolCLR',
    }
    DATA_PATH = 'xxx'
    create_heatmap(DATA_PATH, EACH_SAVE_DIR, "p_value", keyword,
                   'MP_MAE_Test_unsign')
    DATA_PATH = 'xxx'
    create_heatmap(DATA_PATH, EACH_SAVE_DIR, "p_value", keyword,
                   'MP_MAE_Test_sign')

    EACH_SAVE_DIR = f'{SAVE_DIR}/table_6'
    os.makedirs(EACH_SAVE_DIR, exist_ok=True)
    DATA_PATH = 'xxx'
    create_heatmap(DATA_PATH,
                   EACH_SAVE_DIR,
                   "p_value",
                   keyword_4_1,
                   'APTC1_MAE_Test_sign_moe',
                   xy_font_size=14,
                   annot_size=12)
    create_heatmap(DATA_PATH,
                   EACH_SAVE_DIR,
                   "p_value",
                   keyword_4_2,
                   'APTC1_MAE_Test_sign_pmapper',
                   xy_font_size=14,
                   annot_size=12)
    create_heatmap(DATA_PATH,
                   EACH_SAVE_DIR,
                   "p_value",
                   keyword_4_3,
                   'APTC1_MAE_Test_sign_morse',
                   xy_font_size=14,
                   annot_size=12)
    create_heatmap(DATA_PATH,
                   EACH_SAVE_DIR,
                   "p_value",
                   keyword_4_4,
                   'APTC1_MAE_Test_sign_mbtr',
                   xy_font_size=14,
                   annot_size=12)

    DATA_PATH = 'xxx'
    create_heatmap(DATA_PATH,
                   EACH_SAVE_DIR,
                   "p_value",
                   keyword_4_1,
                   'APTC2_MAE_Test_sign_moe',
                   xy_font_size=14,
                   annot_size=12)
    create_heatmap(DATA_PATH,
                   EACH_SAVE_DIR,
                   "p_value",
                   keyword_4_2,
                   'APTC2_MAE_Test_sign_pmapper',
                   xy_font_size=14,
                   annot_size=12)
    create_heatmap(DATA_PATH,
                   EACH_SAVE_DIR,
                   "p_value",
                   keyword_4_3,
                   'APTC2_MAE_Test_sign_morse',
                   xy_font_size=14,
                   annot_size=12)
    create_heatmap(DATA_PATH,
                   EACH_SAVE_DIR,
                   "p_value",
                   keyword_4_4,
                   'APTC2_MAE_Test_sign_mbtr',
                   xy_font_size=14,
                   annot_size=12)

    EACH_SAVE_DIR = f'{SAVE_DIR}/table_7'
    os.makedirs(EACH_SAVE_DIR, exist_ok=True)
    keyword = {
        'MIL_moe_mean_InstanceWrapperMLPRegressor': 'MLP MOE Mean',
        'MIL_moe_no_agg_InstanceWrapperMLPRegressor':
        'MLP MOE Non-aggregation',
        'MIL_pmapper_mean_InstanceWrapperMLPRegressor': 'MLP Pmapper Mean',
        'MIL_pmapper_no_agg_InstanceWrapperMLPRegressor':
        'MLP Pmapper Non-aggregation',
        'MIL_morse_mean_InstanceWrapperMLPRegressor': 'MLP 3D-MoRSE Mean',
        'MIL_morse_no_agg_InstanceWrapperMLPRegressor':
        'MLP 3D-MoRSE Non-aggregation',
        'MIL_2d_ecfp_bit_InstanceWrapperMLPRegressor': 'MLP ECFP4 bit',
        'MIL_2d_ecfp_count_InstanceWrapperMLPRegressor': 'MLP ECFP4 count',
        'MIL_2d_pharm_2d_InstanceWrapperMLPRegressor': 'MLP 2D PFP',
        'RF_pmapper_mean': 'RF Pmapper Mean',
        'RF_pmapper_no_agg': 'RF Pmapper Non-aggregation',
        'RF_2d_ecfp_bit': 'RF ECFP4 bit',
        'RF_2d_ecfp_count': 'RF ECFP4 count',
        'RF_2d_pharm_2d': 'RF 2D PFP',
        'gem_global_minimum': 'GEM Global minimum',
        'gem_no_agg': 'GEM Non-aggregation',
        'unimol_global minimum': 'Uni-Mol Global minimum',
        'unimol_no_agg': 'Uni-Mol Non-aggregation',
        'molclr': 'MolCLR',
    }
    DATA_PATH = 'xxx'
    create_heatmap(DATA_PATH, EACH_SAVE_DIR, "p_value", keyword,
                   'APTC1_MAE_Test_unsign')
    DATA_PATH = 'xxx'
    create_heatmap(DATA_PATH, EACH_SAVE_DIR, "p_value", keyword,
                   'APTC1_MAE_Test_sign')
    DATA_PATH = 'xxx'
    create_heatmap(DATA_PATH, EACH_SAVE_DIR, "p_value", keyword,
                   'APTC2_MAE_Test_unsign')
    DATA_PATH = 'xxx'
    create_heatmap(DATA_PATH, EACH_SAVE_DIR, "p_value", keyword,
                   'APTC2_MAE_Test_sign')

    keyword = {
        'RF_moe_Boltzman weight': 'RF MOE Boltzmann weight',
        'RF_moe_mean': 'RF MOE Mean',
        'RF_moe_global minimum': 'RF MOE Global minimum',
        'RF_moe_random': 'RF MOE Random',
        'RF_moe_no_agg': 'RF MOE Non-aggregation',
        'RF_pmapper_Boltzman weight': 'RF Pmapper Boltzmann weight',
        'RF_pmapper_mean': 'RF Pmapper Mean',
        'RF_pmapper_global minimum': 'RF Pmapper Global minimum',
        'RF_pmapper_random': 'RF Pmapper Random',
        'RF_pmapper_no_agg': 'RF Pmapper Non-aggregation',
        'RF_morse_Boltzman weight': 'RF 3D-MoRSE Boltzmann weight',
        'RF_morse_mean': 'RF 3D-MoRSE Mean',
        'RF_morse_global minimum': 'RF 3D-MoRSE Global minimum',
        'RF_morse_random': 'RF 3D-MoRSE Random',
        'RF_morse_no_agg': 'RF 3D-MoRSE Non-aggregation',
        'RF_mbtr_Boltzman weight': 'RF MBTR Boltzmann weight',
        'RF_mbtr_mean': 'RF MBTR Mean',
        'RF_mbtr_global minimum': 'RF MBTR Global minimum',
        'RF_mbtr_random': 'RF MBTR Random',
        'RF_mbtr_no_agg': 'RF MBTR Non-aggregation',
    }
    DATA_PATH = 'xxx'
    anova(DATA_PATH, keyword)
    DATA_PATH = 'xxx'
    anova(DATA_PATH, keyword)
    DATA_PATH = 'xxx'
    anova(DATA_PATH, keyword)
