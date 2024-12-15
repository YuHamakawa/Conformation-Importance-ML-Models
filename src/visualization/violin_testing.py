'''
Perform statistical tests.
1. visualize it with a Violin plot.
2. perform the Wilcoxon signed-rank test.
3. calculate the mean and standard deviation.
'''

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


def plot_violin(csv_path, save_dir, name, y_lim, y_label):
    """
    Function to create violin plots for all models.
    Args:
        csv_path: str, path to the CSV file containing the data
        save_dir: str, directory to save the plots
        name: str, name of the output PDF file
        y_lim: tuple, (min, max) of y-axis
        y_label: str, label of y-axis
    """
    save_dir = f'{save_dir}/violins'
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    long_df = df.melt(id_vars=["model_name"],
                      var_name="seed_fold",
                      value_name="value")

    sns.set_theme()
    plt.figure(figsize=(20, 10))
    sns.violinplot(data=long_df,
                   x="model_name",
                   y="value",
                   palette="husl",
                   density_norm="width",
                   inner="quartile")

    num_labels = len(df['model_name'].unique())
    plt.xticks(rotation=90,
               ticks=range(num_labels),
               labels=range(1, num_labels + 1),
               fontsize=8)

    plt.yticks(fontsize=24)
    plt.xlabel("")
    plt.ylabel(y_label, fontsize=24)
    if y_lim is None:
        ymax = long_df['value'].max()
        plt.ylim(0, ymax * 1.1)
    else:
        plt.ylim(y_lim[0], y_lim[1])

    plt.tight_layout()

    plt.savefig(f'{save_dir}/violin_{name}.pdf')
    plt.close()

    save_dir = f'{save_dir}/model_csv'
    os.makedirs(save_dir, exist_ok=True)
    model_names = df['model_name'].unique()
    model_names_df = pd.DataFrame(model_names, columns=['Model Names'])
    model_names_df.to_csv(f'{save_dir}/model_names_{name}.csv', index=False)


def mann_whitney_u_test_all_pairs(csv_path, save_dir, name):
    """
    Perform Mann-Whitney U tests for all pairs of models in the given CSV file and save the results.

    Parameters:
    csv_path (str): Path to the CSV file containing the data. The first column should contain model names, and the remaining columns should contain numeric data.
    save_dir (str): Directory where the results CSV file will be saved.
    name (str): Name to be used for the results CSV file.

    Returns:
    None

    The function reads the data from the CSV file, performs Mann-Whitney U tests for all pairs of models, 
    applies Holm's correction for multiple comparisons, and saves the results to a new CSV file in the specified directory.
    """
    df = pd.read_csv(csv_path)
    model_names = df["model_name"].values
    numeric_data = df.iloc[:, 1:]
    results = []

    # Enumerate all combinations of model pairs
    for (i, name1), (j,
                     name2) in itertools.combinations(enumerate(model_names),
                                                      2):
        data1 = numeric_data.iloc[i].values  # 行iのデータ
        data2 = numeric_data.iloc[j].values  # 行jのデータ
        u_stat, p_value = stats.mannwhitneyu(data1,
                                             data2,
                                             alternative="two-sided")  # U検定
        results.append({
            "model_1": name1,
            "model_2": name2,
            "u_statistic": u_stat,
            "p_value": p_value
        })

    result_df = pd.DataFrame(results)
    corrected_p = multipletests(result_df['p_value'], method='holm')
    result_df['corrected_p'] = corrected_p[1]
    result_df.to_csv(f'{save_dir}/u_test_{name}.csv', index=False)


def wilcoxon_signed_rank_test_all_pairs(csv_path, save_dir, name):
    """
    Perform Wilcoxon signed-rank test for all pairs of models in the given CSV file.

    This function reads a CSV file containing model names and their corresponding numeric data,
    performs the Wilcoxon signed-rank test for all pairs of models, and saves the results to a CSV file.

    Parameters:
    csv_path (str): The path to the input CSV file containing model names and numeric data.
    save_dir (str): The directory where the results CSV file will be saved.
    name (str): The name to be used for the results CSV file.

    The input CSV file should have the following structure:
    - The first column should contain model names.
    - The subsequent columns should contain numeric data for each model.

    The output CSV file will contain the following columns:
    - model_1: The name of the first model in the pair.
    - model_2: The name of the second model in the pair.
    - statistic: The test statistic from the Wilcoxon signed-rank test.
    - p_value: The p-value from the Wilcoxon signed-rank test.
    - corrected_p: The p-value corrected for multiple comparisons using the Holm method.

    Notes:
    - The Wilcoxon signed-rank test is performed only for pairs of models with at least 10 non-zero differences.
    - The p-values are corrected for multiple comparisons using the Holm method.
    """
    df = pd.read_csv(csv_path)
    model_names = df["model_name"].values
    numeric_data = df.iloc[:, 1:]
    results = []

    for (i, name1), (j,
                     name2) in itertools.combinations(enumerate(model_names),
                                                      2):
        data1 = numeric_data.iloc[i].values  # 行iのデータ
        data2 = numeric_data.iloc[j].values  # 行jのデータ
        differences = data1 - data2
        non_zero_diff_count = np.sum(differences != 0)
        if (differences.sum() != 0) and (non_zero_diff_count >= 10):
            stat, p_value = stats.wilcoxon(data1,
                                           data2,
                                           alternative="two-sided")
            results.append({
                "model_1": name1,
                "model_2": name2,
                "statistic": stat,
                "p_value": p_value
            })

    result_df = pd.DataFrame(results)
    corrected_p = multipletests(result_df['p_value'], method='holm')
    result_df['corrected_p'] = corrected_p[1]
    result_df.to_csv(f'{save_dir}/wilcoxon_sign_{name}.csv', index=False)


def create_heatmap(csv_path, save_dir, p_column, keyword, name):
    """
    Create a heatmap from a CSV file containing p-values and save it as a PDF.

    Parameters:
    csv_path (str): Path to the CSV file containing the p-values.
    save_dir (str): Directory where the heatmap and related files will be saved.
    p_column (str): The column name in the CSV file that contains the p-values.
    keyword (str): Keyword to filter the model names. Only models containing this keyword will be included in the heatmap.
    name (str): Name to be used for the saved heatmap file and related files.

    Returns:
    None
    """
    save_dir = f'{save_dir}/heatmaps'
    os.makedirs(save_dir, exist_ok=True)

    results_df = pd.read_csv(csv_path)
    # Get unique model names
    models = sorted(set(results_df["model_1"]).union(results_df["model_2"]))

    if keyword is not None:
        models = [model for model in models if keyword in model]

    # Initialize p-value matrix for heatmap
    p_value_matrix = pd.DataFrame(np.nan, index=models, columns=models)

    # Fill the matrix with p-values
    for _, row in results_df.iterrows():
        model_1, model_2, p_value = row["model_1"], row["model_2"], row[
            p_column]
        if (model_1 in models) and (model_2 in models):
            p_value_matrix.loc[model_1, model_2] = p_value
            p_value_matrix.loc[model_2, model_1] = p_value

    # Generate a mask for the upper triangle
    mask = p_value_matrix.isnull()

    # Plotting
    plt.figure(figsize=(10, 10))
    # Generate a custom diverging colormap
    cmap = sns.color_palette("Spectral_r", as_cmap=True)
    ax = sns.heatmap(p_value_matrix,
                     cmap=cmap,
                     vmax=0.1,
                     cbar_kws={"shrink": .6},
                     square=True,
                     mask=mask)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    # Get the colorbar
    cbar = ax.collections[0].colorbar
    # Change the font size of the colorbar label
    cbar.set_label('p-value', fontsize=18)
    # Change the font size of the colorbar tick labels if needed
    cbar.ax.tick_params(labelsize=16)

    # Save the file
    plt.savefig(f'{save_dir}/heatmap_{name}.pdf')
    plt.close()

    save_dir = f'{save_dir}/column_csv'
    os.makedirs(save_dir, exist_ok=True)
    df_columns = pd.DataFrame(p_value_matrix.columns.to_list(),
                              columns=['Model Names'])
    df_columns.to_csv(f'{save_dir}/column_{name}.csv', index=False)


if __name__ == '__main__':
    SAVE_DIR = 'xxx'
    os.makedirs(SAVE_DIR, exist_ok=True)

    PLOT_VIOLIN = True
    TESTING = False  # Mann-Whitney U test
    PLOT_HEATMAP = False
    SIGN_TESTING = False  # Wilcoxon signed-rank test

    ### PQC dataset ###
    y_names = ['dipoleMoment', 'homo', 'gap', 'lumo', 'energy', 'enthalpy']
    metrics_list = [
        'MAE_Train_PQC', 'MAE_Test_PQC', 'R2_Train_PQC', 'R2_Test_PQC'
    ]
    for y_name in tqdm(y_names, desc='y_name'):
        for metrics in tqdm(metrics_list, desc='metrics'):
            NAME = f'{metrics}_{y_name}'
            CSV_PATH = f'xxx'
            if PLOT_VIOLIN:
                if metrics.split('_')[0] == 'MAE':
                    y_label = 'Mean Absolute Error'
                    y_lim = None
                elif metrics.split('_')[0] == 'R2':
                    y_label = 'Coefficient of Determination'
                    y_lim = (0, 1.1)

                plot_violin(CSV_PATH, SAVE_DIR, NAME, y_lim, y_label)
            if TESTING:
                mann_whitney_u_test_all_pairs(CSV_PATH, SAVE_DIR, NAME)
            if PLOT_HEATMAP:
                create_heatmap(f'{SAVE_DIR}/u_test_{NAME}.csv', SAVE_DIR,
                               "p_value", None, NAME)

    os.makedirs(SAVE_DIR, exist_ok=True)
    if SIGN_TESTING:
        for y_name in tqdm(y_names, desc='y_name'):
            metrics = 'R2_Test_PQC'
            NAME = f'{metrics}_{y_name}'
            CSV_PATH = f'xxx'
            # wilcoxon_signed_rank_test_all_pairs(CSV_PATH, SAVE_DIR, NAME)

    # SAVE_DIR = '2d3d/output/241121_testing'
    os.makedirs(SAVE_DIR, exist_ok=True)

    ### MP dataset ###
    CSV_PATH = 'xxx'
    NAME = 'MP_MAE_Train'
    if PLOT_VIOLIN:
        plot_violin(CSV_PATH,
                    SAVE_DIR,
                    NAME, (0, 40),
                    y_label='Mean Absolute Error')
    if TESTING:
        mann_whitney_u_test_all_pairs(CSV_PATH, SAVE_DIR, NAME)
    if PLOT_HEATMAP:
        create_heatmap(f'{SAVE_DIR}/u_test_{NAME}.csv', SAVE_DIR, "p_value",
                       None, NAME)

    CSV_PATH = 'xxx'
    NAME = 'MP_MAE_Test'
    if PLOT_VIOLIN:
        plot_violin(CSV_PATH,
                    SAVE_DIR,
                    'MP_MAE_Test', (0, 120),
                    y_label='Mean Absolute Error')
    if TESTING:
        mann_whitney_u_test_all_pairs(CSV_PATH, SAVE_DIR, NAME)
    if PLOT_HEATMAP:
        create_heatmap(f'{SAVE_DIR}/u_test_{NAME}.csv', SAVE_DIR, "p_value",
                       None, NAME)

    if SIGN_TESTING:
        CSV_PATH = 'xxx'
        NAME = 'MAE_Test_MP'
        wilcoxon_signed_rank_test_all_pairs(CSV_PATH, SAVE_DIR, NAME)

    CSV_PATH = 'xxx'
    NAME = 'MP_R2_Train'
    if PLOT_VIOLIN:
        plot_violin(CSV_PATH,
                    SAVE_DIR,
                    'MP_R2_Train', (0, 1.1),
                    y_label='Coefficient of Determination')
        mann_whitney_u_test_all_pairs(CSV_PATH, SAVE_DIR, NAME)
    if PLOT_HEATMAP:
        create_heatmap(f'{SAVE_DIR}/u_test_{NAME}.csv', SAVE_DIR, "p_value",
                       None, NAME)

    CSV_PATH = 'xxx'
    NAME = 'MP_R2_Test'
    if PLOT_VIOLIN:
        plot_violin(CSV_PATH,
                    SAVE_DIR,
                    'MP_R2_Test', (0, 1.1),
                    y_label='Coefficient of Determination')
    if TESTING:
        mann_whitney_u_test_all_pairs(CSV_PATH, SAVE_DIR, NAME)
    if PLOT_HEATMAP:
        create_heatmap(f'{SAVE_DIR}/u_test_{NAME}.csv', SAVE_DIR, "p_value",
                       None, NAME)

    ### APTC1 dataset ###
    CSV_PATH = 'xxx'
    NAME = 'APTC1_MAE_Train'
    if PLOT_VIOLIN:
        plot_violin(CSV_PATH,
                    SAVE_DIR,
                    'APTC1_MAE_Train', (0, 0.5),
                    y_label='Mean Absolute Error')
    if TESTING:
        mann_whitney_u_test_all_pairs(CSV_PATH, SAVE_DIR, NAME)
    if PLOT_HEATMAP:
        create_heatmap(f'{SAVE_DIR}/u_test_{NAME}.csv', SAVE_DIR, "p_value",
                       None, NAME)

    CSV_PATH = 'xxx'
    NAME = 'APTC1_MAE_Test'
    if PLOT_VIOLIN:
        plot_violin(CSV_PATH,
                    SAVE_DIR,
                    'APTC1_MAE_Test', (0, 0.7),
                    y_label='Mean Absolute Error')
    if TESTING:
        mann_whitney_u_test_all_pairs(CSV_PATH, SAVE_DIR, 'APTC1_MAE_Test')
    if PLOT_HEATMAP:
        create_heatmap(f'{SAVE_DIR}/u_test_APTC1_MAE_Test.csv', SAVE_DIR,
                       "p_value", None, NAME)

    if SIGN_TESTING:
        CSV_PATH = 'xxx'
        NAME = 'MAE_Test_APTC1'
        wilcoxon_signed_rank_test_all_pairs(CSV_PATH, SAVE_DIR, NAME)

    CSV_PATH = 'xxx'
    NAME = 'APTC1_R2_Train'
    if PLOT_VIOLIN:
        plot_violin(CSV_PATH,
                    SAVE_DIR,
                    'APTC1_R2_Train', (0, 1.1),
                    y_label='Coefficient of Determination')
    if TESTING:
        mann_whitney_u_test_all_pairs(CSV_PATH, SAVE_DIR, NAME)
    if PLOT_HEATMAP:
        create_heatmap(f'{SAVE_DIR}/u_test_{NAME}.csv', SAVE_DIR, "p_value",
                       None, NAME)

    CSV_PATH = 'xxx'
    NAME = 'APTC1_R2_Test'
    if PLOT_VIOLIN:
        plot_violin(CSV_PATH,
                    SAVE_DIR,
                    'APTC1_R2_Test', (0, 1.1),
                    y_label='Coefficient of Determination')
    if TESTING:
        mann_whitney_u_test_all_pairs(CSV_PATH, SAVE_DIR, NAME)
    if PLOT_HEATMAP:
        create_heatmap(f'{SAVE_DIR}/u_test_{NAME}.csv', SAVE_DIR, "p_value",
                       None, NAME)

    ### APTC2 dataset ###
    CSV_PATH = 'xxx'
    NAME = 'APTC2_MAE_Train'
    if PLOT_VIOLIN:
        plot_violin(CSV_PATH,
                    SAVE_DIR,
                    'APTC2_MAE_Train', (0, 0.5),
                    y_label='Mean Absolute Error')
    if TESTING:
        mann_whitney_u_test_all_pairs(CSV_PATH, SAVE_DIR, NAME)
    if PLOT_HEATMAP:
        create_heatmap(f'{SAVE_DIR}/u_test_{NAME}.csv', SAVE_DIR, "p_value",
                       None, NAME)

    CSV_PATH = 'xxx'
    NAME = 'APTC2_MAE_Test'
    if PLOT_VIOLIN:
        plot_violin(CSV_PATH,
                    SAVE_DIR,
                    'APTC2_MAE_Test', (0, 1.5),
                    y_label='Mean Absolute Error')
    if TESTING:
        mann_whitney_u_test_all_pairs(CSV_PATH, SAVE_DIR, 'APTC2_MAE_Test')
    if PLOT_HEATMAP:
        create_heatmap(f'{SAVE_DIR}/u_test_APTC2_MAE_Test.csv', SAVE_DIR,
                       "p_value", None, NAME)

    if SIGN_TESTING:
        CSV_PATH = 'xxx'
        NAME = 'MAE_Test_APTC2'
        wilcoxon_signed_rank_test_all_pairs(CSV_PATH, SAVE_DIR, NAME)
