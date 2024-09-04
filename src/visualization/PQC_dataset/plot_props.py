'''
Plotting for the CSV file, which summarizes the results of MOPAC calculations.
For each calculated property, plot the coefficient of variation.
'''
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def preprocess_df_cv(data_path):
    '''coefficient of variation
    this function does not match this analysis standard deviation and mean would be better.
    '''
    desc_df = pd.read_csv(data_path)
    use_columns = [
        'mol2d_idx', 'final_heat_of_formation', 'total_energy', 'homo_ev',
        'lumo_ev', 'dipole_sum_total'
    ]
    # Extract the specified columns
    desc_df = desc_df[use_columns]

    # Rename the columns
    desc_df = desc_df.rename(
        columns={
            'mol2d_idx': 'cid',
            'final_heat_of_formation': 'Enthalpy',
            'total_energy': 'Energy',
            'homo_ev': 'HOMO',
            'lumo_ev': 'LUMO',
            'dipole_sum_total': 'Dipole Moment'
        })
    # Add new column 'gap' which is the difference between 'lumo' and 'homo'
    desc_df['Gap'] = desc_df['LUMO'] - desc_df['HOMO']
    # group cid
    desc_df['cid'] = desc_df['cid'].apply(lambda x: x.split('_')[0])
    # Group by 'cid' and calculate standard deviation for each group
    std_df = desc_df.groupby('cid').std()
    # Group by 'cid' and calculate mean for each group
    mean_df = desc_df.groupby('cid').mean()
    # Calculate coefficient of variation for each group
    cv_df = std_df / mean_df
    # change columns
    cv_df = cv_df[[
        'Dipole Moment', 'HOMO', 'Gap', 'LUMO', 'Energy', 'Enthalpy'
    ]]
    # Remove compounds with only one conformer (5481 compounds)
    # Remove compounds with mean equal to 0 (11 compounds)
    cv_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    cv_df.dropna(inplace=True)
    return cv_df


def preprocess_df_std(data_path):
    '''calclate standard deciation and mean to each compound
    '''
    desc_df = pd.read_csv(data_path)
    use_columns = [
        'mol2d_idx', 'final_heat_of_formation', 'total_energy', 'homo_ev',
        'lumo_ev', 'dipole_sum_total'
    ]
    # Extract the specified columns
    desc_df = desc_df[use_columns]

    # Rename the columns
    desc_df = desc_df.rename(
        columns={
            'mol2d_idx': 'cid',
            'final_heat_of_formation': 'Enthalpy',
            'total_energy': 'Energy',
            'homo_ev': 'HOMO',
            'lumo_ev': 'LUMO',
            'dipole_sum_total': 'Dipole moment'
        })
    # Add new column 'gap' which is the difference between 'lumo' and 'homo'
    desc_df['HOMO-LUMO gap'] = desc_df['LUMO'] - desc_df['HOMO']
    # group cid
    desc_df['cid'] = desc_df['cid'].apply(lambda x: x.split('_')[0])
    # Group by 'cid' and calculate standard deviation for each group
    std_df = desc_df.groupby('cid').std()
    # change columns
    std_df = std_df[[
        'Dipole moment', 'HOMO', 'HOMO-LUMO gap', 'LUMO', 'Energy', 'Enthalpy'
    ]]
    std_df.dropna(inplace=True)

    # Group by 'cid' and calculate mean for each group
    mean_df = desc_df.groupby('cid').mean()
    mean_df = mean_df[[
        'Dipole moment', 'HOMO', 'HOMO-LUMO gap', 'LUMO', 'Energy', 'Enthalpy'
    ]]
    mean_series = mean_df.mean()
    range_series = mean_df.max() - mean_df.min()
    quantile_series = mean_df.quantile(0.75) - mean_df.quantile(0.25)
    return std_df, mean_series, range_series, quantile_series


def plot_violin_property(std_df, quantile_series, save_dir):
    '''Plot violin plot for each property'''
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="variable",
                   y="value",
                   data=pd.melt(std_df / quantile_series),
                   fill=False)
    plt.xticks(rotation=45, fontsize=14)
    # plt.yscale('log')

    # # Display mean and range for each property
    # for i, (mean, range_) in enumerate(zip(mean_series, range_series)):
    #     plt.text(i - 0.1,
    #              std_df.values.flatten().max() + 0.1,
    #              f'Mean-Mean:\n{mean:.2f}\nMean-Range:\n{range_:.2f}',
    #              verticalalignment='top',
    #              fontsize=16)

    # plt.title('Violin plot of standard deviation for each property',fontsize=16)
    plt.xlabel('Property', fontsize=18)
    plt.ylabel('Standard Deviation / Interquartile Range', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    # plt.subplots_adjust(top=0.9)
    # plt.ylim(0, 1)
    plt.savefig(f'{save_dir}/violin_plot.png')
    plt.close()


def plot_histograms(std_df, mean_series, range_series, save_dir):
    '''Plot histograms for each property'''
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    for i, var in enumerate(std_df.columns):
        row = i // 3
        col = i % 3
        sns.histplot(std_df[var].values, kde=False, ax=axs[row, col])
        axs[row, col].set_title(var, fontsize=22)
        # Remove y-axis label
        axs[row, col].set_ylabel('')

        # Display mean and range for each property
        mean = mean_series[i]
        range_ = range_series[i]
        axs[row, col].text(
            0.95,
            0.95,
            f'Mean-Range: {range_:.2f}',  # f'Mean-Mean: {mean:.2f}\nMean-Range: {range_:.2f}',
            verticalalignment='top',
            horizontalalignment='right',
            transform=axs[row, col].transAxes,
            fontsize=19)

        axs[row, col].tick_params(axis='both', which='major', labelsize=16)

    # plt.suptitle('Histgrams of standard deviation for each property',fontsize=20)
    axs[0, 0].set_xlabel('Standard Deviation', fontsize=22)
    axs[0, 0].set_ylabel('Frequency', fontsize=22)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/histograms.png')
    plt.close()


def plot_rotatable_histgram(data_path, save_dir):
    '''
    plot rotatable bonds histgram for figure in paper
    '''
    desc_df = pd.read_csv(data_path, usecols=['mol2d_idx', 'RotatableBonds'])
    desc_df['cid'] = desc_df['mol2d_idx'].apply(lambda x: x.split('_')[0])
    mean_df = desc_df.groupby('cid')['RotatableBonds'].mean().reset_index()
    # Plot histogram with seaborn
    plt.figure(figsize=(10, 6))
    sns.histplot(data=mean_df, x='RotatableBonds', bins=28, kde=False)
    # Get descriptive statistics
    desc_stats = mean_df['RotatableBonds'].describe()
    # Add descriptive statistics to the plot
    stats_text = f"number of compounds: {desc_stats['count']:.0f}\nmean: {desc_stats['mean']:.2f}\nmax: {desc_stats['max']:.0f}\nthird quartile: {desc_stats['75%']:.0f}\nmedian: {desc_stats['50%']:.0f}\nfirst quartile: {desc_stats['25%']:.0f}\nmin: {desc_stats['min']:.0f}"
    plt.text(0.95,
             0.95,
             stats_text,
             fontsize=16,
             va='top',
             ha='right',
             transform=plt.gca().transAxes)

    # plt.title('Histogram of Mean Rotatable Bonds')
    plt.xlabel('Number of rotatable bonds', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(f'{save_dir}/histogram_rotatable_bond.png')


def plot_rmsd_histgram(data_path, save_dir):
    '''
    plot mean of rmsd from correct conformer histgram for figure in paper
    '''
    desc_df = pd.read_csv(data_path, usecols=['mol2d_idx', 'rmsd'])
    desc_df['cid'] = desc_df['mol2d_idx'].apply(lambda x: x.split('_')[0])
    mean_df = desc_df.groupby('cid')['rmsd'].mean().reset_index()
    # Plot histogram with seaborn
    plt.figure(figsize=(10, 6))
    sns.histplot(data=mean_df, x='rmsd', bins=28, kde=False)
    # Get descriptive statistics
    desc_stats = mean_df['rmsd'].describe()
    # Add descriptive statistics to the plot
    stats_text = f"number of compounds: {desc_stats['count']:.0f}\nmean: {desc_stats['mean']:.3f}\nmax: {desc_stats['max']:.3f}\nthird quartile: {desc_stats['75%']:.3f}\nmedian: {desc_stats['50%']:.3f}\nfirst quartile: {desc_stats['25%']:.3f}\nmin: {desc_stats['min']:.3f}"
    plt.text(0.95,
             0.95,
             stats_text,
             fontsize=16,
             va='top',
             ha='right',
             transform=plt.gca().transAxes)

    plt.xlabel('Average RMSD from correct conformer', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(f'{save_dir}/histogram_rmsd.png')


def plot_conformer_histgram(data_path, save_dir):
    '''
    plot number of conformer histgram for figure in paper
    '''
    desc_df = pd.read_csv(data_path, usecols=['mol2d_idx'])
    desc_df['cid'] = desc_df['mol2d_idx'].apply(lambda x: x.split('_')[0])
    mean_df = desc_df.groupby('cid')['mol2d_idx'].size().reset_index()
    # Plot histogram with seaborn
    plt.figure(figsize=(10, 6))
    sns.histplot(data=mean_df, x='mol2d_idx', bins=28, kde=False)
    # Get descriptive statistics
    desc_stats = mean_df['mol2d_idx'].describe()
    # Add descriptive statistics to the plot
    stats_text = f"number of conformers: {desc_df.shape[0]:.0f}\nmean: {desc_stats['mean']:.3f}\nmax: {desc_stats['max']:.3f}\nthird quartile: {desc_stats['75%']:.3f}\nmedian: {desc_stats['50%']:.3f}\nfirst quartile: {desc_stats['25%']:.3f}\nmin: {desc_stats['min']:.3f}"
    plt.text(0.95,
             0.95,
             stats_text,
             fontsize=16,
             va='top',
             ha='right',
             transform=plt.gca().transAxes)

    plt.xlabel('Number of conformers generated from each compound',
               fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(f'{save_dir}/histogram_conformer.png')


if __name__ == '__main__':
    TEST_MODE = False
    DATA_PATH = 'xxx'
    if TEST_MODE:
        DATA_PATH = 'xxx'
    SAVE_DIR = 'xxx'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # plot_rotatable_histgram(DATA_PATH, SAVE_DIR)
    # plot_rmsd_histgram(DATA_PATH, SAVE_DIR)
    # plot_conformer_histgram(DATA_PATH, SAVE_DIR)

    std_df, mean_series, range_series, quantile_series = preprocess_df_std(
        DATA_PATH)
    # std_df.to_csv('2d3d/dataset/PubChemQC-PM6/step12/standard_deviation.csv')
    # plot_violin_property(std_df, quantile_series, SAVE_DIR)
    plot_histograms(std_df, mean_series, range_series, SAVE_DIR)
