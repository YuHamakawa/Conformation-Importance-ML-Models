import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    # r2 = r2_score(true, pred)
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    return mse, mae


def plot_scatter_aptc1(save_dir, fig_name, y_train, y_pred_train, y_test,
                       y_pred_test):
    # Calculate and print metrics
    mse_train, mae_train = calcMetrics(y_train, y_pred_train)
    mse_test, mae_test = calcMetrics(y_test, y_pred_test)

    # Visualize prediction results
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(y_train, y_pred_train, label='Train', alpha=0.7)
    plt.scatter(y_test, y_pred_test, label='Test', alpha=0.7)
    # Prepare for drawing a diagonal line
    yvalues = np.concatenate([y_test, y_pred_test]).flatten()
    ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
    # Display R2, MSE, MAE values on the graph
    plt.text(
        0.01,
        0.98,
        #f'         (Train)  (Test)\n    R2: {r2_train:.3f}, {r2_test:.3f}\n MSE: {mse_train:.3f}, {mse_test:.3f}\n MAE: {mae_train:.3f}, {mae_test:.3f}',
        f'(Train) MSE: {mse_train:.3f}, MAE: {mae_train:.3f}\n(Test) MSE: {mse_test:.3f}, MAE: {mae_test:.3f}',
        ha='left',
        va='top',
        fontsize=16,
        transform=ax.transAxes)
    # Draw a diagonal line
    plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01],
             [ymin - yrange * 0.01, ymax + yrange * 0.01],
             color='k')
    plt.xlabel('Observations', fontsize=18)
    plt.ylabel('Predictions', fontsize=18)
    plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.legend(loc='lower right', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{fig_name}.png')
    plt.close()


def plot_scatter_aptc2(data_dir, save_dir, fig_name):
    df_list = []
    for i in range(40):
        if fig_name == 'aptc2_uni_no-agg':
            data_path = f'{data_dir}/fold_{i}/test_set_predictions_mean.csv'
        else:
            data_path = f'{data_dir}/fold_{i}/test_set_predictions.csv'
        df = pd.read_csv(data_path)
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    y_test = df['actual']
    y_pred = df['predicted']
    mse, mae = calcMetrics(y_test, y_pred)

    # fig = plt.figure(figsize=(10, 5))
    # ax = fig.add_subplot(1, 1, 1)
    # plt.scatter(y_test, y_pred, alpha=0.7, color='tab:orange')

    plt.figure(figsize=(10, 5))
    sns.set_theme()
    ax = sns.scatterplot(x=y_test, y=y_pred)

    yvalues = np.concatenate([y_test, y_pred]).flatten()
    ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
    plt.text(0.01,
             0.98,
             f'MSE: {mse:.3f}, MAE: {mae:.3f}',
             ha='left',
             va='top',
             fontsize=16,
             transform=ax.transAxes)
    plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01],
             [ymin - yrange * 0.01, ymax + yrange * 0.01],
             color='k')
    plt.xlabel('Observations', fontsize=18)
    plt.ylabel('Predictions', fontsize=18)
    plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{fig_name}.pdf')
    plt.close()


if __name__ == '__main__':
    save_dir = 'xxx'
    os.makedirs(save_dir, exist_ok=True)

    aptc2_data_path = 'xxx'
    plot_scatter_aptc2(aptc2_data_path, save_dir, 'Pma_aptc2_MIL-IW_no-agg')
    aptc2_data_path = 'xxx'
    plot_scatter_aptc2(aptc2_data_path, save_dir, 'Morse_aptc2_MIL-IW_no-agg')
    aptc2_data_path = 'xxx'
    plot_scatter_aptc2(aptc2_data_path, save_dir, '2DPFP_aptc2_RF_')
    aptc2_data_path = 'xxx'
    plot_scatter_aptc2(aptc2_data_path, save_dir, 'aptc2_uni_no-agg')
