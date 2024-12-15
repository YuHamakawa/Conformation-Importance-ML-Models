import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn.functional as F
import yaml
from dataset.dataset_test import MolTestDatasetWrapper
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

torch.multiprocessing.set_sharing_strategy('file_system')

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print(
        "Please install apex for mixed precision training from: https://github.com/NVIDIA/apex"
    )
    apex_support = False


def _save_config_file(model_checkpoints_folder, task_name):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy(
            f'MolCLR/config_finetune_{task_name}.yaml',
            os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class FineTune(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        # current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        current_time = self.config['output_dir']
        dir_name = current_time + '_' + config['task_name']
        self.dataset = dataset
        log_dir = os.path.join('MolCLR', 'finetune', dir_name,
                               f'seed_{self.dataset.seed}',
                               config['dataset']['target'],
                               f'fold_{self.dataset.fold}')
        self.writer = SummaryWriter(log_dir=log_dir)
        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in [
                    'qm7', 'qm8', 'qm9', 'PQC', 'APTC1', 'APTC2', 'MP'
            ]:
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter):
        # get the prediction
        __, pred = model(data)  # [N,C]

        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())
        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders(
        )

        self.normalizer = None
        if self.config["task_name"] in ['qm7', 'qm9']:
            labels = []
            for d, __ in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels.shape)

        elif self.config["task_name"] in ['PQC', 'APTC1', 'APTC2', 'MP']:
            '''Is normalize labels needed ?
            '''
            labels = []
            for d in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels)

        if self.config['model_type'] == 'gin':
            from models.ginet_finetune import GINet
            model = GINet(self.config['dataset']['task'],
                          **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_finetune import GCN
            model = GCN(self.config['dataset']['task'],
                        **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)

        layer_list = []
        for name, param in model.named_parameters():
            if 'pred_head' in name:
                print(name, param.requires_grad)
                layer_list.append(name)

        params = list(
            map(
                lambda x: x[1],
                list(
                    filter(lambda kv: kv[0] in layer_list,
                           model.named_parameters()))))
        base_params = list(
            map(
                lambda x: x[1],
                list(
                    filter(lambda kv: kv[0] not in layer_list,
                           model.named_parameters()))))

        optimizer = torch.optim.Adam([{
            'params': base_params,
            'lr': self.config['init_base_lr']
        }, {
            'params': params
        }],
                                     self.config['init_lr'],
                                     weight_decay=eval(
                                         self.config['weight_decay']))

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model,
                                              optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.writer.log_dir,
                                                'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder, self.config["task_name"])

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0

        # Initialize lists to store losses
        train_losses = []
        valid_losses = []

        for epoch_counter in range(self.config['epochs']):
            epoch_train_loss = 0.0
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                loss = self._step(model, data, n_iter)
                epoch_train_loss += loss.item()

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss',
                                           loss,
                                           global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # Calculate average training loss for the epoch
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification':
                    valid_loss, valid_cls = self._validate(model, valid_loader)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(
                            model.state_dict(),
                            os.path.join(model_checkpoints_folder,
                                         'model.pth'))
                elif self.config['dataset']['task'] == 'regression':
                    valid_loss, valid_rgr = self._validate(model, valid_loader)
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        torch.save(
                            model.state_dict(),
                            os.path.join(model_checkpoints_folder,
                                         'model.pth'))
                valid_losses.append(valid_loss)
                self.writer.add_scalar('validation_loss',
                                       valid_loss,
                                       global_step=valid_n_iter)
                valid_n_iter += 1
                print(
                    f"Epoch {epoch_counter}: Avg Train Loss = {avg_train_loss}, Valid Loss = {valid_loss}"
                )

        # Save losses to a DataFrame
        loss_df = pd.DataFrame({
            'Epoch': range(self.config['epochs']),
            'TrainLoss': train_losses,
            'ValidLoss': valid_losses
        })

        # Save DataFrame to CSV
        loss_df.to_csv(os.path.join(self.writer.log_dir, 'losses.csv'),
                       index=False)
        # save predictions
        train_pred_true_df = self._test(model, train_loader)
        test_pred_true_df = self._test(model, test_loader)
        train_pred_true_df.to_csv(os.path.join(self.writer.log_dir,
                                               'train_set_predictions.csv'),
                                  index=False)
        test_pred_true_df.to_csv(os.path.join(self.writer.log_dir,
                                              'test_set_predictions.csv'),
                                 index=False)

        train_pred = train_pred_true_df['predicted']
        train_true = train_pred_true_df['actual']
        test_pred = test_pred_true_df['predicted']
        test_true = test_pred_true_df['actual']

        # 結果をデータフレームに追加
        metrics_df = pd.DataFrame(
            {
                'Seed': self.dataset.seed,
                'Fold': self.dataset.fold,
                'R2 Train': r2_score(train_true, train_pred),
                'R2 Test': r2_score(test_true, test_pred),
                'MSE Train': mean_squared_error(train_true, train_pred),
                'MSE Test': mean_squared_error(test_true, test_pred),
                'MAE Train': mean_absolute_error(train_true, train_pred),
                'MAE Test': mean_absolute_error(test_true, test_pred),
            },
            index=[0])

        return metrics_df

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('MolCLR/ckpt',
                                              self.config['fine_tune_from'],
                                              'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder,
                                                 'model.pth'),
                                    map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data

        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                print('Validation loss:', valid_loss, 'MAE:', mae)
                return valid_loss, mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print('Validation loss:', valid_loss, 'RMSE:', rmse)
                return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:, 1])
            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints',
                                  'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data

        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)

            pred_true_df = pd.DataFrame({
                'predicted': predictions.flatten(),
                'actual': labels.flatten()
            })

            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                self.mae = mean_absolute_error(labels, predictions)
                print('Test loss:', test_loss, 'Test MAE:', self.mae)
            else:
                self.rmse = mean_squared_error(labels,
                                               predictions,
                                               squared=False)
                print('Test loss:', test_loss, 'Test RMSE:', self.rmse)

            return pred_true_df

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            self.roc_auc = roc_auc_score(labels, predictions[:, 1])
            print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)


def main(cid_train_path, cid_test_path, seed, fold, test_mode, config):
    dataset = MolTestDatasetWrapper(cid_train_path, cid_test_path, seed, fold,
                                    test_mode, config['batch_size'],
                                    **config['dataset'])

    fine_tune = FineTune(dataset, config)
    metrics_df = fine_tune.train()

    if config['dataset']['task'] == 'classification':
        return fine_tune.roc_auc
    if config['dataset']['task'] == 'regression':
        if config['task_name'] in ['qm7', 'qm8', 'qm9']:
            return fine_tune.mae
        elif config['task_name'] in ['PQC', 'APTC1', 'APTC2', 'MP']:
            return metrics_df
        else:
            return fine_tune.rmse


if __name__ == "__main__":

    TEST_MODE = False
    DATASET = 'PQC'

    if len(sys.argv) == 2:
        if int(sys.argv[1]) == 11:
            seeds = [32]
            DATASET = 'PQC'
            target_list = ['dipoleMoment', 'homo', 'gap']
        elif int(sys.argv[1]) == 12:
            seeds = [32]
            DATASET = 'PQC'
            target_list = ['lumo', 'energy', 'enthalpy']
        elif int(sys.argv[1]) == 21:
            seeds = [42]
            DATASET = 'PQC'
            target_list = ['dipoleMoment', 'homo', 'gap']
        elif int(sys.argv[1]) == 22:
            seeds = [42]
            DATASET = 'PQC'
            target_list = ['lumo', 'energy', 'enthalpy']
        elif int(sys.argv[1]) == 31:
            seeds = [52]
            DATASET = 'PQC'
            # target_list = ['dipoleMoment', 'homo', 'gap']
            target_list = ['homo', 'gap']
        elif int(sys.argv[1]) == 32:
            seeds = [52]
            DATASET = 'PQC'
            target_list = ['lumo', 'energy', 'enthalpy']

        elif int(sys.argv[1]) == 4:
            DATASET = 'APTC1'
        elif int(sys.argv[1]) == 5:
            DATASET = 'APTC2'
        elif int(sys.argv[1]) == 6:
            DATASET = 'MP'
            seeds = [12, 22]
        elif int(sys.argv[1]) == 7:
            DATASET = 'MP'
            seeds = [32, 42, 52]

    if DATASET == 'PQC':
        CONFIG_PATH = 'MolCLR/config_finetune_PQC.yaml'
        # seeds = [32, 42, 52]
        folds = 5
    elif DATASET == 'APTC1':
        CONFIG_PATH = 'MolCLR/config_finetune_APTC1.yaml'
        # seeds = [12, 22, 32, 42, 52]
        folds = 5
    elif DATASET == 'APTC2':
        CONFIG_PATH = 'MolCLR/config_finetune_APTC2.yaml'
        # seeds = [0]  # seed is none cause LOOCV
        folds = 40
    elif DATASET == 'MP':
        CONFIG_PATH = 'MolCLR/config_finetune_MP.yaml'
        # seeds = [12, 22, 32, 42, 52]
        folds = 5

    if TEST_MODE:
        seeds = [42]
        folds = 2

    config = yaml.load(open(CONFIG_PATH, "r"), Loader=yaml.FullLoader)
    '''
    if config['task_name'] == 'BBBP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bbbp/BBBP.csv'
        target_list = ["p_np"]

    elif config['task_name'] == 'Tox21':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/tox21/tox21.csv'
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
            "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE",
            "SR-MMP", "SR-p53"
        ]

    elif config['task_name'] == 'ClinTox':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/clintox/clintox.csv'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif config['task_name'] == 'HIV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/hiv/HIV.csv'
        target_list = ["HIV_active"]

    elif config['task_name'] == 'BACE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bace/bace.csv'
        target_list = ["Class"]

    elif config['task_name'] == 'SIDER':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/sider/sider.csv'
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders",
            "Product issues", "Eye disorders", "Investigations",
            "Musculoskeletal and connective tissue disorders",
            "Gastrointestinal disorders", "Social circumstances",
            "Immune system disorders",
            "Reproductive system and breast disorders",
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "General disorders and administration site conditions",
            "Endocrine disorders", "Surgical and medical procedures",
            "Vascular disorders", "Blood and lymphatic system disorders",
            "Skin and subcutaneous tissue disorders",
            "Congenital, familial and genetic disorders",
            "Infections and infestations",
            "Respiratory, thoracic and mediastinal disorders",
            "Psychiatric disorders", "Renal and urinary disorders",
            "Pregnancy, puerperium and perinatal conditions",
            "Ear and labyrinth disorders", "Cardiac disorders",
            "Nervous system disorders",
            "Injury, poisoning and procedural complications"
        ]

    elif config['task_name'] == 'MUV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/muv/muv.csv'
        target_list = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548',
            'MUV-852', 'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858',
            'MUV-713', 'MUV-733', 'MUV-652', 'MUV-466', 'MUV-832'
        ]

    elif config['task_name'] == 'FreeSolv':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/freesolv/freesolv.csv'
        target_list = ["expt"]

    elif config["task_name"] == 'ESOL':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/esol/esol.csv'
        target_list = ["measured log solubility in mols per litre"]

    elif config["task_name"] == 'Lipo':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/lipophilicity/Lipophilicity.csv'
        target_list = ["exp"]

    elif config["task_name"] == 'qm7':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm7/qm7.csv'
        target_list = ["u0_atom"]

    elif config["task_name"] == 'qm8':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm8/qm8.csv'
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0",
            "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM"
        ]

    elif config["task_name"] == 'qm9':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm9/qm9.csv'
        target_list = [
            'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv'
        ]
    '''

    if config["task_name"] == 'PQC':
        config['dataset']['task'] = 'regression'
        config['dataset'][
            'data_path'] = '2d3d/dataset/PubChemQC-PM6/step17/molclr_input/molclr_input_data.csv'

    elif config["task_name"] == 'APTC1':
        config['dataset']['task'] = 'regression'
        config['dataset'][
            'data_path'] = '2d3d/dataset/APTC/step8/molclr_input/aptc1/molclr_input_data.csv'
        target_list = ['ACTIVITY']

    elif config["task_name"] == 'APTC2':
        config['dataset']['task'] = 'regression'
        config['dataset'][
            'data_path'] = '2d3d/dataset/APTC/step8/molclr_input/aptc2/molclr_input_data.csv'
        target_list = ['ACTIVITY']

    elif config["task_name"] == 'MP':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = '2d3d/dataset/MP/step2/mp_195.csv'
        target_list = ['mpC']

    else:
        raise ValueError('Undefined downstream task!')

    if TEST_MODE:
        config['output_dir'] = config['output_dir'] + '_test'
        config['epochs'] = 1
    print(config)

    for seed in seeds:
        for target in target_list:
            results_df = pd.DataFrame(columns=[
                'Seed', 'Fold', 'R2 Train', 'R2 Test', 'MSE Train', 'MSE Test',
                'MAE Train', 'MAE Test'
            ])
            for fold in range(folds):
                config['dataset']['target'] = target
                if DATASET == 'PQC':
                    cid_train_path = f'2d3d/dataset/PubChemQC-PM6/step17/index/seed_{seed}/train_{fold}.npy'
                    cid_test_path = f'2d3d/dataset/PubChemQC-PM6/step17/index/seed_{seed}/test_{fold}.npy'
                elif DATASET == 'APTC1':
                    cid_train_path = f'2d3d/dataset/APTC/step7/index/aptc1/seed_{seed}/train_{fold}.npy'
                    cid_test_path = f'2d3d/dataset/APTC/step7/index/aptc1/seed_{seed}/test_{fold}.npy'
                elif DATASET == 'APTC2':
                    cid_train_path = f'2d3d/dataset/APTC/step7/index/aptc2/train_{fold}.npy'
                    cid_test_path = f'2d3d/dataset/APTC/step7/index/aptc2/test_{fold}.npy'
                elif DATASET == 'MP':
                    cid_train_path = f'2d3d/dataset/MP/step2/index/seed_{seed}/train_{fold}.npy'
                    cid_test_path = f'2d3d/dataset/MP/step2/index/seed_{seed}/test_{fold}.npy'

                df_append = main(cid_train_path, cid_test_path, seed, fold,
                                 TEST_MODE, config)

                results_df = pd.concat([results_df, df_append],
                                       axis=0,
                                       ignore_index=True)

            save_dir = os.path.join(
                'MolCLR', 'finetune',
                config['output_dir'] + '_' + config['task_name'],
                f'seed_{seed}', target)
            results_df.to_csv(os.path.join(save_dir, 'results.csv'),
                              index=False)

            mean_std_df = pd.DataFrame({
                f'Mean_{seed}_{target}':
                results_df.mean(),
                f'Std_{seed}_{target}':
                results_df.std()
            }).T
            mean_std_df.to_csv(os.path.join(save_dir, 'mean_std.csv'),
                               index=True)
