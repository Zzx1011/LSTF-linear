from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if self.args.group_training:
            # 返回每组的 data_loader
            data_sets, data_loaders = data_provider(self.args, flag, return_group=True)
            return data_sets, data_loaders
        else:
            data_set, data_loader = data_provider(self.args, flag)
            return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data_list, train_loader_list = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data_list, vali_loader_list = self._get_data(flag='val')
            test_data_list, test_loader_list = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader_list[0])
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for group_idx, train_loader in enumerate(train_loader_list):
            print(f"Training group {group_idx}...")
            path = os.path.join(self.args.checkpoints, setting, f"group_{group_idx}")
            if not os.path.exists(path):
                os.makedirs(path)

            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []
                self.model.train()
                epoch_time = time.time()

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    print("batch_x shape:", batch_x.shape)
                    print("batch_y shape:", batch_y.shape)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                            loss = criterion(outputs, batch_y)
                    else:
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        # # reshape back to [B, pred_len, C]
                        # CB, pred_len, _ = outputs.shape
                        # C = 3 if group_idx == 0 else 2
                        # B = CB // C
                        # outputs = outputs.reshape(C, B, pred_len).permute(1, 2, 0)  # -> [B, pred_len, C]
                        # batch_y = batch_y.reshape(C, B, pred_len).permute(1, 2, 0)


                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)

                    train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

                train_loss = np.average(train_loss)

                if not self.args.train_only:
                    vali_loss = self.vali(vali_data_list[group_idx], vali_loader_list[group_idx], criterion)
                    test_loss = self.vali(test_data_list[group_idx], test_loader_list[group_idx], criterion)

                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                    early_stopping(vali_loss, self.model, path)
                else:
                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                        epoch + 1, train_steps, train_loss))
                    early_stopping(train_loss, self.model, path)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    def visual_test(self, setting):
        test_data_list, test_loader_list = self._get_data(flag='test')

        folder_path = f'./feature_visualization/{setting}/'
        os.makedirs(folder_path, exist_ok=True)

        for group_idx, test_loader in enumerate(test_loader_list):
            all_x = []

            for batch_x, _, _, _ in test_loader:
                batch_x = batch_x.cpu().numpy()  # (B, input_len, C)
                all_x.append(batch_x)

            all_x = np.concatenate(all_x, axis=0)  # (total_samples, input_len, C)
            input_len = all_x.shape[1]
            num_features = all_x.shape[2]



            plt.figure(figsize=(16, 6))
            for feature_idx in range(num_features):
                plt.plot(all_x[feature_idx], label=f'Feature {feature_idx}', linewidth=1)

            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.title(f'Input Sequence Visualization - Group {group_idx}')
            plt.legend(loc='upper right', fontsize='small', ncol=2)
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f'group{group_idx}_input_only.png'))
            plt.close()

        
    def test(self, setting, test=0):
        test_data_list, test_loader_list = self._get_data(flag='test')

        preds_group = []
        trues_group = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for group_idx, test_loader in enumerate(test_loader_list):
            # === 加载每组自己的 best checkpoint
            if test:
                print(f'Loading model for group {group_idx}')
                group_ckpt_path = os.path.join('./checkpoints/', setting, f'group_{group_idx}', 'checkpoint.pth')
                self.model.load_state_dict(torch.load(group_ckpt_path))

            preds = []
            trues = []
            inputx = []

            self.model.eval()

            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                    pred = outputs.detach().cpu().numpy()
                    true = batch_y.detach().cpu().numpy()

                    preds.append(pred)
                    trues.append(true)

                    inputx.append(batch_x[0].detach().cpu().numpy())

                    if i % 20 == 0:
                        input_sample = batch_x[0].detach().cpu().numpy()
                        gt = np.concatenate((input_sample[:, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input_sample[:, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, f"group{group_idx}_{i}.pdf"))

            # 拼接 batch 维度 (axis=0)，不要拼特征！！
            preds = np.concatenate(preds, axis=0)  # (batch_size, pred_len, group_feature_dim)
            trues = np.concatenate(trues, axis=0)

            preds_group.append(preds)  # 保存每组结果
            trues_group.append(trues)

        # === 所有 group 拼回完整的特征 ===
        preds = np.concatenate(preds_group, axis=-1)  # 在特征维度拼接
        trues = np.concatenate(trues_group, axis=-1)

        print('test shape:', preds.shape, trues.shape)
        # (batch_size, pred_len, total_feature_dim)

        # 选择第一个样本进行可视化
        sample_idx = 0
        pred_sample = preds[sample_idx]  # shape: (pred_len, total_feature_dim)
        true_sample = trues[sample_idx]  # shape: (pred_len, total_feature_dim)

        plt.figure(figsize=(12, 6))
        for i in range(pred_sample.shape[1]):  # 遍历所有特征
            plt.plot(true_sample[:, i], label=f'True F{i}', linestyle='--')
            plt.plot(pred_sample[:, i], label=f'Pred F{i}')

        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.title(f'Prediction vs True (Sample {sample_idx})')
        plt.legend(ncol=2, fontsize='small')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, f'pred_vs_true_sample{sample_idx}.png'))
        plt.show()

        # 计算预测值和真实值的相关系数
        preds_feature1 = preds[:, :, 1].flatten()  
        preds_feature3 = preds[:, :, 3].flatten()
        trues_feature1 = trues[:, :, 1].flatten()
        trues_feature3 = trues[:, :, 3].flatten()

        # 计算预测值中 feature1 和 feature3 的相关系数
        corr_preds = np.corrcoef(preds_feature1, preds_feature3)[0, 1]

        # 计算真实值中 feature1 和 feature3 的相关系数
        corr_trues = np.corrcoef(trues_feature1, trues_feature3)[0, 1]

        print(f"Correlation in PREDICTIONS (feature1 vs feature3): {corr_preds:.4f}")
        print(f"Correlation in GROUND TRUTH (feature1 vs feature3): {corr_trues:.4f}")

        # 保存总的测试结果
        result_folder = './results/' + setting + '/'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # 单独计算每个特征的 MSE
        mse_per_feature = []
        for i in range(preds.shape[-1]):
            mse_i = np.mean((preds[..., i] - trues[..., i]) ** 2)
            mse_per_feature.append(mse_i)
            print(f"Feature {i} MSE: {mse_i:.6f}")

        # 也算一下整体的指标
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('Overall mse:{}, mae:{}'.format(mse, mae))

        # 保存结果
        with open(os.path.join(result_folder, "result.txt"), 'a') as f:
            f.write(setting + "\n")
            f.write('Overall mse:{}, mae:{}, rse:{}, corr:{}\n'.format(mse, mae, rse, corr))
            for i, mse_i in enumerate(mse_per_feature):
                f.write(f"Feature {i} mse: {mse_i:.6f}\n")
            f.write('\n')

        np.save(os.path.join(result_folder, 'pred.npy'), preds)
        np.save(os.path.join(result_folder, 'true.npy'), trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return
