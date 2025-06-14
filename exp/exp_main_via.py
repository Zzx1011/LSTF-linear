import re
import sys
sys.path.append('/home/zzx/projects/rrg-timsbc/zzx/LTSF-Linear')
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
import copy
warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.feature_groups = [[0, 2, 6], [1, 3], [4, 5]] 
        self.models = self._build_models()
        self.model_group1 = self._build_model()
        self.model_group2 = self._build_model()
        self.model_group3 = self._build_model()

    def split_features(self, x):
        """
        x: [B, T, C=7]
        return: x_group1, x_group2, x_group3 
        """
        group1 = x[:, :, [0, 2, 6]]  # feature0,2,6
        group2 = x[:, :, [1, 3]]     # feature1,3
        group3 = x[:, :, [4, 5]]     # feature4,5
        return group1, group2, group3

    def _build_models(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
        }
        models = []
        for indices in self.feature_groups:
            args_copy = copy.deepcopy(self.args)
            args_copy.enc_in = len(indices)  
            model = model_dict[self.args.model].Model(args_copy).float()
            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
            model.to(self.device)
            models.append(model)
        return models
    
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
        model.to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # params = []
        # for model in self.models:
        #     params += list(model.parameters())
        params = list(self.model_group1.parameters()) + \
             list(self.model_group2.parameters()) + \
             list(self.model_group3.parameters())
        model_optim = optim.Adam(params, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _process_input(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        outputs = []
        for i, indices in enumerate(self.feature_groups):
            group_x = batch_x[:, :, indices]
            if 'Linear' in self.args.model:
                out = self.models[i](group_x)
            else:
                if self.args.output_attention:
                    out = self.models[i](group_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    out = self.models[i](group_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs.append(out)
        combined = torch.stack(outputs, dim=-1)
        return self.fusion(combined)

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

    def train_group(self, model, train_loader, criterion, optimizer, feature_idx, epoch, device):
        model.train()

        for i, (batch_x, batch_y, _) in enumerate(train_loader):
            batch_x = batch_x[:, :, feature_idx].to(device)  # (B, L, C_group)
            batch_y = batch_y[:, :, feature_idx].to(device)  # (B, P, C_group)

            B, L, C_group = batch_x.shape
            P = batch_y.shape[1]

            # Reshape: (B, L, C_group) → (B * C_group, L, 1)
            batch_x = batch_x.permute(0, 2, 1).reshape(B * C_group, L, 1)
            batch_y = batch_y.permute(0, 2, 1).reshape(B * C_group, P, 1)

            # Forward
            output = model(batch_x)  # (B * C_group, P, 1)

            # Compute loss
            loss = criterion(output, batch_y)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")

    def train(self, setting):
        
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping_g1 = EarlyStopping(patience=self.args.patience, verbose=True)
        early_stopping_g2 = EarlyStopping(patience=self.args.patience, verbose=True)
        early_stopping_g3 = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim_g1 = self._select_optimizer()
        model_optim_g2 = self._select_optimizer()
        model_optim_g3 = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_g1 = []
            train_loss_g2 = []
            train_loss_g3 = []

            # self.model.train()
            self.model_group1.train()
            self.model_group2.train()
            self.model_group3.train()

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim_g1.zero_grad()
                model_optim_g2.zero_grad()
                model_optim_g3.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                x_g1, x_g2, x_g3 = self.split_features(batch_x)
                y_g1, y_g2, y_g3 = self.split_features(batch_y)
                dec_inp_g1, dec_inp_g2, dec_inp_g3 = self.split_features(dec_inp)

                def reshape_input(x):  # [B, T, C] -> [C*B, T, 1]
                    B, T, C = x.shape
                    return x.permute(2, 0, 1).reshape(C * B, T, 1)

                def reshape_output(y, C):  # [C*B, T, 1] -> [B, T, C]
                    y = y.reshape(C, -1, y.shape[1], 1)  # [C, B, T, 1]
                    return y.permute(1, 2, 0, 3).squeeze(-1)  # [B, T, C]
                
                x_g1_r, x_g2_r, x_g3_r = reshape_input(x_g1).to(self.device), reshape_input(x_g2).to(self.device), reshape_input(x_g3).to(self.device)
                y_g1_r, y_g2_r, y_g3_r = reshape_input(y_g1).to(self.device), reshape_input(y_g2).to(self.device), reshape_input(y_g3).to(self.device)
                dec_g1_r, dec_g2_r, dec_g3_r = reshape_input(dec_inp_g1), reshape_input(dec_inp_g2), reshape_input(dec_inp_g3)
                
                x_g1_r = x_g1_r.to(self.device)
                x_g2_r = x_g2_r.to(self.device)
                x_g3_r = x_g3_r.to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            # outputs = self.model(batch_x)
                            out_g1 = self.model_group1(x_g1_r)
                            out_g2 = self.model_group2(x_g2_r)
                            out_g3 = self.model_group3(x_g3_r)
                        else:
                            if self.args.output_attention:
                                # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                out_g1 = self.model_group1(x_g1_r, batch_x_mark.repeat_interleave(x_g1_r.shape[-1], 0), dec_inp_g1, batch_y_mark.repeat_interleave(x_g1.shape[-1], 0))[0]
                                out_g2 = self.model_group2(x_g2_r, batch_x_mark.repeat_interleave(x_g2_r.shape[-1], 0), dec_inp_g2, batch_y_mark.repeat_interleave(x_g2.shape[-1], 0))[0]
                                out_g3 = self.model_group3(x_g3_r, batch_x_mark.repeat_interleave(x_g3_r.shape[-1], 0), dec_inp_g3, batch_y_mark.repeat_interleave(x_g3.shape[-1], 0))[0]
                            else:
                                # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                out_g1 = self.model_group1(x_g1_r, batch_x_mark.repeat_interleave(x_g1_r.shape[-1], 0), dec_inp_g1, batch_y_mark.repeat_interleave(x_g1.shape[-1], 0))
                                out_g2 = self.model_group2(x_g2_r, batch_x_mark.repeat_interleave(x_g2_r.shape[-1], 0), dec_inp_g2, batch_y_mark.repeat_interleave(x_g2.shape[-1], 0))
                                out_g3 = self.model_group3(x_g3_r, batch_x_mark.repeat_interleave(x_g3_r.shape[-1], 0), dec_inp_g3, batch_y_mark.repeat_interleave(x_g3.shape[-1], 0))
                            
                else:
                    if 'Linear' in self.args.model:
                            # outputs = self.model(batch_x)
                            
                            out_g1 = self.model_group1(x_g1_r)
                            out_g2 = self.model_group2(x_g2_r)
                            out_g3 = self.model_group3(x_g3_r)
                    else:
                        if self.args.output_attention:
                            # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            out_g1 = self.model_group1(x_g1, batch_x_mark.repeat_interleave(x_g1.shape[-1], 0), dec_inp_g1, batch_y_mark.repeat_interleave(x_g1.shape[-1], 0))[0]
                            out_g2 = self.model_group2(x_g2, batch_x_mark.repeat_interleave(x_g2.shape[-1], 0), dec_inp_g2, batch_y_mark.repeat_interleave(x_g2.shape[-1], 0))[0]
                            out_g3 = self.model_group3(x_g3, batch_x_mark.repeat_interleave(x_g3.shape[-1], 0), dec_inp_g3, batch_y_mark.repeat_interleave(x_g3.shape[-1], 0))[0]
                            
                        else:
                            # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                            out_g1 = self.model_group1(x_g1, batch_x_mark.repeat_interleave(x_g1.shape[-1], 0), dec_inp_g1, batch_y_mark.repeat_interleave(x_g1.shape[-1], 0))
                            out_g2 = self.model_group2(x_g2, batch_x_mark.repeat_interleave(x_g2.shape[-1], 0), dec_inp_g2, batch_y_mark.repeat_interleave(x_g2.shape[-1], 0))
                            out_g3 = self.model_group3(x_g3, batch_x_mark.repeat_interleave(x_g3.shape[-1], 0), dec_inp_g3, batch_y_mark.repeat_interleave(x_g3.shape[-1], 0))


                out_g1 = reshape_output(out_g1, x_g1.shape[-1])
                out_g2 = reshape_output(out_g2, x_g2.shape[-1])
                out_g3 = reshape_output(out_g3, x_g3.shape[-1])
                y_g1_r = reshape_output(y_g1_r, x_g1.shape[-1])
                y_g2_r = reshape_output(y_g2_r, x_g2.shape[-1])
                y_g3_r = reshape_output(y_g3_r, x_g3.shape[-1])

                # print(outputs.shape,batch_y.shape)
                f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # loss = criterion(outputs, batch_y)
                # train_loss.append(loss.item())
                out_g1 = out_g1[:, -self.args.pred_len:, f_dim:]
                out_g2 = out_g2[:, -self.args.pred_len:, f_dim:]
                out_g3 = out_g3[:, -self.args.pred_len:, f_dim:]
                y_g1_r = y_g1_r[:, -self.args.pred_len:, f_dim:]
                y_g2_r = y_g2_r[:, -self.args.pred_len:, f_dim:]
                y_g3_r = y_g3_r[:, -self.args.pred_len:, f_dim:]
                loss_g1 = criterion(out_g1, y_g1_r)
                loss_g2 = criterion(out_g2, y_g2_r)
                loss_g3 = criterion(out_g3, y_g3_r)
                train_loss_g1.append(loss_g1.item())
                train_loss_g2.append(loss_g2.item())
                train_loss_g3.append(loss_g3.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss_g1: {2:.7f}".format(i + 1, epoch + 1, loss_g1.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    # iter_count = 0
                    time_now = time.time()
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss_g2: {2:.7f}".format(i + 1, epoch + 1, loss_g2.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    # iter_count = 0
                    time_now = time.time()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss_g3: {2:.7f}".format(i + 1, epoch + 1, loss_g3.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    
                if self.args.use_amp:
                    scaler.scale(loss_g1).backward()
                    scaler.step(model_optim_g1)
                    scaler.update()
                    scaler.scale(loss_g2).backward()
                    scaler.step(model_optim_g2)
                    scaler.update()
                    scaler.scale(loss_g3).backward()
                    scaler.step(model_optim_g3)
                    scaler.update()
                else:
                    loss_g1.backward()
                    loss_g2.backward()
                    loss_g3.backward()
                    model_optim_g1.step()
                    model_optim_g2.step()
                    model_optim_g3.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            def report_loss(name, loss, model, early_stopping, path):
                if not self.args.train_only:
                    vali_loss = self.vali(vali_data, vali_loader, criterion)
                    test_loss = self.vali(test_data, test_loader, criterion)
                    print(f"Epoch: {epoch+1}, Steps: {train_steps} | {name} Train Loss: {loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
                    early_stopping(vali_loss, model, path)
                else:
                    print(f"Epoch: {epoch+1}, Steps: {train_steps} | {name} Train Loss: {loss:.7f}")
                    early_stopping(loss, model, path)
                if early_stopping.early_stop:
                    print(f"Early stopping for {name}")
            
            if not os.path.exists(path+ "_g1"):
                os.makedirs(path+ "_g1")
            if not os.path.exists(path+ "_g2"):
                os.makedirs(path+ "_g2")
            if not os.path.exists(path+ "_g3"):
                os.makedirs(path+ "_g3")

            train_loss_g1 = np.average(train_loss_g1)
            report_loss("Group1", train_loss_g1, self.model_group1, early_stopping_g1, path + "_g1")

            train_loss_g2 = np.average(train_loss_g2)
            report_loss("Group2", train_loss_g2, self.model_group2, early_stopping_g2, path + "_g2")
                
            train_loss_g3 = np.average(train_loss_g3)
            report_loss("Group3", train_loss_g3, self.model_group3, early_stopping_g3, path + "_g3")
                

            adjust_learning_rate(model_optim_g1, epoch + 1, self.args)
            adjust_learning_rate(model_optim_g2, epoch + 1, self.args)
            adjust_learning_rate(model_optim_g3, epoch + 1, self.args)

        best_model_path_g1 = path + '/' + 'checkpoint.pth'
        self.model_group1.load_state_dict(torch.load(best_model_path_g1))
        best_model_path_g2 = path + '/' + 'checkpoint.pth'
        self.model_group2.load_state_dict(torch.load(best_model_path_g2))
        best_model_path_g3 = path + '/' + 'checkpoint.pth'
        self.model_group3.load_state_dict(torch.load(best_model_path_g3))

        return self.model_group1, self.model_group2, self.model_group3

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        self.model_group1.eval()
        self.model_group2.eval()
        self.model_group3.eval()
        if test:
            print('loading model')
            self.model_group1.load_state_dict(torch.load(os.path.join('./checkpoints/checkpoints/' + setting + '_g1', 'checkpoint.pth')))
            self.model_group2.load_state_dict(torch.load(os.path.join('./checkpoints/checkpoints/' + setting + '_g2', 'checkpoint.pth')))
            self.model_group3.load_state_dict(torch.load(os.path.join('./checkpoints/checkpoints/' + setting + '_g3', 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                x_g1, x_g2, x_g3 = self.split_features(batch_x)
                y_g1, y_g2, y_g3 = self.split_features(batch_y)
                dec_inp_g1, dec_inp_g2, dec_inp_g3 = self.split_features(dec_inp)

                def reshape_input(x):  # [B, T, C] -> [C*B, T, 1]
                    B, T, C = x.shape
                    return x.permute(2, 0, 1).reshape(C * B, T, 1)

                def reshape_output(y, C):  # [C*B, T, 1] -> [B, T, C]
                    y = y.reshape(C, -1, y.shape[1], 1)  # [C, B, T, 1]
                    return y.permute(1, 2, 0, 3).squeeze(-1)  # [B, T, C]
                
                x_g1_r, x_g2_r, x_g3_r = reshape_input(x_g1).to(self.device), reshape_input(x_g2).to(self.device), reshape_input(x_g3).to(self.device)
                y_g1_r, y_g2_r, y_g3_r = reshape_input(y_g1).to(self.device), reshape_input(y_g2).to(self.device), reshape_input(y_g3).to(self.device)
                dec_g1_r, dec_g2_r, dec_g3_r = reshape_input(dec_inp_g1), reshape_input(dec_inp_g2), reshape_input(dec_inp_g3)
                
                x_g1_r = x_g1_r.to(self.device)
                x_g2_r = x_g2_r.to(self.device)
                x_g3_r = x_g3_r.to(self.device)

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
                            # outputs = self.model(batch_x)
                            out_g1 = self.model_group1(x_g1_r)
                            out_g2 = self.model_group2(x_g2_r)
                            out_g3 = self.model_group3(x_g3_r)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                out_g1 = reshape_output(out_g1, x_g1.shape[-1])
                out_g2 = reshape_output(out_g2, x_g2.shape[-1])
                out_g3 = reshape_output(out_g3, x_g3.shape[-1])
                y_g1_r = reshape_output(y_g1_r, x_g1.shape[-1])
                y_g2_r = reshape_output(y_g2_r, x_g2.shape[-1])
                y_g3_r = reshape_output(y_g3_r, x_g3.shape[-1])

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # outputs = outputs.detach().cpu().numpy()
                # batch_y = batch_y.detach().cpu().numpy()
                out_g1 = out_g1[:, -self.args.pred_len:, f_dim:]
                out_g2 = out_g2[:, -self.args.pred_len:, f_dim:]
                out_g3 = out_g3[:, -self.args.pred_len:, f_dim:]
                y_g1_r = y_g1_r[:, -self.args.pred_len:, f_dim:]
                y_g2_r = y_g2_r[:, -self.args.pred_len:, f_dim:]
                y_g3_r = y_g3_r[:, -self.args.pred_len:, f_dim:]

                # pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                # true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                pred = torch.cat([out_g1, out_g2, out_g3], dim=-1)
                true = torch.cat([y_g1_r, y_g2_r, y_g3_r], dim=-1)
                pred = pred.detach().cpu().numpy()  
                true = true.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='[Long-term Series Forecasting] via LTSF-Linear')
    # Add your arguments here
    # Example:
    parser.add_argument('--model', type=str, default='Autoformer', help='model name')
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset name')
    parser.add_argument('--features', type=str, default='M', help='M for multivariate data S for single')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    # Add more arguments as needed

    args = parser.parse_args()
    
    exp = Exp_Main(args)
    exp.train('experiment_setting')  # Replace with your actual setting