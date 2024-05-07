import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle


class dataSegLoader(object):
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1  - self.horizon
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1 - self.horizon

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.double(self.train[index:index + self.win_size]), np.double(self.train[index + self.win_size:index + self.win_size + self.horizon])
        elif (self.mode == 'test'):
            return np.double(self.test[index:index + self.win_size]), np.double(self.test[index + self.win_size:index + self.win_size + self.horizon])

class SMDSegLoader(dataSegLoader):
    def __init__(self, data_path, win_size, step, mode="train", horizon=1):
        self.data_path = data_path
        self.horizon = horizon
        self.mode = mode
        self.step = step
        self.win_size = win_size
        index = 200000
        length = 50000
        # index = 230000
        # length = 20000
        data = np.load(self.data_path + "/SMD_train.npy")[index:index + length]
        test_data = np.load(self.data_path + "/SMD_test.npy")[index:index + length]

        self.test = test_data
        self.train = data

        self.test_labels = np.load(self.data_path + "/SMD_test_label.npy")[index:index + length].astype(int)
        print(self.train.shape,self.test.shape,self.test_labels.shape)
        
        
class SWaTSegLoader(dataSegLoader):
    def __init__(self, data_path, win_size, step, mode="train", horizon=1):
        self.data_path = data_path
        self.horizon = horizon
        self.mode = mode
        self.step = step
        self.win_size = win_size
        index = 370000
        length = 20000
        data = np.load(data_path + "/swat_x_train_de5.npy")[index:index + length]
        test_data = np.load(data_path + "/swat_x_test_de5.npy")[index:index + length]
        self.test = test_data
        self.train = data
        self.test_labels = np.load(data_path + "/swat_y_test_de5.npy")[index:index + length].astype(int)
        print(self.train.shape,self.test.shape,self.test_labels.shape)
  
class MSLSegLoader(dataSegLoader):
    def __init__(self, data_path, win_size, step, mode="train", horizon=1):
        self.data_path = data_path
        self.horizon = horizon
        self.mode = mode
        self.step = step
        self.win_size = win_size
        index = 10000*3
        length = 20000
        # index = 0 #用于收敛性实验
        # length = 5000 #用于收敛性实验
        self.scaler = MinMaxScaler()
        data = np.load(data_path + "/MSL_train.npy")[index:index + length]
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = np.load(data_path + "/MSL_test.npy")[index:index + length]
        self.test = test_data
        self.scaler.fit(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")[index:index + length].astype(int)
        print(self.train.shape,self.test.shape,self.test_labels.shape)


class SMAPSegLoader(dataSegLoader):
    def __init__(self, data_path, win_size, step, mode="train", horizon=1):
        self.data_path = data_path
        self.horizon = horizon
        self.mode = mode
        self.step = step
        self.win_size = win_size
        index = 10000
        length = 20000
        # length = 50000 #用于收敛性实验
        data = np.load(data_path + "/SMAP_train.npy")[index:index + length]

        test_data = np.load(data_path + "/SMAP_test.npy")[index:index + length]
        self.test = test_data
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")[index:index + length].astype(int)
        print(self.train.shape,self.test.shape,self.test_labels.shape)

class PSMSegLoader(dataSegLoader):
    def __init__(self, data_path, win_size, step, mode="train", horizon=1):
        self.data_path = data_path
        self.horizon = horizon
        self.mode = mode
        self.step = step
        self.win_size = win_size
        # index = 0
        # length = 20000
        index = 40000
        length = 20000
        # index = 0 #用于收敛性实验
        # length = 50000 #用于收敛性实验
        data = pd.read_csv(data_path + '/PSM_train.csv')[index:index + length]
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        test_data = pd.read_csv(data_path + '/PSM_test.csv')[index:index + length]
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = test_data
        self.train = data
        self.test_labels = pd.read_csv(data_path + '/PSM_test_label.csv')[index:index + length].values[:, 1:].reshape(-1).astype(int)
        print(self.train.shape,self.test.shape,self.test_labels.shape)


def get_loader_segment(data_path, batch_size, win_size=6, step=1, mode='train', dataset='SMD', horizon=1):
    dataset = eval(dataset+'SegLoader')(data_path, win_size, step, mode, horizon=horizon)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)
    return dataset, data_loader

