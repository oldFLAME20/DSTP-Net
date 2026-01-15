import torch
import os
import random
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class NIPS_TS_WaterSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Water_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Water_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/NIPS_TS_Water_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])


class NIPS_TS_SwanSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Swan_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Swan_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/NIPS_TS_Swan_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class NIPS_TS_CCardSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_creditcard_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_creditcard_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/NIPS_TS_creditcard_test_label.npy")
        # --- 关键修改：在这里计算并打印点级别的异常率 ---
        if self.mode in ['test']:  # 只在需要用到测试标签时计算
            total_points = len(self.test_labels)
            # 在WADI数据集中, -1 代表异常
            anomaly_points = np.sum(self.test_labels == -1)

            if total_points > 0:
                anomaly_ratio = (anomaly_points / total_points) * 100
                print(f"---  NIPS_TS_CCard Dataset Anomaly Analysis点级别的异常率 ---")
                print(f"Total test points: {total_points}")
                print(f"Anomaly points (label == -1): {anomaly_points}")
                print(f"Point-level anomaly rate: {anomaly_ratio:.4f}%")
                print(f"-------------------------------------")

                # 2. 计算非重叠窗口级别的异常率 (Non-overlapping Window-level Anomaly Rate)
                # 使用 win_size 作为非重叠窗口的步长
                non_overlap_step = self.win_size
                total_windows = 0
                anomalous_windows = 0

                for i in range(0, total_points, non_overlap_step):
                    # 获取当前窗口的标签
                    window_labels = self.test_labels[i: i + non_overlap_step]

                    if len(window_labels) < non_overlap_step:
                        continue

                    total_windows += 1

                    # 检查窗口内是否有异常点 (-1)
                    if np.any(window_labels == -1):
                        anomalous_windows += 1

                if total_windows > 0:
                    window_anomaly_ratio = (anomalous_windows / total_windows) * 100
                    print(f"--- NIPS_TS_CCard Non-Overlapping Window Analysis (win_size={self.win_size}) ---")
                    print(f"Total non-overlapping windows: {total_windows}")
                    print(f"Windows with at least one anomaly: {anomalous_windows}")
                    print(f"[ANSWER] Window-level anomaly rate: {window_anomaly_ratio:.4f}%")
                    print(f"----------------------------------------------------------")
                else:
                    print("Warning: Could not calculate window-level anomaly rate (total_windows is zero).")

            else:
                print("Warning: WADI test labels are empty or not loaded correctly.")
    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])


class SWaTSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/SWaT_Dataset_Normal_v1.csv', header=1)
        data = data.values[:, 1:-1]

        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = pd.read_csv(data_path + '/SWaT_Dataset_Attack_v0.csv')
        self.test_labels = test_data['Normal/Attack'].values.reshape(-1, 1)


        test_data = test_data.values[:, 1:-1]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)
        self.train = data

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])


class PSMSegLoader(Dataset):


    def __init__(self, data_path, win_size, step, mode="train"):
        """
               专门为PSM数据集的.npy文件结构定制的加载器。
               :param data_path: 数据集所在的根目录，例如 '.../data/PSM'
               :param win_size: 滑动窗口大小
               :param step: 滑动步长
               :param mode: "train" 或 "test"
               """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # --- 数据加载与预处理 ---

        # 1. 加载训练数据 (PSM_train.npy)
        train_path = os.path.join(data_path, 'PSM_train.npy')
        train_data = np.load(train_path)

        # 处理潜在的NaN值
        train_data = np.nan_to_num(train_data)

        # 2. 在训练数据上拟合(fit)标准化器
        self.scaler.fit(train_data)

        # 3. 标准化(transform)训练数据并赋值给 self.train
        self.train = self.scaler.transform(train_data)

        # 4. 加载测试数据 (PSM_test.npy)
        test_path = os.path.join(data_path, 'PSM_test.npy')
        test_data = np.load(test_path)

        # 处理潜在的NaN值
        test_data = np.nan_to_num(test_data)

        # 5. 使用在训练集上拟合的同一个标准化器来标准化测试数据
        self.test = self.scaler.transform(test_data)

        # 6. 加载测试标签 (PSM_test_label.npy)
        label_path = os.path.join(data_path, 'PSM_test_label.npy')
        self.test_labels = np.load(label_path)

        # 确保标签是二维的 (num_samples, 1)，以方便后续处理
        if self.test_labels.ndim == 1:
            self.test_labels = self.test_labels.reshape(-1, 1)

        # 7. 为训练数据创建全零标签 (假设训练集无异常)
        self.train_labels = np.zeros((self.train.shape[0], 1))

        print(f"--- PSM Npy Loader initialized for mode: '{self.mode}' ---")
        print(f"Train data shape: {self.train.shape}")
        print(f"Test data shape: {self.test.shape}")
        print(f"Test labels shape: {self.test_labels.shape}")


    #     self.mode = mode
    #     self.step = step
    #     self.win_size = win_size
    #     self.scaler = StandardScaler()
    #     data = pd.read_csv(data_path + '/train.csv')
    #     data = data.values[:, 1:]
    #
    #     data = np.nan_to_num(data)
    #
    #     self.scaler.fit(data)
    #     data = self.scaler.transform(data)
    #     test_data = pd.read_csv(data_path + '/test.csv')
    #
    #     test_data = test_data.values[:, 1:]
    #     test_data = np.nan_to_num(test_data)
    #
    #     self.test = self.scaler.transform(test_data)
    #
    #     self.train = data
    #
    #     self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]
    #
    #     print("test:", self.test.shape)
    #     print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class MSLSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class SMAPSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class SMDSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
class WADISegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        """
        专门为WADI数据集的.npy文件结构定制的加载器。
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # --- 数据加载与预处理 ---
        # 1. 加载训练数据 (WADI_train.npy)
        train_path = os.path.join(data_path, 'WADI_train.npy')
        train_data = np.load(train_path)
        train_data = np.nan_to_num(train_data)

        # 2. 在训练数据上拟合(fit)标准化器
        self.scaler.fit(train_data)

        # 3. 标准化(transform)训练数据
        self.train = self.scaler.transform(train_data)

        # 4. 加载测试数据 (WADI_test.npy)
        test_path = os.path.join(data_path, 'WADI_test.npy')
        test_data = np.load(test_path)
        test_data = np.nan_to_num(test_data)

        # 5. 使用同一个标准化器来标准化测试数据
        self.test = self.scaler.transform(test_data)

        # 6. 加载测试标签 (WADI_test_label.npy)
        label_path = os.path.join(data_path, 'WADI_test_label.npy')
        self.test_labels = np.load(label_path)  # 原始标签是 1 (正常) 和 -1 (异常)

        # 确保标签是二维的 (num_samples, 1)
        if self.test_labels.ndim == 1:
            self.test_labels = self.test_labels.reshape(-1, 1)

        # --- 关键修改：在这里计算并打印点级别的异常率 ---
        if self.mode in ['test', 'val']:  # 只在需要用到测试标签时计算
            total_points = len(self.test_labels)
            # 在WADI数据集中, -1 代表异常
            anomaly_points = np.sum(self.test_labels == -1)

            if total_points > 0:
                anomaly_ratio = (anomaly_points / total_points) * 100
                print(f"--- WADI Dataset Anomaly Analysis点级别的异常率 ---")
                print(f"Total test points: {total_points}")
                print(f"Anomaly points (label == -1): {anomaly_points}")
                print(f"Point-level anomaly rate: {anomaly_ratio:.4f}%")
                print(f"-------------------------------------")

                # 2. 计算非重叠窗口级别的异常率 (Non-overlapping Window-level Anomaly Rate)
                # 使用 win_size 作为非重叠窗口的步长
                non_overlap_step = self.win_size
                total_windows = 0
                anomalous_windows = 0

                for i in range(0, total_points, non_overlap_step):
                    # 获取当前窗口的标签
                    window_labels = self.test_labels[i: i + non_overlap_step]

                    # 确保窗口是完整的，如果不是则忽略 (这是常见做法)
                    if len(window_labels) < non_overlap_step:
                        continue

                    total_windows += 1

                    # 检查窗口内是否有异常点 (-1)
                    if np.any(window_labels == -1):
                        anomalous_windows += 1

                if total_windows > 0:
                    window_anomaly_ratio = (anomalous_windows / total_windows) * 100
                    print(f"--- WADI Non-Overlapping Window Analysis (win_size={self.win_size}) ---")
                    print(f"Total non-overlapping windows: {total_windows}")
                    print(f"Windows with at least one anomaly: {anomalous_windows}")
                    print(f"[ANSWER] Window-level anomaly rate: {window_anomaly_ratio:.4f}%")
                    print(f"----------------------------------------------------------")
                else:
                    print("Warning: Could not calculate window-level anomaly rate (total_windows is zero).")

            else:
                print("Warning: WADI test labels are empty or not loaded correctly.")

        # 7. 为训练数据创建全零标签
        self.train_labels = np.zeros((self.train.shape[0], 1))

        print(f"--- WADI Npy Loader initialized for mode: '{self.mode}' ---")
        print(f"Train data shape: {self.train.shape}")
        print(f"Test data shape: {self.test.shape}")
        print(f"Test labels shape: {self.test_labels.shape}")

    def __len__(self):
        # 这个方法和其他Loader完全一样
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        # 这个方法和其他Loader完全一样
        index = index * self.step
        if self.mode == "train":
            # 注意：您的代码中train模式返回的是test_labels的第一个窗口，这里保持一致
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])



def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', val_ratio=0.2):
    '''
    model : 'train' or 'test'
    '''
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'SWaT'):
        dataset = SWaTSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'WADI'):
        dataset = WADISegLoader(data_path, win_size, step, mode)
    elif (dataset == 'NIPS_TS_Water'):
        dataset = NIPS_TS_WaterSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'NIPS_TS_Swan'):
        dataset = NIPS_TS_SwanSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'NIPS_TS_creditcard'):
        dataset = NIPS_TS_CCardSegLoader(data_path, win_size, step, mode)
    shuffle = False
    if mode == 'train':
        shuffle = True

        dataset_len = int(len(dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))

        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)


        indices = torch.arange(dataset_len)
        

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(dataset, val_sub_indices)
        
        train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        k_use_len = int(train_use_len*0.1)
        k_sub_indices = indices[:k_use_len]
        k_subset = Subset(dataset, k_sub_indices)
        k_loader = DataLoader(dataset=k_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        return train_loader, val_loader, k_loader

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    
    return data_loader, data_loader