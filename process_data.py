import pickle
import random

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
import numpy as np


def split_data(paths: list, time_steps, batch_size):
    XS_list = []
    for path in paths:
        with open(path, "rb") as f:
            XS = pickle.load(f)
            XS_list.append(XS)
    XS = np.concatenate(XS_list, axis=0)
    n = XS.shape[0]
    XS_train_list = []
    YS_list = []
    for i in range(n):
        if i <= n -2 *time_steps:
            XS_train = XS[i: i + time_steps]
            YS = XS[i + time_steps: (i + 2 * time_steps), 0, 0, :, :]   # 形成YS并且选择特征0作为target
            XS_train_list.append(XS_train)
            YS_list.append(YS)
    # shape: batch_sizt, time_steps, height, features, H, L
    XS_train_list = np.array(XS_train_list)
    YS_list = np.array(YS_list)
    dataset = TensorDataset(torch.tensor(XS_train_list), torch.tensor(YS_list))
    train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_iter


def load_net2_data(root_path, root_rain_path, path: str, batch_size):
    with open(root_path + path, "rb") as f:
        XS = pickle.load(f)
        XS = XS[:, :, [0, 2], :, :]    # shape: time_steps, height, features, H, L

    with open(root_rain_path + path, "rb") as f:
        YS = pickle.load(f)            # shape: time_steps, H, L
        print(YS.shape)
    return (XS, YS)


def load_dir(root_path, path):
    data = np.load(str(root_path) + str(path))
    return data


if __name__ == "__main__":
    paths = ["data_dir_000.pkl", "data_dir_001.pkl"]
    load_net2_data("./data/", "./rain_data/", paths, 2)