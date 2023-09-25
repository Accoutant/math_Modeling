import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
import numpy as np


def split_data(paths: list, time_steps, batch_size):
    XS_list = []
    for path in paths:
        with open("./data/"+path, "rb") as f:
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


def load_dir(root_path, path):
    data = np.load(str(root_path) + str(path))
    return data


if __name__ == "__main__":
    """
    path = "data_dir_000.pkl"
    train_iter = split_data(path, 10, 1)
    test = next(iter(train_iter))[0][0, 1, 0, 2, :, :]
    print(test.shape)
    plt.imshow(test.detach().numpy())
    plt.show()
    """
    list = [1, 2, 3, 4, 5, 6, 7]
    n = len(list)
    for i in range(0, n, 3):
        path = list[i:i+3]
        print(path)