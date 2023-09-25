from d2l import torch as d2l
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import torch



def view(root_path="./NJU_CPOL_update2308/dBZ/1.0km/data_dir_046"):
    root_path = (root_path)
    path_list = os.listdir(root_path)

    images = []
    for path in path_list:
        image = np.load(root_path+ '/' + str(path))
        images.append(image)

    d2l.show_images(images, 5, 5, scale=5)
    plt.show()


def load_dir(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    # data.shape: time_steps, height, features, H, L
    return data


def count_valid(X, threshold=10):
    # X.shape: time_steps, height, features, H, L
    X = X[:, 0, 0, :, :]
    temp = torch.zeros(X.shape)
    check = X > threshold
    hit = (check == True).sum()
    return hit / torch.numel(torch.tensor(X))


def count_valid_all():
    path_list = os.listdir("./data")
    hits = []
    for path in path_list:
        path = "./data/" + path
        data = load_dir(path)
        hit = count_valid(data)
        hits.append(hit)
        print("path %s hit %.4f" % (path, hit))
    plt.hist(hits)


if __name__ == "__main__":
    #count_valid_all()
    view("./NJU_CPOL_update2308/dBZ/1.0km/data_dir_100")