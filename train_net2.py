from net_2 import RainNet, TrainerRain
from torch import nn, optim
from d2l import torch as d2l
import os
from view import count_valid_all
import pickle

rainnet = RainNet()


lr = 0.1
batch_size = 6
max_epochs = 3
optimizer = optim.SGD
loss = nn.MSELoss()

with open("choose_list.pkl", "rb") as f:
    paths = pickle.load(f)
trainerrain = TrainerRain(rainnet, optimizer, loss, lr, device=d2l.try_gpu())
trainerrain.fit(paths, max_epochs=max_epochs, batch_size=batch_size)
