from net_2 import RainNet, TrainerRain
from torch import nn, optim
from d2l import torch as d2l
import os
from view import count_valid_all

rainnet = RainNet()


lr = 0.01
batch_size = 6
max_epochs = 3
optimizer = optim.SGD
loss = nn.MSELoss()

paths = count_valid_all()
trainerrain = TrainerRain(rainnet, optimizer, loss, lr, device=d2l.try_gpu())
trainerrain.fit(paths, max_epochs=max_epochs, batch_size=batch_size)
