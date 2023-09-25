import torch
from net import CovWithLstm
from net import Trainer
from torch import nn, optim
import os
from d2l import torch as d2l

net = CovWithLstm(10)

max_epoch = 2
batch_size = 2
lr = 0.01
loss = nn.MSELoss()
optimizer = optim.SGD
path_list = os.listdir("./data")
print(path_list)
trainer = Trainer(net, optimizer, loss, lr, device=d2l.try_gpu())
trainer.fit(path_list[:], max_epochs=max_epoch, batch_size=6, jump=1, k=5)
torch.save(net.state_dict(), "param_used.pkl")
