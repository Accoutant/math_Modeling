import torch
from net import CovWithLstm
from net import Trainer
from torch import nn, optim
from d2l import torch as d2l
from view import count_valid_all

net = CovWithLstm(10)
net.load_state_dict(torch.load("param1.pkl"))
max_epoch = 5
batch_size = 6
lr = 0.5
loss = nn.MSELoss()
optimizer = optim.SGD
path_list = count_valid_all()
print(path_list)
trainer = Trainer(net, optimizer, loss, lr, device=d2l.try_gpu())
trainer.fit(path_list[:], max_epochs=max_epoch, batch_size=batch_size, jump=1, k=2)
torch.save(net.state_dict(), "param_used.pkl")
