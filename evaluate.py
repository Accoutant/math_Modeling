from net import CovWithLstm, evaluate
import os
from d2l import torch as d2l
import matplotlib.pyplot as plt
from process_data import split_data
import torch

path_list = os.listdir("./data")
net = CovWithLstm(time_steps=10)

net = net.to(d2l.try_gpu())
net.load_state_dict(torch.load("param2.pkl"))
test_iter = split_data(["data_dir_080.pkl"], 10, 6)
test_data = next(iter(test_iter))[0]
test_data = test_data.to(d2l.try_gpu())
output = net(test_data)
output = output.cpu()
target = next(iter(test_iter))[1]
d2l.show_images(output[0].detach().numpy(), 2, 5)
d2l.show_images(target[0].detach().numpy(), 2, 5)
plt.show()

evaluate(net, path_list, time_steps=10, batch_size=6, jump=1, device=d2l.try_gpu())