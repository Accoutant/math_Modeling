import random
from net import matrix
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from process_data import load_net2_data
from d2l import torch as d2l
import matplotlib.pyplot as plt

# y = a(ZH的b次方 * ZDR的c次方)


class RainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cov3d = nn.Sequential(nn.Conv3d(3, 1, kernel_size=(1, 1, 1)),
                                   nn.ReLU())
        self.a = nn.Parameter(data=torch.randn(1) + 1e-4, requires_grad=True)
        self.b = nn.Parameter(data=torch.rand(1) + 1e-4, requires_grad=True)
        self.c = nn.Parameter(data=torch.rand(1) + 1e-4, requires_grad=True)

    def forward(self, XS):
        # XS.shape: time_steps, height, features, H, L
        XS = self.cov3d(XS).squeeze(1)
        # XS.shape: batch_size, time_steps, features, H, l
        ZH = XS[:, 0, :, :]   # shape: batch_size, time_steps, H, L
        print(ZH)
        ZDR = XS[:, 1, :, :]
        output = torch.log(self.a * (torch.pow(ZH, self.b) * torch.pow(ZDR, self.c)))
        # output.shape: batch_size, time_steps, H, L
        return output


class TrainerRain:
    def __init__(self, net, optimizer, loss, lr, device):
        self.device = device
        self.net = net.to(self.device)
        self.lr = lr
        self.optimizer = optimizer(self.net.parameters(), weight_decay=0, lr=self.lr)
        self.loss = loss

    def fit(self, path_list: list, max_epochs, batch_size, seed=2023, k=2):
        random.seed(seed)
        random.shuffle(path_list)
        print(len(path_list))
        train_path_list = path_list[:130]
        test_path_list = path_list[130:]
        n = len(train_path_list)
        animitor = d2l.Animator(xlabel="epoch", ylabel="CSI")
        for epoch in range(max_epochs):
            num_path = 1
            for train_path in train_path_list:
                XS, YS = load_net2_data(root_path="./data/", root_rain_path="./rain_data/",
                                        path=train_path, batch_size=batch_size)
                XS = torch.tensor(XS)
                YS = torch.tensor(YS)
                dataset = TensorDataset(XS, YS)
                train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                iter = 1
                for X, Y in train_iter:
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    output = self.net(X)
                    loss = self.loss(output, Y)
                    loss = loss + torch.log(k / loss + 1)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                    self.optimizer.step()
                    print("| epoch %d | path_num %d/%d | iter %d/%d | loss %.4f |" % (epoch+1, num_path, n, iter, len(train_iter), loss))
                    iter += 1
                num_path += 1
            torch.save(self.net.state_dict(), "param_net_2" + str(epoch+1) + ".pkl")

            # 测试数据
            metric = d2l.Accumulator(2)
            with torch.no_grad():
                for test_path in test_path_list:
                    test_XS, test_YS = load_net2_data(root_path="./data/", root_rain_path="./rain_data/",
                                                      path=test_path, batch_size=batch_size)
                    test_XS = torch.tensor(test_XS)
                    test_YS = torch.tensor(test_YS)
                    test_dataset = TensorDataset(test_XS, test_YS)
                    test_iter = DataLoader(test_dataset)
                    for test_X, test_Y in test_iter:
                        test_X = test_X.to(self.device)
                        test_Y = test_Y.to(self.device)
                        test_output = self.net(test_X)
                        hit, miss, false_hit = matrix(test_output, test_Y, device=self.device, threshold=10)
                        CSI = hit.sum().item() / (hit.sum().item() + miss.sum().item() + false_hit.sum().item() + 1e-4)
                        metric.add(CSI, 1)
                        print("test_path %s, CSI %.5f" % (test_path, CSI))
            animitor.add(epoch+1, metric[0]/metric[1])



def my_collect(batch):
    X_list, Y_list = [], []
    for X, Y in batch:
        X_list.append(X)
        Y_list.append(Y)
    return torch.tensor(X_list), torch.tensor(Y_list)