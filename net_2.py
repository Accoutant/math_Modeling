import random
from net import matrix
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from process_data import load_net2_data

# y = a(ZH的b次方 * ZDR的c次方)


class RainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cov3d = nn.Sequential(nn.Conv3d(3, 1, kernel_size=(1, 1, 1)),
                                   nn.ReLU())
        self.a = nn.Parameter(data=torch.randn(1), requires_grad=True)
        self.b = nn.Parameter(data=torch.randn(1), requires_grad=True)
        self.c = nn.Parameter(data=torch.randn(1), requires_grad=True)

    def forward(self, XS):
        # XS.shape: bath_size, time_steps, height, features, H, L
        shape = XS.shape[:2] + XS.shape[3:]
        XS = torch.flatten(XS, 0, 1)
        XS = self.cov3d(XS).squeeze(1)
        XS = torch.reshape(XS, shape)
        # XS.shape: batch_size, time_steps, features, H, l
        ZH = XS[:, :, 0, :, :]   # shape: batch_size, time_steps, H, L
        ZDR = XS[:, :, 1, :, :]
        output = self.a * (torch.pow(ZH, self.b) * torch.pow(ZDR, self.c))
        # output.shape: batch_size, time_steps, H, L
        return output


class TrainerRain:
    def __init__(self, net, optimizer, loss, lr, device):
        self.device = device
        self.net = net.to(self.device)
        self.lr = lr
        self.optimizer = optimizer(self.net.parameters(), weight_decay=0, lr=self.lr)
        self.loss = loss

    def fit(self, path_list: list, max_epochs, batch_size, seed=2023):
        random.seed(seed)
        random.shuffle(path_list)
        train_path_list = path_list[:130]
        test_path_list = path_list[130:]
        n = len(train_path_list)
        for epoch in range(max_epochs):
            num_path = 1
            for train_path in train_path_list:
                XS, YS = load_net2_data(root_path="./data/", root_rain_path="./rain_data/",
                                        path=train_path, batch_size=batch_size)
                dataset = TensorDataset(XS, YS)
                train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                iter = 1
                for X, Y in train_iter:
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    output = self.net(X)
                    loss = self.loss(output, Y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                    self.optimizer.step()
                    print("| epoch %d | path_num %d/%d | iter %d/%d | loss %.4f |" % (epoch, num_path, n, iter, len(train_iter), loss))
                    iter += 1
                num_path += 1
            torch.save(self.net.state_dict(), "param_net_2" + str(epoch+1) + ".pkl")

            # 测试数据
            with torch.no_grad():
                for test_path in test_path_list:
                    test_XS, test_YS = load_net2_data(root_path="./data/", root_rain_path="./rain_data/",
                                                      path=test_path, batch_size=batch_size)
                    test_dataset = TensorDataset(test_XS, test_YS)
                    test_iter = DataLoader(test_dataset)
                    for test_X, test_Y in test_iter:
                        test_X = test_X.to(self.device)
                        test_Y = test_Y.to(self.device)
                        test_output = self.net(test_X)
                        hit, miss, false_hit = matrix(test_output, test_Y, device=self.device, threshold=35)
                        CSI = hit.sum().item() / (hit.sum().item() + miss.sum().item(), false_hit.sum().item())
                        print("test_path %s, CSI %.5f" % (test_path, CSI))


