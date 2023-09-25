import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
from process_data import split_data
import random


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Conv2d(3, 64, (7, 7), 2, 3)
        self.b2 = nn.Sequential(nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.MaxPool2d(3, 2, 1))
        self.cov1 = d2l.Residual(64, 64, use_1x1conv=False, strides=1)
        self.cov2 = d2l.Residual(64, 128, use_1x1conv=True, strides=2)
        self.cov3 = d2l.Residual(128, 256, use_1x1conv=True, strides=2)
        self.cov4 = d2l.Residual(256, 512, use_1x1conv=True, strides=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, X):
        # X.shape: batch_size, features, L, H
        X0 = self.b1(X)
        X1 = self.b2(X0)
        X1 = self.cov1(X1)
        X2 = self.cov2(X1)
        X3 = self.cov3(X2)
        X4 = self.cov4(X3)
        X_flattened = self.adaptiveavgpool(X4)
        X_flattened = torch.flatten(X_flattened, 1, -1)
        output = (X0, X1, X2, X3, X4)
        return output, X_flattened


class CovWithLstmEncoder(nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.cov3d = nn.Sequential(nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(1, 1, 1)),
                     nn.ReLU(),
                     nn.Conv3d(in_channels=3, out_channels=1, kernel_size=(1, 1, 1)))
        self.encoderblocks = nn.Sequential()
        for i in range(time_steps):
            self.encoderblocks.add_module("encoderblock" + str(i), EncoderBlock())
        self.lstm = nn.LSTM(input_size=2048, hidden_size=2048, num_layers=1, batch_first=True)

    def forward(self, XS: torch.Tensor):
        # XS.shape: data_shape: batch_size, time_steps, height, features, H, L
        shape = XS.shape[:2] + XS.shape[3:]   # shape: batch_size, time_steps, features, H, L
        XS = torch.flatten(XS, start_dim=0, end_dim=1)  # 将XS展开以通过3维卷积
        XS = self.cov3d(XS).squeeze(1)
        XS = torch.reshape(XS, shape)   # 重塑形状
        # XS.shape: batch_size, time_steps(10), features, L, H
        batch_size = XS.shape[0]
        time_steps = XS.shape[1]
        XS = XS.permute(1, 0, 2, 3, 4)
        outputs = []
        X_flatteneds = torch.zeros((time_steps, batch_size, 2048), device=d2l.try_gpu())
        for t, X in enumerate(XS):
            # X.shape: batch_size, features, L, H
            output, X_flattened = self.encoderblocks[t](X)
            X_flatteneds[t, :, :] = X_flattened
            outputs.append(output)
        X_flatteneds = X_flatteneds.permute(1, 0, 2)
        # X_flatteneds.shape: batch_size, time_steps, 2048
        lstm_outputs, (h, c) = self.lstm(X_flatteneds)
        # lstm_output.shape: batch_size, time_steps, 2048
        lstm_outputs = lstm_outputs.reshape(batch_size, time_steps, 512, 2, 2)
        # lstm_output.shape: batch_size, time_steps, 512, 2, 2
        return outputs, lstm_outputs


class Reresidual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(input_channels, num_channels, (3, 3), stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(num_channels, num_channels, kernel_size=(4, 4), stride=strides, padding=1)
        if use_1x1conv:
            self.conv3 = nn.ConvTranspose2d(input_channels, num_channels, kernel_size=(4, 4), stride=strides, padding=1)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_up = nn.UpsamplingNearest2d(scale_factor=4)
        self.cov4 = nn.ConvTranspose2d(1024, 256, (4, 4), 2, 1)
        self.cov3 = Reresidual(512, 128, use_1x1conv=True, strides=2)
        self.cov2 = Reresidual(256, 64, use_1x1conv=True, strides=2)
        self.cov1 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.uncov = nn.ConvTranspose2d(128, 1, (4, 4), 2, 1)

    def forward(self, X, state):
        # X就是lstm.shape lstm_output.shape: batch_size, 512, 2, 2
        X = self.lstm_up(X)  # 把X变为512, 4, 4
        X4 = self.cov4(torch.cat((X, state[4]), dim=1))
        X3 = self.cov3(torch.cat((X4, state[3]), dim=1))
        X2 = self.cov2(torch.cat((X3, state[2]), dim=1))
        X1 = self.cov1(torch.cat((X2, state[1]), dim=1))
        X1 = self.unpool(X1)
        output = self.uncov(torch.cat((X1, state[0]), dim=1))
        return output


class CovWithLstmDecoder(nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.time_steps = time_steps
        self.decoderblocks = nn.Sequential()
        for i in range(time_steps):
            self.decoderblocks.add_module("decoderblock"+str(i), DecoderBlock())


    def forward(self, outputs, lstm_outputs):
        # len(outputs)=10
        batch_size = lstm_outputs.shape[0]
        # lstm_output.shape: batch_size, time_steps, 512, 2, 2
        decoder_outputs = []
        for i in range(self.time_steps):
            decoder_output = self.decoderblocks[i](lstm_outputs[:, i, :, :, :].squeeze(1), outputs[i])
            decoder_outputs.append(decoder_output)
        return torch.cat(decoder_outputs, dim=1)


class CovWithLstm(nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.encoder = CovWithLstmEncoder(time_steps)
        self.decoder = CovWithLstmDecoder(time_steps)

    def forward(self, XS):
        # XS.shape: batch_size, time_steps, features, H, L
        outputs, lstm_outputs = self.encoder(XS)
        Y = self.decoder(outputs, lstm_outputs)
        return Y


class Trainer:
    def __init__(self, net, optimizer, loss, lr, device):
        self.device = device
        self.net = net.to(self.device)
        self.net.apply(self.init_weights)
        self.lr = lr
        self.optimizer = optimizer(self.net.parameters(), weight_decay=0, lr=self.lr)
        self.loss = loss

    def fit(self, path_list: list, max_epochs, batch_size, jump, k, seed=2023):
        # data_shape: batch_size, time_steps, height, features, H, L
        random.seed(seed)
        random.shuffle(path_list)
        print(path_list)
        train_path_list = path_list[:130]
        test_path_list = path_list[130:]
        animator = d2l.Animator(xlabel="epoch", ylabel="CSI")
        for epoch in range(max_epochs):
            path_num = 1
            total_path_num = len(train_path_list)
            num_list = len(train_path_list)
            for i in range(0, num_list, jump):
                paths = train_path_list[i : i + jump]
                train_iter = split_data(paths, 10, batch_size)
                n = 1
                n_all = len(train_iter)
                for XS, YS in train_iter:
                    XS = XS.to(self.device)
                    YS = YS.to(self.device)
                    output = self.net(XS)
                    loss = self.loss(output, YS)
                    loss = loss + torch.log(k/loss + 1)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                    self.optimizer.step()
                    print("| epoch %d | path_num %d/%d | iter %d/%d | loss %.2f |" % (epoch+1, path_num, total_path_num, n, n_all, loss))
                    n += 1
                path_num += jump

            # save_params
            torch.save(self.net.state_dict(), "param" + str(epoch+1) + ".pkl")
            # ecaluate
            with torch.no_grad():
                CIS = evaluate(self.net, test_path_list, time_steps=10, batch_size=batch_size, jump=jump, device=self.device, save_path="param" + str(epoch+1) + ".pkl")



    def init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Conv3d or type(m) == nn.ConvTranspose2d:
            nn.init.xavier_normal_(m.weight)


def evaluate(net, path_list: list, time_steps, batch_size, jump, device, save_path):
    net = net.to(device)
    net.load_state_dict(torch.load(save_path))
    path_list = path_list[130:]
    n = len(path_list)
    metric = d2l.Accumulator(2)
    for i in range(0, n, jump):
        paths = path_list[i: i + jump]
        test_iter = split_data(paths, time_steps, batch_size)
        for XS, YS in test_iter:
            XS = XS.to(device)
            YS = YS.to(device)
            output = net(XS)
            # output.shape: batch_size, time_steps, H, L
            hit, miss, false_hit = matrix(output, YS, device)
            hit = torch.sum(hit).item()
            miss = torch.sum(miss).item()
            false_hit = torch.sum(false_hit).item()
            CSI = hit / (hit + miss + false_hit)
            metric.add(CSI, 1)
            print(CSI)
            print("CSI %.5f" % (metric[0]/metric[1]))
    print(metric[0])
    return metric[0]/metric[1]


def matrix(output, target, device, threshold=35):
    # output.shape: batch_size, time_steps, H, L
    # target.shape: batch_size, time_steps, H, L
    def get_evmatrix(X, threshold=threshold):
        temp = X > threshold
        X[temp] = 1
        X[~temp] = 0
        return X
    output_matrix = get_evmatrix(output)
    target_matrix = get_evmatrix(target)
    hit_matrix = torch.eq(output_matrix, target_matrix) & (output_matrix == 1)
    miss_matrix = (target_matrix == 1) & (output_matrix == 0)
    false_matrix = (target_matrix == 0) & (output_matrix == 1)

    shape = output.shape
    hit = torch.zeros(shape, device=device)
    miss = torch.zeros(shape, device=device)
    false_hit = torch.zeros(shape, device=device)

    hit[hit_matrix] = 1
    miss[miss_matrix] = 1
    false_hit[false_matrix] = 1

    return hit, miss, false_hit


"""
# output.shape: batch_size, time_steps, H, L
output = torch.randint(0, 90, (6, 10, 256, 256))
target = torch.randint(0, 90, (6, 10, 256, 256))

a = torch.zeros(output.shape)



hit, miss, false_hit = matrix(output, target, device=torch.cpu)
hit = torch.sum(hit).item()

miss = torch.sum(miss).item()
false_hit = torch.sum(false_hit).item()
print(hit/(hit + miss + false_hit))
"""



