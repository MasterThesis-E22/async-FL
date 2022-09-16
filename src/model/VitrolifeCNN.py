import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class VitrolifeCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(stride=3, kernel_size=32, out_channels=8, in_channels=1)
        self.conv2 = nn.Conv2d(stride=2, kernel_size=32, out_channels=16, in_channels=8)
        self.conv3 = nn.Conv2d(kernel_size=16, out_channels=32, in_channels=16, padding="same")
        self.conv4 = nn.Conv2d(kernel_size=16, out_channels=64, in_channels=32, padding="same")

        self.pool = nn.MaxPool2d(2, 2)
        self.dropOut = nn.Dropout(0.2)
        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(out_features=256, in_features=256)
        self.dense2 = nn.Linear(out_features=2, in_features=256)

        self.batchNorm1 = nn.BatchNorm2d(num_features=8)
        self.batchNorm2 = nn.BatchNorm2d(num_features=16)
        self.batchNorm3 = nn.BatchNorm2d(num_features=32)
        self.batchNorm4 = nn.BatchNorm2d(num_features=64)
    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 250, 250)

        tensor = self.conv1(tensor)
        tensor = self.batchNorm1(tensor)
        tensor = F.relu(tensor)
        tensor = self.dropOut(tensor)

        tensor = self.conv2(tensor)
        tensor = self.batchNorm2(tensor)
        tensor = F.relu(tensor)
        tensor = self.pool(tensor)
        tensor = self.dropOut(tensor)

        tensor = self.conv3(tensor)
        tensor = self.batchNorm3(tensor)
        tensor = F.relu(tensor)
        tensor = self.pool(tensor)
        tensor = self.dropOut(tensor)

        tensor = self.conv4(tensor)
        tensor = self.batchNorm4(tensor)
        tensor = F.relu(tensor)
        tensor = self.pool(tensor)

        tensor = self.flatten(tensor)
        tensor = self.dense1(tensor)
        tensor = F.relu(tensor)
        tensor = self.dense2(tensor)
        #tensor = F.sigmoid(tensor)
        # torch.reshape(tensor, (-1,))
        return tensor

    def train_one_epoch(self, epoch, dev, train_dl, model, loss_func, opti, mu):
        if mu != 0:
            global_model = copy.deepcopy(model)
        # 设置迭代次数
        data_sum = 0
        for epoch in range(epoch):
            for data, label in train_dl:
                data, label = data.to(dev), label.to(dev)
                # 模型上传入数据
                preds = model(data)
                # 计算损失函数
                loss = loss_func(preds, label)
                data_sum += label.size(0)
                # 正则项
                if mu != 0:
                    proximal_term = 0.0
                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss = loss + (mu / 2) * proximal_term
                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                opti.step()
                # 将梯度归零，初始化梯度
                opti.zero_grad()
        # 返回当前Client基于自己的数据训练得到的新的模型参数
        weights = copy.deepcopy(model.state_dict())
        for k, v in weights.items():
            weights[k] = weights[k].cpu().detach()
        return data_sum, weights
