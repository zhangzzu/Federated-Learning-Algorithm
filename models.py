# Several basic machine learning models
import torch
from torch import nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    """A simple implementation of Logistic regression model"""
    def __init__(self, num_feature, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_feature, output_size)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    """A simple implementation of Deep Neural Network model"""
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.h1 = 600
        self.h2 = 300
        self.h3 = 100
        self.model = nn.Sequential(
            nn.Linear(input_dim, 600),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(600, 300),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(300, 100),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(100, output_dim))

    def forward(self, x):
        return self.model(x)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # imput 输入是1*28*28,通道是1;kernel_zise 5*5的卷积核
        # 经过conv1，通道变为10*24*24
        #in_channels, out_channels
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 2*2的池化层
        self.pooling = nn.MaxPool2d(2)
        # 线性舒展层
        self.fc = nn.Linear(320, 10)

    # 需实现 前驱函数 forward()

    def forward(self, x):
        # 样本的数量
        batch_size = x.size(0)
        # 先卷积 再池化 最后relu
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        # 展开层 20*4*4 转换成一维320向量
        x = x.view(batch_size, -1)
        # 线性变换 确定0-9的概率
        x = self.fc(x)
        return x


class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
