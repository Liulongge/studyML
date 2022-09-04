import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        
        self.shortcut = nn.Sequential()
#         resnet就是这样设计的啊，你可以看第一张网络结构图，
# shortcut都是加在通道和输出通道不相等的地方，
# 如果输入输出通道相同，就都是加在步长不为1的地方
        if stride != 1 or inchannel != outchannel:
#             print(stride,inchannel,outchannel)        
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
#         print(type(ResidualBlock)) 用类进行初始化
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
        #ResNet18 18 =1(初始卷积）+4*4+1（全连接）

    def make_layer(self, block, channels, num_blocks, stride):
#         print(stride)
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
#         print(strides)
        layers = []#将model children 加到列表中
        for stride in strides:#每次重复两次残差块，总共4层卷积层参数
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels 
        return nn.Sequential(*layers)#返回所有model.childen
    #inchanels: 64--->64--->128--->256
    #outchanels:64--->128--->256--->512
    #strides[1,1]--->[2,1]--->[2,1]--->[2,1]

    def forward(self, x):
        out = self.conv1(x)#shape [2, 64, 32, 32]
#         print(out.shape)
        out = self.layer1(out)#shape 第一次 inchannel ==outchannel,stride ==1,不使用残差块
        #shape [2, 64, 32, 32]
#         print(out.shape)
        out = self.layer2(out)
#         print(out.shape) [2, 128, 16, 16]
        out = self.layer3(out)
#         print(out.shape)[2, 256, 8, 8]
        out = self.layer4(out)
#         print(out.shape)[2, 512, 4, 4]
        out = F.avg_pool2d(out, 4)
#         print(out.shape)[2, 512, 1, 1]
        out = out.view(out.size(0), -1)
#         print(out.shape)shape[2, 512]
        out = self.fc(out)
#         print(out.shape)torch.Size([2, 10])
        return out


def ResNet18():

    return ResNet(ResidualBlock)
