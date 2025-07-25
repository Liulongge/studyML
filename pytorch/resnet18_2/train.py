'''ResNet-18 Image classfication for cifar-10 with PyTorch 



'''
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



# 训练
if __name__ == "__main__":
        # 定义是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 超参数设置
    EPOCH = 135   #遍历数据集次数
    pre_epoch = 0  # 定义已经遍历数据集的次数
    BATCH_SIZE = 128      #批处理尺寸(batch_size)
    LR = 0.01        #学习率

    # 准备数据集并预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./CIFAR10/', train=True, download=True, transform=transform_train) #训练数据集
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

    testset = torchvision.datasets.CIFAR10(root='./CIFAR10/', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    # Cifar-10的标签
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 模型定义-ResNet
    net = ResNet18().to(device)
    # 定义损失函数和优化方式
    m='./model/'
    criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    best_acc = 85  #2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), 'net_%03d.pth' % (epoch + 1))
                    torch.save(optimizer.state_dict(), 'optimizer_%03d.pth' % (epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
