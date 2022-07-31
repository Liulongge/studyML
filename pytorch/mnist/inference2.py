import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
 
batch_size_test = 1000

# 加载数据
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

element = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(element) # next() 返回迭代器的下一个项目

# 定义网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)
 
    def forward(self, x):
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        print("======== batch_size", batch_size)
        x = self.fc(x)
        # print(x)
        return x
 
 
# 方式1，加载模型
network = Net()
network.load_state_dict(torch.load('./model2.pth'))

# 推理
output = network(example_data)

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域。
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))

plt.show()
print(network)