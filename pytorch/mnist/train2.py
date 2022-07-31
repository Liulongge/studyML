import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# 准备数据集
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)
''' 
Conv2d(in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1,bias=True, padding_mode='zeros')
in_channels: 输入的通道数目 【必选】
out_channels: 输出的通道数目 【必选】
kernel_size: 卷积核的大小, 类型为int 或者元组，当卷积是方形的时候，只需要一个整数边长即可，卷积不是方形，要输入一个元组表示 高和宽。【必选】
stride: 卷积每次滑动的步长为多少，默认是 1 【可选】
padding: 设置在所有边界增加 值为 0 的边距的大小(也就是在feature map 外围增加几圈 0), 例如当 padding =1 的时候，如果原来大小为 3 x 3 ，那么之后的大小为 5 x 5 。即在外围加了一圈 0 。【可选】
dilation: 控制卷积核之间的间距（什么玩意？请看例子）【可选】

class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
kernel_size(int or tuple) - max pooling的窗口大小,
stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
padding(int or tuple, optional) - 输入的每一条边补充0的层数
dilation(int or tuple, optional) - 一个控制窗口中元素步幅的参数
return_indices - 如果等于True, 会返回输出最大值的序号, 对于上采样操作会有帮助
ceil_mode - 如果等于True, 计算输出信号大小的时候, 会使用向上取整, 代替默认的向下取整的操作
''' 
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)
 
    def forward(self, x):
        batch_size = x.size(0) # 在某一个维度，我们可以传入数字-1，自动对维度进行计算并变化：
        x = torch.nn.functional.relu(self.pooling(self.conv1(x)))
        x = torch.nn.functional.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1) # 在pytorch中的view()函数就是用来改变tensor的形状的，例如将2行3列的tensor变为1行6列，其中-1表示会自适应的调整剩余的维度
        
        x = self.fc(x)
        return x
 
 
model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
 
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
 
 
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs,target=inputs.to(device),target.to(device)
        optimizer.zero_grad()
 
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d,%.5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0
    torch.save(model.state_dict(), './model2.pth')
    torch.save(model.state_dict(), './optimizer2.pth')
def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            inputs,target=data
            inputs,target=inputs.to(device),target.to(device)
            outputs=model(inputs)
            _,predicted=torch.max(outputs.data,dim=1)
            total+=target.size(0)
            correct+=(predicted==target).sum().item()
    print('Accuracy on test set:%d %% [%d%d]' %(100*correct/total,correct,total))
 
if __name__ =='__main__':
    for epoch in range(10):
        train(epoch)
        test()