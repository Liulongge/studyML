import torch
from restnet18 import ResNet18
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose(
    [transforms.Resize(32),
     transforms.CenterCrop(32),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])#采用和训练方法一样的标准化处理
# load image
path = "./img/"
file_list = os.listdir(path)
for file in file_list :
    print(file)
    img = Image.open(path + file)
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    class_indict = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = ResNet18()
    # load model weights
    model_weight_path = "./net_007.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))#载入训练好的模型参数
    model.eval()#使用eval()模式
    with torch.no_grad():#不跟踪损失梯度
        # predict class
        output = model(img)
        num, predicted = torch.max(output.data, 1)
        predict = torch.softmax(output, dim=1)#通过softmax得到概率分布
    if float(predict[0][predicted].numpy()) < 0.7 :
        print("预测失败！！！")
    else :
        print(class_indict[int(predicted)], ": 概率%.3f" % float(predict[0][predicted].numpy()))
plt.show()
