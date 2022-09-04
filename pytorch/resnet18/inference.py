import torch
from restnet18 import RestNet18
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])#采用和训练方法一样的标准化处理
# load image
img = Image.open("./img/street.jpg")
# plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)
class_indict = {0:'飞机', 1:'汽车', 2:'鸟', 3:'猫', 4:'鹿', 5:'狗', 6:'青蛙', 7:'马', 8:'船', 9:'卡车'}

model = RestNet18()
model.load_state_dict(torch.load("./model.pth", map_location=device))#载入训练好的模型参数
model.eval()#使用eval()模式
with torch.no_grad():#不跟踪损失梯度
    # predict class
    output = torch.squeeze(model(img))#压缩batch维度
    predict = torch.softmax(output, dim=0)#通过softmax得到概率分布
    predict_cla = torch.argmax(predict).numpy()#寻找最大值所对应的索引
print(class_indict[int(predict_cla)], predict[predict_cla].numpy())#打印类别信息和概率
plt.show()