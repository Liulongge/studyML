
from models.fine_tune_model import fine_tune_model
from global_config import *
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])#采用和训练方法一样的标准化处理
# load image
img = Image.open("./img/ants.jpg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)
class_indict = {0:'蚂蚁', 1:'蜜蜂'}

model = fine_tune_model()
model.load_state_dict(torch.load(MODEL_SAVE_FILE, map_location=device))#载入训练好的模型参数
model.eval()#使用eval()模式
with torch.no_grad():#不跟踪损失梯度
    # predict class
    output = torch.squeeze(model(img))#压缩batch维度
    predict = torch.softmax(output, dim=0)#通过softmax得到概率分布
    predict_cla = torch.argmax(predict).numpy()#寻找最大值所对应的索引
print(class_indict[int(predict_cla)], predict[predict_cla].numpy())#打印类别信息和概率
plt.show()

