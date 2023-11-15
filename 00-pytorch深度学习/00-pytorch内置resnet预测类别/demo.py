from torchvision import models
from torchvision import transforms
# print(dir(models))
# 输出：
# ['AlexNet', 'ConvNeXt', 'DenseNet', 'EfficientNet', 'GoogLeNet', 'GoogLeNetOutputs', 
#  'Inception3', 'InceptionOutputs', 'MNASNet', 'MobileNetV2', 'MobileNetV3', 'RegNet', 
#  'ResNet', 'ShuffleNetV2', 'SqueezeNet', 'VGG', 'VisionTransformer', '_GoogLeNetOutputs', 
#  '_InceptionOutputs', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', 
#  '__name__', '__package__', '__path__', '__spec__', '_utils', 'alexnet', 'convnext', 
#  'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'densenet', 
#  'densenet121', 'densenet161', 'densenet169', 'densenet201', 'detection', 'efficientnet', 
#  'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 
#  'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'feature_extraction', 'googlenet', 
#  'inception', 'inception_v3', 'mnasnet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 
#  'mnasnet1_3', 'mobilenet', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 
#  'mobilenetv2', 'mobilenetv3', 'optical_flow', 'quantization', 'regnet', 'regnet_x_16gf', 
#  'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 
#  'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 
#  'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet', 'resnet101', 
#  'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d', 
#  'segmentation', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 
#  'shufflenet_v2_x2_0', 'shufflenetv2', 'squeezenet', 'squeezenet1_0', 'squeezenet1_1', 
#  'vgg', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
#  'video', 'vision_transformer', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2']

resnet = models.resnet101(pretrained=True)
# print(resnet)

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),          # 图片统一缩放到256*256
    transforms.CenterCrop((224, 224)),      # 裁剪为224*224，如果是512的话，超出244的区域填充为黑色
    transforms.ToTensor(),                  # 将图像转换为tensor
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # 数据归一化
    ])

from PIL import Image
img = Image.open('./bird.jpg')
img_t = preprocess(img)         # 图像预处理
# img.show()
print("图像预处理后, tensor shape: ", img_t.shape)              # torch.Size([3, 224, 224]), 此时为tensor

import torch
batch_t = torch.unsqueeze(img_t, 0) # 作用：扩展维度, 增加1维, 返回一个新的张量，对输入的既定位置插入维度
print("扩充维度后, tensor shape: ", batch_t.shape)                # torch.Size([1, 3, 224, 224])

resnet.eval()                       # 设置网络未推理模式
out = resnet(batch_t)               # 推理
index = int(out.argmax())
print("推理后, 网络输出shape: ", out.shape)
print("argmax求概率最大值索引: ", index)
print("输出概率求和不为1: ", float(out.sum()))

precentage = torch.nn.functional.softmax(out, dim = 1)[0]
print("softmax归一化后和为1: ", float(precentage.sum()))

# 读取标签
labels=[]
with open('./imagenet_classes.txt') as f:
    for line in f:
        i = line.find(",")
        result = line[i + 2:]
        labels.append(result.strip()) # strip()用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列

print("输出类别: {}, 概率值: {}".format(labels[index], float(precentage[index])))
