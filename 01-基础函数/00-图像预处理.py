
# torchvision：计算机视觉工具包
# torchvision.transforms : 常用的图像预处理方法
# torchvision.datasets : 常用数据集的dataset实现，MNIST，CIFAR-10，ImageNet等
# torchvision.model : 常用的模型预训练，AlexNet，VGG， ResNet，GoogLeNet等

# 数据预处理方法：数据中心化；数据标准化；缩放；裁剪；旋转；填充；噪声添加；灰度变换；线性变换；仿射变换；亮度、饱和度及对比度变换等
# compose将一系列transforms方法进行有序组合包装，依次按顺序的对图像进行操作


import torchvision.transforms as transforms
from PIL import Image
img = Image.open('./bird.jpg')
img.show()

# transforms.Compose: 将一系列的transforms方法进行有序的组合包装，依次按顺序的对图像进行操作
# transforms.Resize: 改变图像大小

# transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'): 
#   从图片中随机裁剪出尺寸为size的图片

# transforms.CenterCrop(size): 
#   从图像中心裁剪图片, size：所需裁剪图片尺寸

# transforms.ToTensor: 将图像转换成张量，同时会进行归一化的一个操作，将张量的值从0-255转到0-1

# transforms.Normalize(mean, std, inplace): 
#   将数据进行标准化, 逐channel的对图像进行标准化。output = （input - mean）/ std
#   mean: 各通道的均值, std: 各通道的标准差, inplace: 是否原地操作

norm_mean = (0.5, 0.5, 0.5)
norm_std = (0.5, 0.5, 0.5)
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),                # 缩放
    transforms.CenterCrop((500, 500)),                    # 中心crop
    # transforms.ToTensor(),                      # 转为tensor，同时进行归一化操作，将像素值的区间从0-255变为0-1
    # transforms.Normalize(norm_mean, norm_std),  # 数据标准化，均值变为0，标准差变为1
])

# 依次有序的从compose中调用数据处理方法
# def __call__(self, img):
#     for t in self.transforms:
#         img = t(img)
#     return img

img_t = train_transform(img)
img_t.show()
