# https://blog.csdn.net/weixin_44503976/article/details/127754679

import numpy as np

image = np.array([[[[1, 2, 3],
                   [1, 2, 3],
                   [1, 2, 3]],

                  [[1, 2, 3],
                   [1, 2, 3],
                   [1, 2, 3]],

                  [[1, 2, 3],
                   [1, 2, 3],
                   [1, 2, 3]]]])
# 创建nchw 1x1x3x3 kernel(输出图像channel为1)
kernel = np.array([[[[1, 2, 3],
                   [1, 2, 3],
                   [1, 2, 3]],

                  [[1, 2, 3],
                   [1, 2, 3],
                   [1, 2, 3]],

                  [[1, 2, 3],
                   [1, 2, 3],
                   [1, 2, 3]]]])



import numpy as np

def convolve(image, kernel, padding=1, stride=1, dilation=2):

    batch, channels, height, width = image.shape
    kernel_n, kernel_c, kernel_h, kernel_w = kernel.shape

    kernel_size = kernel_w
    print(image.shape)
    padded_image = np.pad(image, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    print(padded_image.shape)
    # 空洞卷积输出shape计算公式
    output_height = (height + 2 * padding - kernel_size - (kernel_size - 1) * (dilation -1 )) // stride + 1
    output_width = (width + 2 * padding - kernel_size - (kernel_size - 1) * (dilation -1 )) // stride + 1
    output_channel = kernel_n

    output = np.zeros((1, output_channel, output_height, output_width))
    print(output.shape)
    for h in range(output_height):
        for w in range(output_width):
            for c in range(output_channel):
                # 获取图像信息, n~n+3行, n~n+3列, 全部channel
                window = padded_image[:, :,
                                     h * stride:(h * stride + kernel_size * 2 - 1):2,
                                     w * stride:(w * stride + kernel_size * 2 - 1):2]
                # 图像与第x个kernel计算
                output[0, :, h, w] = np.sum(window * kernel[0, :, :, :])

    return output

output = convolve(image, kernel, padding=2, stride=1, dilation=2)
print("numpy, 空洞卷积:\n", output)

import torch  
import torch.nn as nn  
from torch import tensor

input_tensor = torch.from_numpy(image).float()

atrous_conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, dilation=2, padding=2)  
atrous_conv.weight.data = torch.from_numpy(kernel).float()
# 将输入张量通过空洞卷积层  
output_tensor = atrous_conv(input_tensor)  
print("pytorch, 空洞卷积:\n", output_tensor)  # 输出应为 (1, 16, 32, 32)，因为空洞卷积不会改变输入的空间尺寸（当padding正确时）