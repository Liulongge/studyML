# 参考:
# https://blog.csdn.net/weixin_41936775/article/details/120083958?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171155720516800188552853%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171155720516800188552853&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-120083958-null-null.142^v100^pc_search_result_base7&utm_term=1x1%E5%8D%B7%E7%A7%AF&spm=1018.2226.3001.4187

import numpy as np

def convolve(image, kernel, padding=1, stride=1):

    channels, height, width = image.shape
    kernel_n, kernel_c, kernel_h, kernel_w = kernel.shape

    kernel_size = kernel_w
    padded_image = np.pad(image, ((0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)

    output_height = (height - kernel_size + 2 * padding) // stride + 1
    output_width = (width - kernel_size + 2 * padding) // stride + 1
    output_channel = kernel_n

    output = np.zeros((output_channel, output_height, output_width))

    for h in range(output_height):
        for w in range(output_width):
            for c in range(output_channel):
                # 获取图像信息, n~n+3行, n~n+3列, 全部channel
                window = padded_image[:,
                                     h * stride:h * stride + kernel_size,
                                     w * stride:w * stride + kernel_size]
                # 图像与第x个kernel计算
                output[c, h, w] = np.sum(window * kernel[c, :, :, :])

    return output

# 创建nchw 1x3x3x4图像
image = np.array([[[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]],
                    [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]],
                    [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]])
# 创建nchw 1x3x1x1 kernel(输出图像channel为1)
kernel = np.array([[[[2]],
                    [[2]],
                    [[2]]]])
# 创建nchw 1x3x1x1 kernel(输出图像channel为3)
# kernel = np.array([[[[2]],
#                     [[2]],
#                     [[2]]],

#                    [[[2]],
#                     [[2]],
#                     [[2]]],

#                    [[[2]],
#                     [[2]],
#                     [[2]]]])

output = convolve(image, kernel, padding=0, stride=1)

print("input:\n{}".format(image))
print("kernel:\n{}".format(kernel))
print("output:\n{}".format(output))


import torch
import torch.nn as nn

# 创建一个输入数据
input_data = torch.tensor(image, dtype=torch.float32)

# 定义一个卷积核
kernel = torch.tensor(kernel, dtype=torch.float32)
# 定义卷积层
conv_layer = nn.Conv2d(in_channels=input_data.shape[-3], out_channels=kernel.shape[-3], kernel_size=1, padding=0, bias=False)

# 将定义好的卷积核赋值给卷积层的权重
conv_layer.weight.data = kernel

# 进行卷积运算
output_data = conv_layer(input_data)

print("input:\n{}".format(input_data))
print("kernel:\n{}".format(conv_layer.weight.data))
print("output:\n{}".format(output_data.data))