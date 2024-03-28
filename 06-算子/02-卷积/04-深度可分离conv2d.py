import numpy as np

def depthwise_conv(image, kernel, padding=1, stride=1):

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
                window = padded_image[c,
                                     h * stride:h * stride + kernel_size,
                                     w * stride:w * stride + kernel_size]
                # 图像与第x个kernel计算
                output[c, h, w] = np.sum(window * kernel[c, :, :, :])

    return output

def pointwise_conv(image, kernel, padding=1, stride=1):

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
# 创建nchw 1x1x3x3 kernel(输出图像channel为1)
kernel = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])

kernel2 = np.array([[[[ 1 ]], [[ 2 ]], [[ 3 ]]], 
                    [[[ 1 ]], [[ 2 ]], [[ 3 ]]], 
                    [[[ 1 ]], [[ 2 ]], [[ 3 ]]],
                    [[[ 1 ]], [[ 2 ]], [[ 3 ]]]])

output = depthwise_conv(image, kernel, padding=1, stride=1)
print("output:\n{}".format(output))
output = pointwise_conv(output, kernel2, padding=0, stride=1)
print("output:\n{}".format(output))

import torch
import torch.nn as nn
import numpy as np

# 假设我们有1x3x3x4的输入张量
img = np.array([[[[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]],
                    [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]],
                    [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]]])
input_tensor = torch.tensor(img, dtype=torch.float32)

# 3x1x3x3
kernel = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
kernel = torch.tensor(kernel, dtype=torch.float32)

# 4x3x1x1
kernel2 = np.array([[[[ 1 ]], [[ 2 ]], [[ 3 ]]], 
                    [[[ 1 ]], [[ 2 ]], [[ 3 ]]], 
                    [[[ 1 ]], [[ 2 ]], [[ 3 ]]],
                    [[[ 1 ]], [[ 2 ]], [[ 3 ]]]])
kernel2 = torch.tensor(kernel2, dtype=torch.float32)

# 第一步：深度卷积
# 因为我们只有3个输入通道，所以groups设为3
depthwise_conv = nn.Conv2d(in_channels=3, 
                           out_channels=3, 
                           kernel_size=3, 
                           stride=1, 
                           padding=1, 
                           groups=3,  # 设置groups参数实现深度卷积
                           bias=False)

depthwise_conv.weight.data = kernel
depthwise_output = depthwise_conv(input_tensor)

# 第二步：点卷积
# 这里假设我们要增加到4个输出通道
pointwise_conv = nn.Conv2d(in_channels=3, 
                            out_channels=4, 
                            kernel_size=1, 
                            stride=1, 
                            padding=0)

pointwise_conv.weight.data = kernel2
print("depthwise_output:\n", depthwise_output)
separable_output = pointwise_conv(depthwise_output)
print("pointwise_conv:\n", separable_output)
# depthwise_output:
#  tensor([[[[ 94., 154., 193., 130.],
#           [186., 285., 330., 213.],
#           [106., 154., 175., 106.]],

#          [[ 94., 154., 193., 130.],
#           [186., 285., 330., 213.],
#           [106., 154., 175., 106.]],

#          [[ 94., 154., 193., 130.],
#           [186., 285., 330., 213.],
#           [106., 154., 175., 106.]]]], grad_fn=<ConvolutionBackward0>)
# pointwise_conv:
#  tensor([[[[ 563.7503,  923.7503, 1157.7504,  779.7503],
#           [1115.7504, 1709.7504, 1979.7504, 1277.7504],
#           [ 635.7503,  923.7503, 1049.7504,  635.7503]],

#          [[ 563.8420,  923.8420, 1157.8420,  779.8420],
#           [1115.8420, 1709.8420, 1979.8420, 1277.8420],
#           [ 635.8420,  923.8420, 1049.8420,  635.8420]],

#          [[ 563.8658,  923.8658, 1157.8657,  779.8658],
#           [1115.8657, 1709.8657, 1979.8657, 1277.8657],
#           [ 635.8658,  923.8658, 1049.8657,  635.8658]],

#          [[ 563.4731,  923.4731, 1157.4730,  779.4731],
#           [1115.4730, 1709.4730, 1979.4730, 1277.4730],
#           [ 635.4731,  923.4731, 1049.4730,  635.4731]]]],
#        grad_fn=<ConvolutionBackward0>)