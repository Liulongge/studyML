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
# 创建nchw 1x1x1x1 kernel(输出图像channel为1)
kernel = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
# 创建nchw 1x3x1x1 kernel(输出图像channel为1)
kernel2 = np.array([[[[2]],
                    [[2]],
                    [[2]]]])

# output = convolve(image, kernel, padding=1, stride=1)

# print("input:\n{}".format(image))
# print("kernel:\n{}".format(kernel))
# print("output:\n{}".format(output))


import torch
import torch.nn as nn

# 创建一个输入数据
input_data = torch.tensor(image, dtype=torch.float32)

# 定义卷积层
conv_layer = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False)


conv_layer2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=0, groups=1, bias=False)



# 进行卷积运算
# 将定义好的卷积核赋值给卷积层的权重
kernel = torch.tensor(kernel, dtype=torch.float32)
conv_layer.weight.data = kernel
output_data = conv_layer(input_data)
print(output_data)

kernel = torch.tensor(kernel2, dtype=torch.float32)
conv_layer.weight.data = kernel
output_data2 = conv_layer(output_data)
print(output_data2)

print("input:\n{}".format(input_data))
print("kernel:\n{}".format(conv_layer.weight.data))
print("output:\n{}".format(output_data.data))