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

import torch  
import torch.nn as nn  
from torch import tensor

# 假设输入是一个四维张量，大小为(batch_size, channels, height, width)  
# 例如: (1, 3, 32, 32) 表示一个批次的RGB图像，每张图像的大小为32x32  
input_tensor = torch.from_numpy(image).float()
  
# 创建一个空洞卷积层  
# 参数:  
#   in_channels: 输入通道数  
#   out_channels: 输出通道数  
#   kernel_size: 卷积核的大小  
#   dilation: 空洞卷积的扩张率  
atrous_conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, dilation=2, padding=2)  
atrous_conv.weight.data = torch.from_numpy(kernel).float()
# 将输入张量通过空洞卷积层  
output_tensor = atrous_conv(input_tensor)  
print("权重: ", atrous_conv.weight.data)
print(output_tensor)  # 输出应为 (1, 16, 32, 32)，因为空洞卷积不会改变输入的空间尺寸（当padding正确时）