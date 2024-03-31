# https://blog.csdn.net/weixin_41513917/article/details/102514739
# https://blog.csdn.net/baicaiBC3/article/details/123380479

import numpy as np  
  
def max_pooling_2d(input_data, pool_size=(2, 2), stride=2):  
    _, _, ori_height, ori_width = input_data.shape  
    pad_w = int(ori_width % stride)
    pad_h = int(ori_height % stride)
    input_data = np.pad(input_data, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    # 获取输入数据的维度 
    batch_size, channels, height, width = input_data.shape  
    # 计算输出数据的维度  
    out_height = (height - pool_size[0]) // stride + 1  
    out_width = (width - pool_size[1]) // stride + 1  
    # 初始化输出数组  
    pooled_output = np.zeros((batch_size, channels, out_height, out_width))  

    # 遍历每个batch和每个channel  
    for b in range(batch_size):  
        for c in range(channels):  
            # 提取当前batch和channel的数据  
            data = input_data[b, c, :, :] 
            # 遍历池化窗口  
            for i in range(0, height, stride):  
                for j in range(0, width, stride):  
                    # 提取当前池化窗口的数据  
                    window = data[i:i+pool_size[0], j:j+pool_size[1]]  
                    # 计算当前池化窗口的最大值，并赋值给输出数组  
                    pooled_output[b, c, i//stride, j//stride] = np.average(window) 
      
    return pooled_output

image = np.array([[[[1, 2, 3, 4],
                    [4, 5, 6, 7],
                    [7, 8, 9, 10],
                    [7, 8, 9, 10]],

                    [[1, 2, 3, 4], 
                     [4, 5, 6, 7], 
                     [7, 8, 9, 10],
                     [7, 8, 9, 10]],

                    [[1, 2, 3, 4], 
                     [4, 5, 6, 7],
                     [7, 8, 9, 10], 
                     [7, 8, 9, 10]]]])
print("原始数据:\n", image)
out_put = max_pooling_2d(image, pool_size=(2, 2), stride=2)  
print("numpy输出:\n", out_put)


import torch  
import torch.nn as nn  
from torch import tensor

# 假设我们有一个输入张量，形状为 (batch_size, channels, height, width)  
# 在这个例子中，我们创建一个形状为 (1, 3, 32, 32) 的随机张量作为输入  
input_tensor = tensor(image, dtype=float)  
  
# 创建一个最大池化层，池化窗口大小为 2x2，步长为 2  
maxpool_layer = nn.AvgPool2d(kernel_size=2, stride=2)  
  
# 通过最大池化层传递输入张量  
output_tensor = maxpool_layer(input_tensor)  
  
# 输出张量的形状现在应该是 (1, 3, 16, 16)  
print("torch输出:\n", output_tensor)