# 全连接层(Fully Connected Layer)
# 线性变换结果：y = W · x + b (m维向量)
# https://blog.csdn.net/qq_24951479/article/details/132564552

import numpy as np
 
def fully_connected(inputs, weights, bias):
    # 执行矩阵乘法
    outputs = np.dot(weights, inputs)
    # 添加偏置
    outputs += bias
    return outputs
 
# 示例
weights = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

input_data = np.array([[0],
                       [1],
                       [2],
                       [3],
                       [4],
                       [5],
                       [6],
                       [7],
                       [8],
                       [9]])
bias = np.array([[0],
                 [1],
                 [2],
                 [3],
                 [4]])


output = fully_connected(input_data, weights, bias)
print("numpy实现:\n", output)


import torch
import torch.nn as nn
from torch import tensor
 
# 定义全连接层
# 输入特征数为10，输出特征数为20，带有偏置
linear_layer = nn.Linear(in_features=10, out_features=5, bias=True)
 
# 在这里，输入向量通常是展平（flatten）后的特征，其维度为 [batch_size, in_features]
input_tensor = tensor(input_data, dtype=torch.float32).t()
linear_layer.weight.data = tensor(weights, dtype=torch.float32)
linear_layer.bias.data = tensor(bias, dtype=torch.float32).t()
output_tensor = linear_layer(input_tensor)
print(input_tensor.shape, linear_layer.weight.data.shape, linear_layer.bias.data.shape)
print("pytorch:\n", output_tensor.t())