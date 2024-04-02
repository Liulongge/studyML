import numpy as np  
  
def batch_norm(x, gamma, beta, eps=1e-5):  
    # 获取输入张量的形状  
    N, C, H, W = x.shape  
  
    # channel方向计算均值和方差  
    mean = np.mean(x, axis=(0, 2, 3), keepdims=True)  
    var = np.var(x, axis=(0, 2, 3), keepdims=True)  
  
    # 归一化  
    x_hat = (x - mean) / np.sqrt(var + eps)  
  
    # 缩放和平移 
    gamma = gamma.reshape(1, C, 1, 1)  
    beta = beta.reshape(1, C, 1, 1)  
    y = gamma * x_hat + beta  
  
    return y  

batch_size = 2
channels = 3
height = 2
width = 2 
  
# 随机生成一个输入张量  
x = np.random.randn(batch_size, channels, height, width)  
  
# 初始化 gamma 和 beta 参数  
gamma = np.ones(channels)  
beta = np.zeros(channels)  
  
# 执行批归一化  
output = batch_norm(x, gamma, beta)  
# 输出归一化后的张量  
print("numpy, batchnorm:\n", output)

import torch  
import torch.nn as nn  
from torch import tensor
bn_layer = nn.BatchNorm2d(num_features=channels, eps=1e-5)
bn_layer.weight.data = torch.tensor(gamma, dtype=torch.float32)
bn_layer.bias.data = torch.tensor(beta, dtype=torch.float32)
output = bn_layer(torch.tensor(x, dtype=torch.float32))
print("torch, batchnorm:\n", output)
