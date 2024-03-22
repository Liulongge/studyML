# ReLu(Rectified Linear Unit)

# 线性整流函数(ReLU函数)可以用如下形式表示：
# f(x) = max(0, x)
# 其中x为输入值，f(x)为ReLU函数的输出值。可以看出，当输入值大于等于0时，ReLU函数的输出为输入值本身；而当输入值小于0时，ReLU函数

# 1、python 实现
import numpy as np
 
def relu(x):
    return np.maximum(0, x)
 
# 示例使用
x = np.array([-1, 2, -3, 4])
y = relu(x)
print(y)
# 输出将会是 [0, 2, 0, 4]


# 2、pytorch relu
import torch
import torch.nn as nn

# 创建一个ReLU实例
relu = nn.ReLU()
# 假设有一个输入张量
input_tensor = torch.randn(2, 3)  # 生成一个形状为(2, 3)的随机张量
# 应用ReLU激活函数
output_tensor = relu(input_tensor)
print("Input Tensor:")
print(input_tensor)
print("\nReLU Output Tensor:")
print(output_tensor)
# tensor([[-0.7143,  0.9522,  1.8683],
#         [-1.1039,  0.5947, -0.4799]])

# ReLU Output Tensor:
# tensor([[0.0000, 0.9522, 1.8683],
#         [0.0000, 0.5947, 0.0000]])