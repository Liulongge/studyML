

# Sigmoid函数是一个在生物学中常见的S型函数，也称为S型生长曲线。 
# 在信息科学中，由于其单增以及反函数单增等性质，Sigmoid函数常被用作神经网络的激活函数，将变量映射到0,1之间。
# f(x) = 1 / (1 + e^(-x))

# 1、python 实现
import numpy as np

def sigmoid(x):
    """
    计算Sigmoid函数的值。
    参数: x (float): 输入值。
    返回: float: Sigmoid函数在x处的输出值。
    """
    return 1 / (1 + np.exp(-x))

# 示例：计算单个值的Sigmoid
value = 2.0
result = sigmoid(value)
print("输入:{}\n输出: {}".format(value, result))

# 示例：计算numpy数组的Sigmoid
array = np.array([-1.0, 0.0, 1.0])
sigmoid_array = sigmoid(array)
print("输入:{}\n输出: {}".format(array, sigmoid_array))

# 2、pytorch实现
import torch

# 假设有一个输入张量
input_tensor = torch.randn(2, 3)
# 直接应用Sigmoid函数
output_tensor = torch.sigmoid(input_tensor)
print("输入:{}\n输出: {}".format(input_tensor, output_tensor))