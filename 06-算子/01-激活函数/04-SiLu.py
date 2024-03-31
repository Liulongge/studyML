# SiLU算子是一种激活函数，其公式定义为：SiLU(x) = x * sigmoid(x)，
# 其中sigmoid(x) = 1 / (1 + exp(-x)) 是常见的S型非线性函数，具备将线性关系转换为非线性关系的性质。
# SiLU激活函数也被称为Swish激活函数，是Sigmoid和ReLU的改进版。
# 它具有无上界有下界、平滑、非单调的特性，在深层模型上的效果优于ReLU，可以看做是平滑的ReLU激活函数。
# 在神经网络中，SiLU激活函数起到了非常重要的作用。它具有平滑的导数，在反向传播中可以有效地传递梯度，有利于网络的训练。
# 此外，SiLU函数在输入较大或较小的情况下，可以将输出值压缩到接近0或1，模拟生物神经元的激活过程。


# sigmoid
# https://zhuanlan.zhihu.com/p/669570231?utm_id=0

# silu
# https://blog.csdn.net/m0_63260018/article/details/131033318?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-131033318-blog-135061417.235^v43^pc_blog_bottom_relevance_base9&spm=1001.2101.3001.4242.1&utm_relevant_index=3


# 1、python 实现
import numpy as np

def sigmoid(x):
    """
    计算Sigmoid函数的值。
    参数: x (float): 输入值。
    返回: float: Sigmoid函数在x处的输出值。
    """
    return 1 / (1 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)

# 示例：计算单个值的Sigmoid
value = 2.0
result = silu(value)
print("python, silu, 输入:{}, 输出: {}".format(value, result))

import torch.nn as nn
from torch import tensor
silu_layer = nn.SiLU()

value = 2.0
result = silu_layer(tensor(value))
print("pytorch, silu, 输入:{}, 输出: {}".format(value, result))
